"""
Task 1: Deep CNN with Full Training Set Strategy
File: main2.0.py
Structure:
  ML-project-main/
    dataset/            <-- 数据集
    Student_ID/
      Task1/
        main.py      <-- 本脚本
"""
import os
import math
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm  # 导入进度条库
from PIL import Image, ImageEnhance
from sklearn.metrics import f1_score, precision_score, recall_score

# 解决某些环境下的 OpenMP 报错
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==================== 0. 全局配置 ====================
class Config:
    # 1. 获取当前脚本所在目录 (.../Student_ID/Task1)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. 【核心修改】向上跳两级找到 dataset
    # Task1 -> Student_ID -> ML-project-main -> dataset
    data_dir = os.path.join(current_dir, '..', '..', 'dataset')
    
    # 训练超参数
    img_size = 128       
    batch_size = 32      
    lr = 0.0005          
    epochs = 20          
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_path = os.path.join(current_dir, 'model_final.pth')
    seed = 2025

# 【关键修复】实例化配置类，并赋值给全局变量 cfg
cfg = Config()

# 检查路径是否存在，打印调试信息
if os.path.exists(os.path.join(cfg.data_dir, 'train', 'img')):
    print(f"[Config] 成功定位数据集: {os.path.abspath(cfg.data_dir)}")
else:
    print(f"[Error] 找不到数据集路径: {os.path.abspath(cfg.data_dir)}")
    print("请检查目录结构是否为: ML-project-main/dataset 和 ML-project-main/Student_ID/Task1/main2.0.py")
    # 不要直接报错退出，防止IDE误报，但后续会失败
    
print(f"[Config] Device: {cfg.device}")

# 设置随机种子
random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)#疑似没有在白名单？

# ==================== 1. 核心模块 (手动层实现) ====================

class Layer:
    def forward(self, x, training=True): raise NotImplementedError
    def backward(self, grad): raise NotImplementedError
    def get_params(self): return {}

class Conv2d(Layer):
    """手动卷积层：利用 Unfold (Im2Col) 加速梯度计算"""
    def __init__(self, in_c, out_c, k=3, p=1):
        self.k, self.p, self.in_c, self.out_c = k, p, in_c, out_c
        # Kaiming Initialization
        scale = math.sqrt(2.0 / (in_c * k * k))
        self.W = (torch.randn(out_c, in_c, k, k) * scale).to(cfg.device)
        self.b = torch.zeros(out_c).to(cfg.device)
        # Adam Cache
        self.m_W, self.v_W = torch.zeros_like(self.W), torch.zeros_like(self.W)
        self.m_b, self.v_b = torch.zeros_like(self.b), torch.zeros_like(self.b)

    def forward(self, x, training=True):
        if training: self.cache = x
        return F.conv2d(x, self.W, self.b, padding=self.p)

    def backward(self, grad_output):
        x = self.cache
        N = x.shape[0]
        # dW: Matmul(Grad_view, Input_unfold.T)
        x_unf = F.unfold(x, self.k, padding=self.p)
        grad_view = grad_output.view(N, self.out_c, -1)
        dw = torch.matmul(grad_view, x_unf.transpose(1, 2))
        self.dW = torch.sum(dw, dim=0).view(self.out_c, self.in_c, self.k, self.k) / N
        # db
        self.db = torch.sum(grad_output, dim=(0, 2, 3)) / N
        # dX
        return F.conv_transpose2d(grad_output, self.W, padding=self.p)
    
    def get_params(self):
        return {'W': self.W, 'b': self.b, 'dW': self.dW, 'db': self.db, 
                'm_W': self.m_W, 'v_W': self.v_W, 'm_b': self.m_b, 'v_b': self.v_b}

class Linear(Layer):
    def __init__(self, in_f, out_f):
        scale = math.sqrt(2.0 / in_f)
        self.W = (torch.randn(in_f, out_f) * scale).to(cfg.device)
        self.b = torch.zeros(out_f).to(cfg.device)
        self.m_W, self.v_W = torch.zeros_like(self.W), torch.zeros_like(self.W)
        self.m_b, self.v_b = torch.zeros_like(self.b), torch.zeros_like(self.b)

    def forward(self, x, training=True):
        if training: self.cache = x
        return torch.matmul(x, self.W) + self.b

    def backward(self, grad_output):
        x = self.cache
        self.dW = torch.matmul(x.transpose(0, 1), grad_output) / x.shape[0]
        self.db = torch.sum(grad_output, dim=0) / x.shape[0]
        return torch.matmul(grad_output, self.W.transpose(0, 1))

    def get_params(self):
        return {'W': self.W, 'b': self.b, 'dW': self.dW, 'db': self.db,
                'm_W': self.m_W, 'v_W': self.v_W, 'm_b': self.m_b, 'v_b': self.v_b}

class ReLU(Layer):
    def forward(self, x, training=True):
        if training: self.cache = x
        return F.relu(x)
    def backward(self, grad): return grad * (self.cache > 0).float()

class MaxPool2d(Layer):
    def __init__(self, k=2, s=2): self.k, self.s = k, s
    def forward(self, x, training=True):
        out, idx = F.max_pool2d(x, self.k, self.s, return_indices=True)
        if training: self.cache = (x.shape, idx)
        return out
    def backward(self, grad):
        shape, idx = self.cache
        return F.max_unpool2d(grad, idx, self.k, self.s, output_size=shape)

class Flatten(Layer):
    def forward(self, x, training=True):
        if training: self.cache = x.shape
        return x.view(x.size(0), -1)
    def backward(self, grad): return grad.view(self.cache)

class Sigmoid(Layer):
    def forward(self, x, training=True):
        out = torch.sigmoid(x)#疑似不在白名单？
        if training: self.cache = out
        return out
    def backward(self, grad): return grad * self.cache * (1.0 - self.cache)

# ==================== 2. 模型容器与优化器 ====================

class Sequential:
    def __init__(self, layers): self.layers = layers
    def forward(self, x, training=True):
        for l in self.layers: x = l.forward(x, training)
        return x
    def backward(self, grad):
        for l in reversed(self.layers): grad = l.backward(grad)
    def save(self, path):
        # 仅保存权重数据，转为CPU tensor保存
        state = []
        for l in self.layers:
            params = l.get_params()
            if params:
                # 过滤掉 dW, db 等梯度，只存权重 W, b 和 Adam 状态 m, v
                layer_state = {k: v.cpu() for k,v in params.items() if not k.startswith('d')}
                state.append(layer_state)
            else:
                state.append(None)
        torch.save(state, path)
    
    def load(self, path):
        if not os.path.exists(path): return
        state = torch.load(path)
        for l, s in zip(self.layers, state):
            if s: 
                p = l.get_params()
                for k, v in s.items(): 
                    if k in p: p[k].data = v.to(cfg.device)

class AdamOptimizer:
    def __init__(self, model, lr=0.001):
        self.model = model
        self.lr = lr
        self.b1, self.b2, self.eps, self.t = 0.9, 0.999, 1e-8, 0
    def step(self):
        self.t += 1
        for l in self.model.layers:
            p = l.get_params()
            if not p: continue
            for n in ['W', 'b']:
                g, m, v, w = p[f'd{n}'], p[f'm_{n}'], p[f'v_{n}'], p[n]
                m.mul_(self.b1).add_(g, alpha=1-self.b1)
                v.mul_(self.b2).add_(g**2, alpha=1-self.b2)
                m_h, v_h = m/(1-self.b1**self.t), v/(1-self.b2**self.t)
                w.sub_(m_h * self.lr / (torch.sqrt(v_h) + self.eps))

# ==================== 3. 构建 10 层模型 ====================
def build_10_layer_model():
    return Sequential([
        # Block 1: 128 -> 64
        Conv2d(3, 32), ReLU(), Conv2d(32, 32), ReLU(), MaxPool2d(),
        # Block 2: 64 -> 32
        Conv2d(32, 64), ReLU(), Conv2d(64, 64), ReLU(), MaxPool2d(),
        # Block 3: 32 -> 16
        Conv2d(64, 128), ReLU(), Conv2d(128, 128), ReLU(), MaxPool2d(),
        # Block 4: 16 -> 8
        Conv2d(128, 256), ReLU(), Conv2d(256, 256), ReLU(), MaxPool2d(),
        # FC
        Flatten(), Linear(256*8*8, 512), ReLU(), Linear(512, 1), Sigmoid()
    ])

# ==================== 4. 数据策略 (重点) ====================

def get_balanced_train_paths(data_dir):
    """
    仅加载 dataset/train。
    """
    train_dir = os.path.join(data_dir, 'train')
    img_dir = os.path.join(train_dir, 'img')
    txt_dir = os.path.join(train_dir, 'txt')
    
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Train directory not found: {img_dir}")
        
    all_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    pos_samples = []
    neg_samples = []
    
    for f in all_files:
        path = os.path.join(img_dir, f)
        # 检查同名 txt 文件
        has_label = os.path.exists(os.path.join(txt_dir, f.replace('.png', '.txt')))
        if has_label:
            pos_samples.append((path, 1))
        else:
            neg_samples.append((path, 0))
            
    print(f"[Raw Stats] Pos: {len(pos_samples)}, Neg: {len(neg_samples)}")
    
    # 策略执行
    pos_multiplier = 4  # 正样本增强倍数
    target_neg_ratio = 1.2 # 负样本保留比例
    
    # 1. 扩充正样本
    balanced_pos = pos_samples * pos_multiplier
    
    # 2. 降采样负样本
    target_neg_count = int(len(balanced_pos) * target_neg_ratio)
    if len(neg_samples) > target_neg_count:
        random.shuffle(neg_samples) # 打乱后截取
        balanced_neg = neg_samples[:target_neg_count]
    else:
        balanced_neg = neg_samples
    
    final_data = balanced_pos + balanced_neg
    random.shuffle(final_data)
    
    print(f"[Balanced Stats] Pos (x{pos_multiplier}): {len(balanced_pos)}, "
          f"Neg (Sampled): {len(balanced_neg)}, Total: {len(final_data)}")
    
    return final_data

def get_test_paths(data_dir):
    """加载 dataset/test"""
    test_dir = os.path.join(data_dir, 'test')
    img_dir = os.path.join(test_dir, 'img')
    txt_dir = os.path.join(test_dir, 'txt')
    
    data = []
    if os.path.exists(img_dir):
        for f in os.listdir(img_dir):
            if f.endswith('.png'):
                lbl = 1 if os.path.exists(os.path.join(txt_dir, f.replace('.png', '.txt'))) else 0
                data.append((os.path.join(img_dir, f), lbl))
    return data

def data_loader(data_list, batch_size, training=True):
    """
    数据生成器
    Training=True: 开启强力增强 (Flip, Rotate, Brightness, Contrast)
    Training=False: 仅 Resize
    """
    random.shuffle(data_list)
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i+batch_size]
        X, Y = [], []
        for path, label in batch:
            try:
                img = Image.open(path).convert('RGB').resize((cfg.img_size, cfg.img_size))
                
                if training:
                    # === 强力数据增强 ===
                    # 1. 几何变换
                    if random.random() > 0.5: img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    if random.random() > 0.5: img = img.transpose(Image.FLIP_TOP_BOTTOM)
                    rot = random.choice([None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270])
                    if rot: img = img.transpose(rot)
                    
                    # 2. 光照/颜色变换
                    if random.random() > 0.3:
                        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
                    if random.random() > 0.3:
                        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.4))
                
                arr = np.array(img, dtype=np.float32) / 255.0
                arr = arr.transpose(2, 0, 1) # HWC -> CHW
                X.append(arr)
                Y.append(label)
            except: pass
            
        if X:
            yield torch.tensor(np.array(X)).float().to(cfg.device), \
                  torch.tensor(np.array(Y)).float().unsqueeze(1).to(cfg.device)

# ==================== 5. 训练与测试流程 ====================

def train():
    print("\n>>> Phase 1: Training on Full 'dataset/train' Set")
    
    # 1. 获取平衡后的全量训练数据
    train_data = get_balanced_train_paths(cfg.data_dir)
    
    # 计算一轮有多少个 Batch，用于进度条显示
    # 注意：如果数据量不能整除 Batch Size，最后一批可能会少一点，但大致准确
    steps_per_epoch = len(train_data) // cfg.batch_size
    
    # 2. 初始化
    model = build_10_layer_model()
    optimizer = AdamOptimizer(model, lr=cfg.lr)
    
    #3.跟踪最佳loss
    best_loss = float('inf')
    
    print(f"Starting training for {cfg.epochs} epochs on {cfg.device}...")
    
    for epoch in range(cfg.epochs):
        model_loss = 0
        steps = 0
        
        # === 核心修改：使用 tqdm 包装 data_loader ===
        # desc: 进度条左边的文字
        # total: 总步数，用于计算剩余时间
        pbar = tqdm(data_loader(train_data, cfg.batch_size, training=True), 
                    total=steps_per_epoch, 
                    desc=f"Epoch {epoch+1}/{cfg.epochs}",
                    ncols=100) # 限制进度条宽度，防止换行
        
        for X, Y in pbar:
            # Forward
            pred = model.forward(X, training=True)
            
            # Loss
            pred = torch.clamp(pred, 1e-7, 1-1e-7)#疑似不在白名单？
            weights = torch.ones_like(Y)
            weights[Y==1] = 1.2 
            
            loss = -(weights * (Y * torch.log(pred) + (1-Y) * torch.log(1-pred))).mean()
            current_loss = loss.item() # 获取当前 batch 的 loss
            model_loss += current_loss
            steps += 1
            
            # Backward
            grad = -(weights * (Y/pred - (1-Y)/(1-pred))) / X.shape[0]
            model.backward(grad)
            optimizer.step()
            
            # === 核心修改：实时更新进度条右边的 Loss ===
            # 显示当前 batch 的 loss 和累计平均 loss
            avg_loss_so_far = model_loss / steps
            pbar.set_postfix({'loss': f"{current_loss:.4f}", 'avg': f"{avg_loss_so_far:.4f}"})
        
        # 这一轮跑完后的最终平均 Loss
        avg_loss = model_loss / steps if steps > 0 else 0
        # 进度条结束后，为了日志整洁，可以手动打印一行总结（可选，因为 tqdm 已经显示了）
        # print(f"Epoch {epoch+1} Finished. Avg Loss: {avg_loss:.4f}")
        
        # 每 5 轮保存一次
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(cfg.current_dir, f'model_epoch_{epoch+1}.pth')
            model.save(ckpt_path)
        # 只要更小就覆盖 best_loss_model.pth
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_loss_path = os.path.join(cfg.current_dir, 'best_loss_model.pth')
            model.save(best_loss_path)

    # 保存最终模型
    model.save(cfg.save_path)
    print(f"\nTraining Finished. Final model saved to: {cfg.save_path}")
'''    
def evaluate():
    """
    测试阶段：加载 'dataset/test' 并计算指标。
    """
    print("\n>>> Phase 2: Evaluating on 'dataset/test' Set")
    
    if not os.path.exists(cfg.save_path):
        print("Model not found. Please train first.")
        return

    # 加载模型
    model = build_10_layer_model()
    model.load(cfg.save_path)
    
    # 加载测试数据 (无增强，无平衡，真实分布)
    test_data = get_test_paths(cfg.data_dir)
    if not test_data:
        print("[Warning] No test data found. Skipping evaluation.")
        return

    print(f"Test samples: {len(test_data)}")
    
    preds, targets = [], []
    
    # 推理
    for X, Y in data_loader(test_data, cfg.batch_size, training=False):
        out = model.forward(X, training=False)
        preds.extend(out.cpu().numpy().flatten())
        targets.extend(Y.cpu().numpy().flatten())
        
    preds = np.array(preds)
    targets = np.array(targets)
    
    print("\n--- Performance Report (Threshold Searching) ---")
    best_f1 = 0
    best_thresh = 0.5
    
    for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
        p_label = (preds > t).astype(int)
        f1 = f1_score(targets, p_label)
        prec = precision_score(targets, p_label, zero_division=0)
        rec = recall_score(targets, p_label, zero_division=0)
        print(f"Thresh {t} -> F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            
    print(f"\nFinal Result (Best F1): {best_f1:.4f} at Threshold {best_thresh}")
    print("Tip: Remember this threshold for your submission script!")
'''
if __name__ == "__main__":
    train()
    #evaluate()
