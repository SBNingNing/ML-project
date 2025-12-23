import os
import math
import torch
import torch.nn.functional as F

# ==================== 全局配置 ====================
class Config:
    # 路径配置：Task1 -> Student_ID -> ML-project-main -> dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', '..', 'dataset')
    
    # 训练超参数
    img_size = 128
    batch_size = 32
    lr = 0.0005
    epochs = 20
    
    # 判定阈值 (后续在 evaluate.py 中算出最佳值后，请修改这里)
    threshold = 0.85 
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 2025

# ==================== 手动实现的层 ====================
class Layer:
    def forward(self, x, training=True): raise NotImplementedError
    def backward(self, grad): raise NotImplementedError
    def get_params(self): return {}

class Conv2d(Layer):
    def __init__(self, in_c, out_c, k=3, p=1):
        self.k, self.p, self.in_c, self.out_c = k, p, in_c, out_c
        scale = math.sqrt(2.0 / (in_c * k * k))
        self.W = (torch.randn(out_c, in_c, k, k) * scale).to(Config.device)
        self.b = torch.zeros(out_c).to(Config.device)
        
        # 【修复点】初始化梯度为 None，防止 evaluate 时报错
        self.dW = None
        self.db = None
        
        # Adam 状态
        self.m_W, self.v_W = torch.zeros_like(self.W), torch.zeros_like(self.W)
        self.m_b, self.v_b = torch.zeros_like(self.b), torch.zeros_like(self.b)

    def forward(self, x, training=True):
        if training: self.cache = x
        return F.conv2d(x, self.W, self.b, padding=self.p)

    def backward(self, grad_output):
        x = self.cache
        N = x.shape[0]
        x_unf = F.unfold(x, self.k, padding=self.p)
        grad_view = grad_output.view(N, self.out_c, -1)
        dw = torch.matmul(grad_view, x_unf.transpose(1, 2))
        self.dW = torch.sum(dw, dim=0).view(self.out_c, self.in_c, self.k, self.k) / N
        self.db = torch.sum(grad_output, dim=(0, 2, 3)) / N
        return F.conv_transpose2d(grad_output, self.W, padding=self.p)
    
    def get_params(self):
        # 【修复点】安全返回参数字典
        params = {
            'W': self.W, 'b': self.b, 
            'm_W': self.m_W, 'v_W': self.v_W, 
            'm_b': self.m_b, 'v_b': self.v_b
        }
        # 只有在梯度存在时才返回，避免 NoneType 错误（虽然 save 时会过滤，但保持严谨）
        if self.dW is not None: params['dW'] = self.dW
        if self.db is not None: params['db'] = self.db
        return params

class Linear(Layer):
    def __init__(self, in_f, out_f):
        scale = math.sqrt(2.0 / in_f)
        self.W = (torch.randn(in_f, out_f) * scale).to(Config.device)
        self.b = torch.zeros(out_f).to(Config.device)
        
        # 【修复点】初始化梯度
        self.dW = None
        self.db = None
        
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
        # 【修复点】安全返回
        params = {
            'W': self.W, 'b': self.b,
            'm_W': self.m_W, 'v_W': self.v_W,
            'm_b': self.m_b, 'v_b': self.v_b
        }
        if self.dW is not None: params['dW'] = self.dW
        if self.db is not None: params['db'] = self.db
        return params

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
        out = torch.sigmoid(x)
        if training: self.cache = out
        return out
    def backward(self, grad): return grad * self.cache * (1.0 - self.cache)

class Sequential:
    def __init__(self, layers): self.layers = layers
    def forward(self, x, training=True):
        for l in self.layers: x = l.forward(x, training)
        return x
    def backward(self, grad):
        for l in reversed(self.layers): grad = l.backward(grad)
    def save(self, path):
        # 只保存权重，不保存梯度
        state = []
        for l in self.layers:
            params = l.get_params()
            if params:
                # 过滤掉 dW, db 等梯度 (以及 None 值)
                layer_state = {
                    k: v.cpu() for k,v in params.items() 
                    if not k.startswith('d') and v is not None
                }
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
                    # 只有当参数在当前层存在，且 v 不为 None 时才加载
                    if k in p and p[k] is not None: 
                        p[k].data = v.to(Config.device)

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
                # Adam 只需要 W, b 及其对应的 dW, db
                # 需要确保这些键存在且不为 None
                if f'd{n}' in p and p[f'd{n}'] is not None:
                    g = p[f'd{n}']
                    m, v, w = p[f'm_{n}'], p[f'v_{n}'], p[n]
                    
                    m.mul_(self.b1).add_(g, alpha=1-self.b1)
                    v.mul_(self.b2).add_(g**2, alpha=1-self.b2)
                    m_h, v_h = m/(1-self.b1**self.t), v/(1-self.b2**self.t)
                    w.sub_(m_h * self.lr / (torch.sqrt(v_h) + self.eps))

def build_model():
    return Sequential([
        Conv2d(3, 32), ReLU(), Conv2d(32, 32), ReLU(), MaxPool2d(),
        Conv2d(32, 64), ReLU(), Conv2d(64, 64), ReLU(), MaxPool2d(),
        Conv2d(64, 128), ReLU(), Conv2d(128, 128), ReLU(), MaxPool2d(),
        Conv2d(128, 256), ReLU(), Conv2d(256, 256), ReLU(), MaxPool2d(),
        Flatten(), Linear(256*8*8, 512), ReLU(), Linear(512, 1), Sigmoid()
    ])
