"""
Task 1: Binary Defect Classification
"""

import torch
import numpy as np
import os
import json
from PIL import Image
import random
from collections import Counter


# ==================== 手动实现的 MLP 模型 ====================
class ManualMLP:
    """
    完全手动实现的多层感知机
    不使用 autograd、torch.optim 和 torch.nn 的自动层
    """
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        """
        初始化模型参数
        input_size: 输入维度
        hidden_size: 隐藏层维度
        output_size: 输出维度 (1 for binary classification)
        """
        self.lr = learning_rate
        
        # Xavier 初始化权重
        # 第一层：input -> hidden
        scale1 = torch.sqrt(torch.Tensor([2.0 / (input_size + hidden_size)]))
        self.W1 = torch.randn(input_size, hidden_size) * scale1
        self.b1 = torch.zeros(hidden_size)
        
        # 第二层：hidden -> output
        scale2 = torch.sqrt(torch.Tensor([2.0 / (hidden_size + output_size)]))
        self.W2 = torch.randn(hidden_size, output_size) * scale2
        self.b2 = torch.zeros(output_size)
        
        # 缓存中间结果（用于反向传播）
        self.cache = {}
    
    def sigmoid(self, z):
        """Sigmoid 激活函数"""
        # 防止溢出
        return 1.0 / (1.0 + torch.exp(-torch.clamp(z, -100, 100)))
    
    def relu(self, z):
        """ReLU 激活函数"""
        return torch.where(z > 0, z, torch.zeros_like(z))
    
    def forward(self, X):
        """
        前向传播
        X: (batch_size, input_size)
        返回: (batch_size, 1) 的概率值
        """
        # 第一层：Linear + ReLU
        Z1 = torch.matmul(X, self.W1) + self.b1  # (batch, hidden)
        A1 = self.relu(Z1)
        
        # 第二层：Linear + Sigmoid
        Z2 = torch.matmul(A1, self.W2) + self.b2  # (batch, 1)
        A2 = self.sigmoid(Z2)
        
        # 缓存中间结果
        self.cache = {
            'X': X,
            'Z1': Z1,
            'A1': A1,
            'Z2': Z2,
            'A2': A2
        }
        
        return A2
    
    def backward(self, Y):
        """
        手动反向传播
        Y: 真实标签 (batch_size, 1)
        """
        batch_size = Y.shape[0]
        
        # 取出缓存的中间结果
        X = self.cache['X']
        Z1 = self.cache['Z1']
        A1 = self.cache['A1']
        Z2 = self.cache['Z2']
        A2 = self.cache['A2']
        
        # ===== 输出层反向传播 =====
        # BCE Loss 对 Z2 的导数：dL/dZ2 = A2 - Y
        dZ2 = A2 - Y  # (batch, 1)
        
        # 计算 W2 和 b2 的梯度
        dW2 = torch.matmul(A1.transpose(0, 1), dZ2) / batch_size  # (hidden, 1)
        db2 = torch.sum(dZ2, dim=0) / batch_size  # (1,)
        
        # ===== 隐藏层反向传播 =====
        # 传播到 A1: dL/dA1 = dZ2 * W2^T
        dA1 = torch.matmul(dZ2, self.W2.transpose(0, 1))  # (batch, hidden)
        
        # ReLU 的导数：dA1/dZ1 = 1 if Z1 > 0 else 0
        dZ1 = dA1 * torch.where(Z1 > 0, torch.ones_like(Z1), torch.zeros_like(Z1))
        
        # 计算 W1 和 b1 的梯度
        dW1 = torch.matmul(X.transpose(0, 1), dZ1) / batch_size  # (input, hidden)
        db1 = torch.sum(dZ1, dim=0) / batch_size  # (hidden,)
        
        # 存储梯度
        self.gradients = {
            'dW1': dW1,
            'db1': db1,
            'dW2': dW2,
            'db2': db2
        }
    
    def update_parameters(self):
        """手动更新参数（SGD）"""
        self.W1 = self.W1 - self.lr * self.gradients['dW1']
        self.b1 = self.b1 - self.lr * self.gradients['db1']
        self.W2 = self.W2 - self.lr * self.gradients['dW2']
        self.b2 = self.b2 - self.lr * self.gradients['db2']
    
    def compute_loss(self, Y_pred, Y_true, pos_weight=1.0):
        """
        计算 BCE Loss（带类别权重）
        pos_weight: 正样本（缺陷）的权重
        """
        epsilon = 1e-7  # 防止 log(0)
        Y_pred = torch.clamp(Y_pred, epsilon, 1 - epsilon)
        
        # 加权 BCE Loss
        loss_pos = -pos_weight * Y_true * torch.log(Y_pred)
        loss_neg = -(1 - Y_true) * torch.log(1 - Y_pred)
        loss = torch.mean(loss_pos + loss_neg)
        
        return loss
    
    def save_model(self, path):
        """保存模型参数"""
        model_dict = {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'lr': self.lr
        }
        torch.save(model_dict, path)
        print(f"模型已保存到: {path}")
    
    def load_model(self, path):
        """加载模型参数"""
        model_dict = torch.load(path)
        self.W1 = model_dict['W1']
        self.b1 = model_dict['b1']
        self.W2 = model_dict['W2']
        self.b2 = model_dict['b2']
        self.lr = model_dict['lr']
        print(f"模型已加载: {path}")


# ==================== 数据加载和预处理 ====================
def load_dataset(data_dir, img_size=64):
    """
    加载数据集
    data_dir: 数据集根目录（包含 img/ 和可选的 label/ 文件夹）
    img_size: 缩放后的图片尺寸（建议 64 或 128，原图 320 太大）
    
    返回:
        images: (N, img_size*img_size*3) 的 numpy 数组
        labels: (N,) 的 numpy 数组（0 或 1）
        filenames: 文件名列表
    """
    img_dir = os.path.join(data_dir, 'img')
    label_dir = os.path.join(data_dir, 'label')
    
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"找不到图片目录: {img_dir}")
    
    images = []
    labels = []
    filenames = []
    
    # 遍历所有 PNG 图片
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    
    print(f"正在加载数据集，共 {len(img_files)} 张图片...")
    
    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        
        # 读取图片
        img = Image.open(img_path).convert('RGB')
        
        # Resize（降低维度）
        img = img.resize((img_size, img_size))
        
        # 转为 numpy 数组并归一化
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # 展平为一维向量
        img_flat = img_array.flatten()  # (img_size*img_size*3,)
        
        images.append(img_flat)
        
        # 判断标签：检查对应的 txt 文件是否存在
        base_name = os.path.splitext(img_file)[0]
        label_file = os.path.join(label_dir, base_name + '.txt')
        
        if os.path.exists(label_file):
            label = 1  # Defective
        else:
            label = 0  # Non-defective
        
        labels.append(label)
        filenames.append(base_name)
    
    images = np.array(images)
    labels = np.array(labels)
    
    # 统计类别分布
    counter = Counter(labels)
    print(f"数据加载完成: {len(images)} 张图片")
    print(f"类别分布 - 无缺陷: {counter[0]}, 有缺陷: {counter[1]}")
    
    return images, labels, filenames


def split_dataset(images, labels, filenames, val_ratio=0.2, random_seed=42):
    """
    划分训练集和验证集
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    n = len(images)
    indices = list(range(n))
    random.shuffle(indices)
    
    split_idx = int(n * (1 - val_ratio))
    
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    train_images = images[train_idx]
    train_labels = labels[train_idx]
    train_files = [filenames[i] for i in train_idx]
    
    val_images = images[val_idx]
    val_labels = labels[val_idx]
    val_files = [filenames[i] for i in val_idx]
    
    return (train_images, train_labels, train_files), (val_images, val_labels, val_files)


# ==================== 评价指标 ====================
def compute_metrics(y_true, y_pred_probs, threshold=0.5):
    """
    计算 Precision, Recall, F1-score
    y_true: 真实标签 (numpy array)
    y_pred_probs: 预测概率 (numpy array)
    threshold: 分类阈值
    """
    y_pred = (y_pred_probs >= threshold).astype(int)
    
    # True Positive, False Positive, False Negative, True Negative
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    
    # Precision
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    
    # Recall
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    
    # F1-score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN
    }


# ==================== 训练主函数 ====================
def train():
    """训练主函数"""
    
    # ===== 超参数设置 =====
    IMG_SIZE = 64  # 图片缩放尺寸（原图 320 太大，建议用 64 或 128）
    HIDDEN_SIZE = 128  # 隐藏层神经元数量
    LEARNING_RATE = 0.001  # 学习率
    EPOCHS = 50  # 训练轮数
    BATCH_SIZE = 32  # 批次大小
    POS_WEIGHT = 3.0  # 正样本（缺陷）权重（用于处理类别不平衡）
    
    # 数据路径（请根据实际情况修改）
    DATA_DIR = './data'  # 假设数据在 Task1/data/ 下
    MODEL_SAVE_PATH = './model_weights.pth'
    
    print("=" * 60)
    print("Task 1: Binary Defect Classification - 训练开始")
    print("=" * 60)
    
    # ===== 加载数据 =====
    try:
        images, labels, filenames = load_dataset(DATA_DIR, img_size=IMG_SIZE)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请确保数据目录存在，并包含 img/ 和 label/ 文件夹")
        return
    
    # 划分训练集和验证集
    (train_images, train_labels, train_files), \
    (val_images, val_labels, val_files) = split_dataset(images, labels, filenames)
    
    print(f"训练集: {len(train_images)} 张, 验证集: {len(val_images)} 张")
    
    # 转为 Tensor
    X_train = torch.from_numpy(train_images).float()
    Y_train = torch.from_numpy(train_labels).float().unsqueeze(1)  # (N, 1)
    
    X_val = torch.from_numpy(val_images).float()
    Y_val = torch.from_numpy(val_labels).float().unsqueeze(1)
    
    # ===== 初始化模型 =====
    input_size = IMG_SIZE * IMG_SIZE * 3  # 64*64*3 = 12288
    model = ManualMLP(input_size=input_size, 
                      hidden_size=HIDDEN_SIZE, 
                      output_size=1, 
                      learning_rate=LEARNING_RATE)
    
    print(f"\n模型结构:")
    print(f"  输入层: {input_size}")
    print(f"  隐藏层: {HIDDEN_SIZE} (ReLU)")
    print(f"  输出层: 1 (Sigmoid)")
    print(f"  总参数量: {input_size * HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE + 1}")
    
    # ===== 训练循环 =====
    print(f"\n开始训练（共 {EPOCHS} 轮）...")
    print("-" * 60)
    
    best_f1 = 0.0
    n_train = len(X_train)
    
    for epoch in range(EPOCHS):
        # 随机打乱训练数据
        perm = torch.randperm(n_train)
        X_train_shuffled = X_train[perm]
        Y_train_shuffled = Y_train[perm]
        
        # Mini-batch 训练
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, n_train, BATCH_SIZE):
            # 取一个 batch
            X_batch = X_train_shuffled[i:i+BATCH_SIZE]
            Y_batch = Y_train_shuffled[i:i+BATCH_SIZE]
            
            # 前向传播
            Y_pred = model.forward(X_batch)
            
            # 计算损失
            loss = model.compute_loss(Y_pred, Y_batch, pos_weight=POS_WEIGHT)
            epoch_loss += loss.item()
            n_batches += 1
            
            # 反向传播
            model.backward(Y_batch)
            
            # 更新参数
            model.update_parameters()
        
        # 计算平均损失
        avg_loss = epoch_loss / n_batches
        
        # 在验证集上评估
        with torch.no_grad():
            val_pred = model.forward(X_val)
            val_loss = model.compute_loss(val_pred, Y_val, pos_weight=POS_WEIGHT)
            
            # 计算评价指标
            metrics = compute_metrics(
                val_labels, 
                val_pred.numpy().flatten(), 
                threshold=0.5
            )
        
        # 打印训练信息
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] "
                  f"Train Loss: {avg_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"F1: {metrics['f1']:.4f} | "
                  f"Precision: {metrics['precision']:.4f} | "
                  f"Recall: {metrics['recall']:.4f}")
        
        # 保存最佳模型
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            model.save_model(MODEL_SAVE_PATH)
    
    print("-" * 60)
    print(f"训练完成！最佳 F1-score: {best_f1:.4f}")
    print(f"模型已保存到: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
