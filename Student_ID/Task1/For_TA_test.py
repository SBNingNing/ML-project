"""
Task 1: Binary Defect Classification - Testing Script (For TA)
用于助教测试的脚本
"""

import torch
import numpy as np
import os
import json
from PIL import Image
import argparse
import sys


# ==================== 手动实现的 MLP 模型（推理版本） ====================
class ManualMLP:
    """
    完全手动实现的多层感知机（仅推理）
    """
    
    def __init__(self, device='cpu'):
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.device = device
    
    def sigmoid(self, z):
        """Sigmoid 激活函数"""
        # 防止溢出 - 使用 torch.where 替代 torch.clamp
        z_safe = torch.where(z > 100, torch.ones_like(z) * 100, z)
        z_safe = torch.where(z_safe < -100, torch.ones_like(z_safe) * (-100), z_safe)
        return 1.0 / (1.0 + torch.exp(-z_safe))
    
    def relu(self, z):
        """ReLU 激活函数"""
        return torch.where(z > 0, z, torch.zeros_like(z))
    
    def forward(self, X):
        """
        前向传播（推理）
        X: (batch_size, input_size)
        返回: (batch_size, 1) 的概率值
        """
        # 确保输入在正确的设备上
        X = X.to(self.device)
        
        # 第一层：Linear + ReLU
        Z1 = torch.matmul(X, self.W1) + self.b1
        A1 = self.relu(Z1)
        
        # 第二层：Linear + Sigmoid
        Z2 = torch.matmul(A1, self.W2) + self.b2
        A2 = self.sigmoid(Z2)
        
        return A2
    
    def load_model(self, path):
        """加载模型参数"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到模型文件: {path}")
        
        model_dict = torch.load(path, map_location=self.device)
        self.W1 = model_dict['W1'].to(self.device)
        self.b1 = model_dict['b1'].to(self.device)
        self.W2 = model_dict['W2'].to(self.device)
        self.b2 = model_dict['b2'].to(self.device)
        
        print(f"模型加载成功: {path}")


# ==================== 数据预处理 ====================
def preprocess_image(img_path, img_size=64):
    """
    预处理单张图片（必须与训练时保持一致！）
    img_path: 图片路径
    img_size: 缩放尺寸（必须与训练时一致）
    
    返回: (1, input_size) 的 Tensor
    """
    # 读取图片
    img = Image.open(img_path).convert('RGB')
    
    # Resize（与训练时一致）
    img = img.resize((img_size, img_size))
    
    # 转为 numpy 数组并归一化
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # 展平为一维向量
    img_flat = img_array.flatten()  # (img_size*img_size*3,)
    
    # 转为 Tensor
    img_tensor = torch.from_numpy(img_flat).float().unsqueeze(0)  # (1, input_size)
    
    return img_tensor


# ==================== 测试主函数 ====================
def test(test_data_path, model_path='./model_weights.pth', img_size=64):
    """
    测试函数
    test_data_path: 测试数据路径（包含 img/ 文件夹）
    model_path: 训练好的模型路径
    img_size: 图片缩放尺寸（必须与训练时一致）
    """
    
    print("=" * 60)
    print("Task 1: Binary Defect Classification - 测试开始")
    print("=" * 60)
    
    # ===== GPU 设备检测 =====
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    if device == 'cuda':
        print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
    
    # 检查测试数据路径
    img_dir = os.path.join(test_data_path, 'img')
    if not os.path.exists(img_dir):
        print(f"错误: 找不到图片目录: {img_dir}")
        sys.exit(1)
    
    # 加载模型
    model = ManualMLP(device=device)
    try:
        model.load_model(model_path)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请先运行 main.py 训练模型！")
        sys.exit(1)
    
    # 获取所有图片文件
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    
    if len(img_files) == 0:
        print(f"错误: 在 {img_dir} 中没有找到任何 PNG 图片")
        sys.exit(1)
    
    print(f"找到 {len(img_files)} 张测试图片")
    print("正在进行预测...")
    
    # 预测结果字典
    results = {}
    
    # 逐张预测
    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        
        # 预处理图片
        img_tensor = preprocess_image(img_path, img_size=img_size)
        
        # 前向传播（推理）
        with torch.no_grad():
            pred_prob = model.forward(img_tensor)
            if device == 'cuda':
                pred_prob = pred_prob.cpu()
            pred_prob = pred_prob.item()
        
        # 二分类：概率 > 0.5 -> True (Defective), 否则 False (Non-defective)
        pred_label = pred_prob > 0.5
        
        # 提取文件名（不带后缀）
        base_name = os.path.splitext(img_file)[0]
        
        # 存储结果
        results[base_name] = pred_label
    
    # 获取 Leader ID（从当前目录的脚本名推断，或使用默认值）
    leader_id = 'PB23071385'
    
    # 保存 JSON 文件
    output_file = f"{leader_id}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n预测完成！")
    print(f"结果已保存到: {output_file}")
    print(f"共预测 {len(results)} 张图片")
    
    # 显示统计信息
    n_defective = sum(results.values())
    n_non_defective = len(results) - n_defective
    print(f"预测结果: 有缺陷 {n_defective} 张, 无缺陷 {n_non_defective} 张")
    
    return results


# ==================== 命令行入口 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Task 1: Binary Defect Classification - Testing')
    parser.add_argument('--test_data_path', type=str, required=True,
                        help='测试数据路径（包含 img/ 文件夹）')
    parser.add_argument('--model_path', type=str, default='./model_weights.pth',
                        help='训练好的模型路径（默认: ./model_weights.pth）')
    parser.add_argument('--img_size', type=int, default=64,
                        help='图片缩放尺寸，必须与训练时一致（默认: 64）')
    
    args = parser.parse_args()
    
    # 执行测试
    test(args.test_data_path, args.model_path, args.img_size)
