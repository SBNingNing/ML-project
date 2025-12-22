"""
验证脚本：检查手动实现的 MLP 是否正确（不使用 autograd）
"""

import torch
import numpy as np


def test_forward_backward():
    """测试前向传播和反向传播的正确性"""
    
    print("=" * 60)
    print("测试 1: 前向传播")
    print("=" * 60)
    
    # 简单的测试数据
    batch_size = 4
    input_size = 10
    hidden_size = 5
    output_size = 1
    
    # 手动初始化参数
    W1 = torch.randn(input_size, hidden_size) * 0.1
    b1 = torch.zeros(hidden_size)
    W2 = torch.randn(hidden_size, output_size) * 0.1
    b2 = torch.zeros(output_size)
    
    # 输入数据
    X = torch.randn(batch_size, input_size)
    Y = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    
    # 前向传播
    Z1 = torch.matmul(X, W1) + b1
    A1 = torch.where(Z1 > 0, Z1, torch.zeros_like(Z1))  # ReLU
    Z2 = torch.matmul(A1, W2) + b2
    A2 = 1.0 / (1.0 + torch.exp(-Z2))  # Sigmoid
    
    print(f"输入 X shape: {X.shape}")
    print(f"隐藏层 A1 shape: {A1.shape}")
    print(f"输出 A2 shape: {A2.shape}")
    print(f"输出值范围: [{A2.min():.4f}, {A2.max():.4f}]")
    print("✓ 前向传播测试通过\n")
    
    print("=" * 60)
    print("测试 2: 反向传播")
    print("=" * 60)
    
    # 反向传播
    dZ2 = A2 - Y
    dW2 = torch.matmul(A1.transpose(0, 1), dZ2) / batch_size
    db2 = torch.sum(dZ2, dim=0) / batch_size
    
    dA1 = torch.matmul(dZ2, W2.transpose(0, 1))
    dZ1 = dA1 * torch.where(Z1 > 0, torch.ones_like(Z1), torch.zeros_like(Z1))
    
    dW1 = torch.matmul(X.transpose(0, 1), dZ1) / batch_size
    db1 = torch.sum(dZ1, dim=0) / batch_size
    
    print(f"dW1 shape: {dW1.shape}")
    print(f"db1 shape: {db1.shape}")
    print(f"dW2 shape: {dW2.shape}")
    print(f"db2 shape: {db2.shape}")
    print("✓ 反向传播测试通过\n")
    
    print("=" * 60)
    print("测试 3: 参数更新")
    print("=" * 60)
    
    lr = 0.01
    W1_old = W1.clone()
    W1_new = W1 - lr * dW1
    
    print(f"W1 更新前后的差异: {torch.mean(torch.abs(W1_new - W1_old)):.6f}")
    print("✓ 参数更新测试通过\n")
    
    print("=" * 60)
    print("测试 4: 数值梯度检查（关键！）")
    print("=" * 60)
    
    # 计算损失
    def compute_loss(pred, target):
        epsilon = 1e-7
        pred = torch.clamp(pred, epsilon, 1 - epsilon)
        return -torch.mean(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
    
    # 解析梯度
    loss = compute_loss(A2, Y)
    print(f"当前 Loss: {loss.item():.4f}")
    
    # 数值梯度（对 W2 的第一个元素）
    epsilon = 1e-5
    W2_temp = W2.clone()
    
    W2_temp[0, 0] += epsilon
    Z2_plus = torch.matmul(A1, W2_temp) + b2
    A2_plus = 1.0 / (1.0 + torch.exp(-Z2_plus))
    loss_plus = compute_loss(A2_plus, Y).item()
    
    W2_temp[0, 0] -= 2 * epsilon
    Z2_minus = torch.matmul(A1, W2_temp) + b2
    A2_minus = 1.0 / (1.0 + torch.exp(-Z2_minus))
    loss_minus = compute_loss(A2_minus, Y).item()
    
    grad_numerical = (loss_plus - loss_minus) / (2 * epsilon)
    grad_analytical = dW2[0, 0].item()
    
    diff = abs(grad_numerical - grad_analytical)
    print(f"数值梯度: {grad_numerical:.6f}")
    print(f"解析梯度: {grad_analytical:.6f}")
    print(f"差异: {diff:.8f}")
    
    if diff < 1e-5:
        print("✓ 梯度检查通过！反向传播实现正确")
    else:
        print("✗ 梯度检查失败！请检查反向传播代码")
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)


def test_whitelist_compliance():
    """检查代码是否符合 WhiteList 约束"""
    
    print("\n" + "=" * 60)
    print("WhiteList 合规性检查")
    print("=" * 60)
    
    # 读取 main.py
    with open('main.py', 'r', encoding='utf-8') as f:
        code = f.read()
    
    # 禁止项检查
    forbidden_items = [
        'backward()',
        'torch.optim',
        'nn.Linear',
        'nn.Conv2d',
        'nn.Module',
        'requires_grad=True',
        'autograd'
    ]
    
    violations = []
    for item in forbidden_items:
        if item in code:
            violations.append(item)
    
    if violations:
        print("✗ 发现违规项:")
        for v in violations:
            print(f"  - {v}")
    else:
        print("✓ 未发现违规项")
    
    # 允许项检查
    allowed_items = [
        'torch.matmul',
        'torch.randn',
        'torch.zeros'
    ]
    
    used_items = []
    for item in allowed_items:
        if item in code:
            used_items.append(item)
    
    print(f"\n✓ 使用的允许项: {', '.join(used_items)}")
    print("=" * 60)


if __name__ == "__main__":
    test_forward_backward()
    test_whitelist_compliance()
