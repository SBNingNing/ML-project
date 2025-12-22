## 项目结构
Python=3.9.25
```
ML-project/
├── Task1/                    # 任务1：二分类（从零实现）
│   ├── main.py              # 训练脚本
│   ├── For_TA_test.py       # 测试脚本（供助教使用）
│   ├── test_implementation.py  # 验证脚本
│   ├── README.md            # 详细文档
│   ├── USAGE.md             # 使用指南
│   ├── model_weights.pth    # 训练后生成的模型
│   └── data/                # 数据目录
│       ├── img/             # 图片文件夹
│       ├── txt/             # 标签文件夹
│       └── README.md        # 数据说明
│
├── Task2/                    # 任务2：（待实现）
│   ├── main.py
│   └── For_TA_test.py
│
├── README.md                 # 本文档
├── reqiurment.txt           # 依赖包
└── WhiteList.txt            # 允许使用的库
```

## Task 1: Binary Defect Classification

### 任务描述

玻璃缺陷二分类任务，使用**完全手动实现的 MLP**（不使用 PyTorch 的自动微分和优化器）。

### 核心约束

🚫 **禁止使用**：
- `torch.autograd`（不能调用 `.backward()`）
- `torch.optim`（不能使用 `SGD`、`Adam` 等优化器）
- `torch.nn.Linear`、`torch.nn.Conv2d` 等自动层

✅ **允许使用**：
- 基础张量运算：`torch.matmul`、`torch.add`、`torch.sum` 等
- 数据处理：`numpy`、`pillow`、`opencv-python`

### 技术特点

- **手动前向传播**：使用 `torch.matmul` 实现矩阵乘法
- **手动反向传播**：基于链式法则计算梯度
- **手动参数更新**：实现 SGD 优化算法
- **加权损失函数**：处理类别不平衡问题

### 模型架构

```
输入: 64×64×3 = 12,288 维（图片展平）
  ↓
隐藏层: 128 神经元 + ReLU
  ↓
输出层: 1 神经元 + Sigmoid
  ↓
输出: 概率值 [0, 1]
```

### 快速开始

#### 1. 安装依赖

```bash
pip install -r reqiurment.txt
```

#### 2. 准备数据

将数据放入 `Task1/data/` 目录：

```
Task1/data/
├── img/              # 所有图片
│   ├── glass_001.png
│   └── ...
└── txt/              # 缺陷标注
    ├── glass_001.txt  # 有此文件表示有缺陷
    └── ...
```

#### 3. 验证实现

```bash
cd Task1
python test_implementation.py
```

#### 4. 训练模型

```bash
python main.py
```

#### 5. 测试模型

修改 `For_TA_test.py` 中的学号，然后运行：

```bash
python For_TA_test.py --test_data_path ./data
```

输出 `PB23000000.json`。

### 评价指标

重点关注 **F1-score**（针对 Defective 类）：

$$\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

### 详细文档

- [Task1/README.md](Task1/README.md) - 详细技术文档
- [Task1/USAGE.md](Task1/USAGE.md) - 使用指南
- [Task1/data/README.md](Task1/data/README.md) - 数据说明

## 依赖环境

```
Python 3.8+
PyTorch 2.0+
NumPy
Pillow
```

## 数学原理

### 前向传播

$$Z^{[1]} = X \cdot W^{[1]} + b^{[1]}$$
$$A^{[1]} = \text{ReLU}(Z^{[1]})$$
$$Z^{[2]} = A^{[1]} \cdot W^{[2]} + b^{[2]}$$
$$\hat{Y} = \text{Sigmoid}(Z^{[2]})$$

### 反向传播

输出层：
$$\frac{\partial L}{\partial Z^{[2]}} = \hat{Y} - Y$$

隐藏层：
$$\frac{\partial L}{\partial Z^{[1]}} = \frac{\partial L}{\partial A^{[1]}} \odot \text{ReLU}'(Z^{[1]})$$

权重梯度：
$$\frac{\partial L}{\partial W^{[2]}} = (A^{[1]})^T \cdot \frac{\partial L}{\partial Z^{[2]}}$$

### 参数更新

$$W = W - \alpha \cdot \frac{\partial L}{\partial W}$$

## 常见问题

### Q: 为什么用 MLP 而不是 CNN？

A: 手动实现卷积的反向传播极其复杂。MLP 足以完成二分类任务，且更容易实现和调试。

### Q: 如何处理类别不平衡？

A: 使用加权 BCE Loss，给缺陷样本赋予更高权重（`POS_WEIGHT=3.0`）。

### Q: 为什么要 Resize 图片？

A: 原图 320×320×3 = 307,200 维，参数量爆炸。Resize 到 64×64 只有 12,288 维，大大降低计算量。

### Q: 如何验证反向传播是否正确？

A: 使用数值梯度检查（Numerical Gradient Checking）：

```python
# 数值梯度
grad_numerical = (loss(W + ε) - loss(W - ε)) / (2ε)

# 解析梯度
grad_analytical = dW

# 检查差异
assert abs(grad_numerical - grad_analytical) < 1e-5
```

运行 `test_implementation.py` 可自动进行梯度检查。

## 提交检查清单

- [ ] `main.py` 能正常训练并保存模型
- [ ] `For_TA_test.py` 能加载模型并输出 JSON
- [ ] JSON 格式正确（key 不带 `.png` 后缀）
- [ ] 修改了学号（`leader_id = 'PB23000000'`）
- [ ] 代码中没有使用禁止项（`backward`、`torch.optim`、`nn.Linear` 等）
- [ ] 运行 `test_implementation.py` 通过
- [ ] 预处理与训练时完全一致

## 性能基准

在标准数据集上的参考性能：

| 指标 | 数值 |
|------|------|
| Accuracy | 85-90% |
| Precision | 75-85% |
| Recall | 80-90% |
| **F1-score** | **78-87%** |

## 作者

2025年机器学习课程项目

## License

MIT

USTC ML-project
