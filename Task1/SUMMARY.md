# Task 1 项目总结

## 🎯 核心目标

实现一个**完全手动**的玻璃缺陷二分类器，不使用 PyTorch 的自动微分、优化器和预定义层。

## 📁 文件说明

| 文件 | 作用 | 重要性 |
|------|------|--------|
| **main.py** | 训练脚本（手动前向/反向传播） | ⭐⭐⭐⭐⭐ |
| **For_TA_test.py** | 测试脚本（供助教评分） | ⭐⭐⭐⭐⭐ |
| test_implementation.py | 验证脚本（梯度检查） | ⭐⭐⭐⭐ |
| README.md | 技术文档 | ⭐⭐⭐ |
| USAGE.md | 使用指南 | ⭐⭐⭐ |
| MATH_DERIVATION.md | 数学推导 | ⭐⭐ |
| data/README.md | 数据说明 | ⭐⭐ |

## ✅ 实现清单

### 已完成功能

- [x] **手动 MLP 模型类** (`ManualMLP`)
  - [x] 前向传播（使用 `torch.matmul`）
  - [x] 反向传播（基于链式法则）
  - [x] 参数更新（SGD）
  - [x] 加权 BCE Loss（处理类别不平衡）
  
- [x] **数据加载和预处理**
  - [x] 读取图片和标签
  - [x] Resize（320→64）
  - [x] 归一化（除以 255）
  - [x] 数据集划分（训练/验证）
  
- [x] **训练流程**
  - [x] Mini-batch 训练
  - [x] 评价指标计算（Precision、Recall、F1）
  - [x] 模型保存（最佳 F1）
  
- [x] **测试脚本**
  - [x] 加载模型
  - [x] 预测测试集
  - [x] 输出 JSON 格式结果

- [x] **验证工具**
  - [x] 梯度检查（数值微分）
  - [x] WhiteList 合规性检查
  
- [x] **文档**
  - [x] 详细技术文档
  - [x] 使用指南
  - [x] 数学推导
  - [x] 数据说明

## 🔑 关键代码片段

### 1. 手动前向传播

```python
def forward(self, X):
    # 第一层：Linear + ReLU
    Z1 = torch.matmul(X, self.W1) + self.b1
    A1 = torch.where(Z1 > 0, Z1, torch.zeros_like(Z1))  # ReLU
    
    # 第二层：Linear + Sigmoid
    Z2 = torch.matmul(A1, self.W2) + self.b2
    A2 = 1.0 / (1.0 + torch.exp(-Z2))  # Sigmoid
    
    return A2
```

### 2. 手动反向传播

```python
def backward(self, Y):
    # 输出层梯度
    dZ2 = A2 - Y  # BCE + Sigmoid 的简化形式
    dW2 = torch.matmul(A1.transpose(0, 1), dZ2) / batch_size
    db2 = torch.sum(dZ2, dim=0) / batch_size
    
    # 隐藏层梯度
    dA1 = torch.matmul(dZ2, self.W2.transpose(0, 1))
    dZ1 = dA1 * torch.where(Z1 > 0, 1, 0)  # ReLU 导数
    dW1 = torch.matmul(X.transpose(0, 1), dZ1) / batch_size
    db1 = torch.sum(dZ1, dim=0) / batch_size
```

### 3. 手动参数更新

```python
def update_parameters(self):
    self.W1 = self.W1 - self.lr * self.gradients['dW1']
    self.b1 = self.b1 - self.lr * self.gradients['db1']
    self.W2 = self.W2 - self.lr * self.gradients['dW2']
    self.b2 = self.b2 - self.lr * self.gradients['db2']
```

### 4. 数据加载

```python
def load_dataset(data_dir, img_size=64):
    for img_file in img_files:
        # 读取和预处理
        img = Image.open(img_path).convert('RGB')
        img = img.resize((img_size, img_size))
        img_array = np.array(img) / 255.0
        img_flat = img_array.flatten()
        
        # 判断标签
        label = 1 if os.path.exists(label_file) else 0
```

## 🚀 使用流程

```bash
# 1. 验证实现
python test_implementation.py

# 2. 训练模型
python main.py

# 3. 测试模型（记得修改学号！）
python For_TA_test.py --test_data_path ./data
```

## 📊 超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| IMG_SIZE | 64 | 图片缩放尺寸 |
| HIDDEN_SIZE | 128 | 隐藏层神经元 |
| LEARNING_RATE | 0.001 | 学习率 |
| EPOCHS | 50 | 训练轮数 |
| BATCH_SIZE | 32 | 批次大小 |
| POS_WEIGHT | 3.0 | 正样本权重 |

## 🔍 关键数学公式

### 前向传播

$$Z^{[1]} = X \cdot W^{[1]} + b^{[1]}$$
$$A^{[1]} = \text{ReLU}(Z^{[1]})$$
$$\hat{Y} = \text{Sigmoid}(A^{[1]} \cdot W^{[2]} + b^{[2]})$$

### 反向传播（核心）

$$\frac{\partial L}{\partial Z^{[2]}} = \hat{Y} - Y$$

$$\frac{\partial L}{\partial W^{[2]}} = (A^{[1]})^T \cdot \frac{\partial L}{\partial Z^{[2]}}$$

### 参数更新

$$W := W - \alpha \cdot \frac{\partial L}{\partial W}$$

## ⚠️ 注意事项

### 必须修改的地方

1. **For_TA_test.py** 第 136 行：
   ```python
   leader_id = 'PB23000000'  # 改为你的学号！
   ```

2. **main.py** 第 282 行（可选）：
   ```python
   DATA_DIR = './data'  # 如果数据在其他位置，修改此处
   ```

### 必须遵守的约束

🚫 **禁止使用**：
- `backward()`
- `torch.optim.*`
- `nn.Linear`、`nn.Conv2d`、`nn.Module`
- `requires_grad=True`

✅ **允许使用**：
- `torch.matmul`、`torch.add`、`torch.sum`
- `torch.randn`、`torch.zeros`
- `numpy`、`pillow`

### 数据格式要求

```
data/
├── img/
│   ├── glass_001.png    # 必须是 PNG 格式
│   └── ...
└── label/
    ├── glass_001.txt    # 有此文件 → Defective (1)
    └── ...              # 无文件 → Non-defective (0)
```

### JSON 输出格式

```json
{
    "glass_001": true,    // ✓ 不带后缀
    "glass_002": false
}
```

❌ **错误格式**：
```json
{
    "glass_001.png": true  // ✗ 不要带后缀
}
```

## 🐛 常见问题

### 问题 1: 模型全预测一个类别

**原因**: 类别不平衡

**解决**: 增大 `POS_WEIGHT`（如 5.0 或 10.0）

### 问题 2: Loss 为 NaN

**原因**: 学习率过大或梯度爆炸

**解决**: 降低学习率或确保 Sigmoid 输入被裁剪

### 问题 3: 梯度检查失败

**原因**: 反向传播实现错误

**解决**: 对照 [MATH_DERIVATION.md](MATH_DERIVATION.md) 检查公式

### 问题 4: FileNotFoundError

**原因**: 数据路径错误

**解决**: 确保 `data/img/` 和 `data/label/` 存在

## 📈 性能预期

| 指标 | 预期范围 |
|------|----------|
| Accuracy | 85-90% |
| Precision | 75-85% |
| Recall | 80-90% |
| **F1-score** | **78-87%** |

## 🎓 学习要点

1. **链式法则**: 反向传播的核心
2. **梯度下降**: 最基础的优化算法
3. **类别不平衡**: 使用加权损失函数
4. **数值梯度检查**: 验证反向传播正确性

## 📚 参考资料

- [README.md](README.md) - 详细技术文档
- [USAGE.md](USAGE.md) - 使用指南
- [MATH_DERIVATION.md](MATH_DERIVATION.md) - 数学推导
- [data/README.md](data/README.md) - 数据说明

## ✨ 项目亮点

1. **完全手动实现**: 不依赖任何自动微分框架
2. **详细注释**: 每个函数都有清晰的说明
3. **数学推导**: 提供完整的反向传播推导
4. **梯度检查**: 自动验证实现正确性
5. **完整文档**: 从入门到精通的全套文档

## 🎉 提交前检查

- [ ] 运行 `test_implementation.py` 通过
- [ ] 运行 `main.py` 训练成功
- [ ] 运行 `For_TA_test.py` 生成 JSON
- [ ] JSON 格式正确（key 不带后缀）
- [ ] 修改了学号（`leader_id`）
- [ ] 没有使用禁止的 API
- [ ] 预处理与训练时一致

---

**祝你顺利完成 Task 1！如有问题，请查阅各文档或运行验证脚本。** 🚀
