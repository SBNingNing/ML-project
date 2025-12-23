# 反向传播数学推导详解

## 符号说明

- $X$: 输入数据，shape = (batch_size, input_size)
- $W^{[l]}$: 第 $l$ 层的权重矩阵
- $b^{[l]}$: 第 $l$ 层的偏置向量
- $Z^{[l]}$: 第 $l$ 层的线性输出（激活前）
- $A^{[l]}$: 第 $l$ 层的激活输出（激活后）
- $Y$: 真实标签
- $\hat{Y}$: 模型预测值
- $L$: 损失函数

## 网络结构

两层 MLP：

```
输入 X
  ↓
第一层: Z^[1] = X·W^[1] + b^[1]
  ↓
ReLU:   A^[1] = max(0, Z^[1])
  ↓
第二层: Z^[2] = A^[1]·W^[2] + b^[2]
  ↓
Sigmoid: A^[2] = σ(Z^[2]) = 1/(1 + e^(-Z^[2]))
  ↓
输出 ŷ = A^[2]
```

## 前向传播公式

### 第一层

$$Z^{[1]} = X \cdot W^{[1]} + b^{[1]}$$

$$A^{[1]} = \text{ReLU}(Z^{[1]}) = \max(0, Z^{[1]})$$

**Shape**:
- $X$: (batch, input_size)
- $W^{[1]}$: (input_size, hidden_size)
- $b^{[1]}$: (hidden_size,)
- $Z^{[1]}$: (batch, hidden_size)
- $A^{[1]}$: (batch, hidden_size)

### 第二层

$$Z^{[2]} = A^{[1]} \cdot W^{[2]} + b^{[2]}$$

$$\hat{Y} = A^{[2]} = \sigma(Z^{[2]}) = \frac{1}{1 + e^{-Z^{[2]}}}$$

**Shape**:
- $A^{[1]}$: (batch, hidden_size)
- $W^{[2]}$: (hidden_size, 1)
- $b^{[2]}$: (1,)
- $Z^{[2]}$: (batch, 1)
- $\hat{Y}$: (batch, 1)

## 损失函数

**Binary Cross-Entropy (BCE) Loss**:

$$L = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

其中 $m$ 是 batch size。

**加权 BCE Loss**（处理类别不平衡）:

$$L = -\frac{1}{m} \sum_{i=1}^{m} \left[ w \cdot y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

其中 $w$ 是正样本权重（如 $w=3$）。

## 反向传播推导

### 第二层（输出层）

#### 1. 对 $Z^{[2]}$ 的导数

根据链式法则：

$$\frac{\partial L}{\partial Z^{[2]}} = \frac{\partial L}{\partial \hat{Y}} \cdot \frac{\partial \hat{Y}}{\partial Z^{[2]}}$$

**BCE Loss 对 Sigmoid 的导数**（推导见附录 A）：

$$\frac{\partial L}{\partial Z^{[2]}} = \hat{Y} - Y = A^{[2]} - Y$$

**Shape**: (batch, 1)

**代码实现**:
```python
dZ2 = A2 - Y  # (batch, 1)
```

#### 2. 对 $W^{[2]}$ 的导数

$$\frac{\partial L}{\partial W^{[2]}} = \frac{\partial L}{\partial Z^{[2]}} \cdot \frac{\partial Z^{[2]}}{\partial W^{[2]}}$$

因为 $Z^{[2]} = A^{[1]} \cdot W^{[2]} + b^{[2]}$，所以：

$$\frac{\partial Z^{[2]}}{\partial W^{[2]}} = A^{[1]}$$

因此：

$$\frac{\partial L}{\partial W^{[2]}} = (A^{[1]})^T \cdot \frac{\partial L}{\partial Z^{[2]}}$$

**Shape**: (hidden_size, 1)

**代码实现**:
```python
dW2 = torch.matmul(A1.transpose(0, 1), dZ2) / batch_size
```

#### 3. 对 $b^{[2]}$ 的导数

$$\frac{\partial L}{\partial b^{[2]}} = \sum_{i=1}^{m} \frac{\partial L}{\partial Z_i^{[2]}}$$

**Shape**: (1,)

**代码实现**:
```python
db2 = torch.sum(dZ2, dim=0) / batch_size
```

### 第一层（隐藏层）

#### 4. 对 $A^{[1]}$ 的导数

$$\frac{\partial L}{\partial A^{[1]}} = \frac{\partial L}{\partial Z^{[2]}} \cdot \frac{\partial Z^{[2]}}{\partial A^{[1]}}$$

因为 $Z^{[2]} = A^{[1]} \cdot W^{[2]} + b^{[2]}$，所以：

$$\frac{\partial Z^{[2]}}{\partial A^{[1]}} = W^{[2]}$$

因此：

$$\frac{\partial L}{\partial A^{[1]}} = \frac{\partial L}{\partial Z^{[2]}} \cdot (W^{[2]})^T$$

**Shape**: (batch, hidden_size)

**代码实现**:
```python
dA1 = torch.matmul(dZ2, W2.transpose(0, 1))
```

#### 5. 对 $Z^{[1]}$ 的导数

$$\frac{\partial L}{\partial Z^{[1]}} = \frac{\partial L}{\partial A^{[1]}} \cdot \frac{\partial A^{[1]}}{\partial Z^{[1]}}$$

**ReLU 的导数**:

$$\frac{\partial A^{[1]}}{\partial Z^{[1]}} = \begin{cases} 1 & \text{if } Z^{[1]} > 0 \\ 0 & \text{if } Z^{[1]} \leq 0 \end{cases}$$

因此：

$$\frac{\partial L}{\partial Z^{[1]}} = \frac{\partial L}{\partial A^{[1]}} \odot \mathbb{1}_{Z^{[1]} > 0}$$

其中 $\odot$ 表示逐元素乘法。

**Shape**: (batch, hidden_size)

**代码实现**:
```python
dZ1 = dA1 * torch.where(Z1 > 0, torch.ones_like(Z1), torch.zeros_like(Z1))
```

#### 6. 对 $W^{[1]}$ 的导数

$$\frac{\partial L}{\partial W^{[1]}} = (X)^T \cdot \frac{\partial L}{\partial Z^{[1]}}$$

**Shape**: (input_size, hidden_size)

**代码实现**:
```python
dW1 = torch.matmul(X.transpose(0, 1), dZ1) / batch_size
```

#### 7. 对 $b^{[1]}$ 的导数

$$\frac{\partial L}{\partial b^{[1]}} = \sum_{i=1}^{m} \frac{\partial L}{\partial Z_i^{[1]}}$$

**Shape**: (hidden_size,)

**代码实现**:
```python
db1 = torch.sum(dZ1, dim=0) / batch_size
```

## 参数更新（SGD）

使用梯度下降更新参数：

$$W^{[1]} := W^{[1]} - \alpha \cdot \frac{\partial L}{\partial W^{[1]}}$$

$$b^{[1]} := b^{[1]} - \alpha \cdot \frac{\partial L}{\partial b^{[1]}}$$

$$W^{[2]} := W^{[2]} - \alpha \cdot \frac{\partial L}{\partial W^{[2]}}$$

$$b^{[2]} := b^{[2]} - \alpha \cdot \frac{\partial L}{\partial b^{[2]}}$$

其中 $\alpha$ 是学习率（如 $\alpha = 0.001$）。

**代码实现**:
```python
W1 = W1 - lr * dW1
b1 = b1 - lr * db1
W2 = W2 - lr * dW2
b2 = b2 - lr * db2
```

## 完整反向传播流程

```
1. 计算输出层梯度:
   dZ2 = A2 - Y

2. 计算输出层参数梯度:
   dW2 = A1^T · dZ2 / m
   db2 = sum(dZ2) / m

3. 反向传播到隐藏层:
   dA1 = dZ2 · W2^T

4. 计算隐藏层梯度（通过 ReLU）:
   dZ1 = dA1 ⊙ 1{Z1 > 0}

5. 计算隐藏层参数梯度:
   dW1 = X^T · dZ1 / m
   db1 = sum(dZ1) / m

6. 更新所有参数:
   W1 := W1 - α·dW1
   b1 := b1 - α·db1
   W2 := W2 - α·dW2
   b2 := b2 - α·db2
```

## 附录 A: BCE + Sigmoid 导数推导

**目标**: 证明

$$\frac{\partial L}{\partial Z} = \hat{Y} - Y = \sigma(Z) - Y$$

其中 $L$ 是 BCE Loss，$\hat{Y} = \sigma(Z)$。

**步骤 1**: BCE Loss 定义

$$L = -\left[ Y \log(\hat{Y}) + (1 - Y) \log(1 - \hat{Y}) \right]$$

**步骤 2**: 对 $\hat{Y}$ 求导

$$\frac{\partial L}{\partial \hat{Y}} = -\frac{Y}{\hat{Y}} + \frac{1 - Y}{1 - \hat{Y}}$$

**步骤 3**: Sigmoid 函数的导数

$$\sigma(Z) = \frac{1}{1 + e^{-Z}}$$

$$\frac{\partial \sigma}{\partial Z} = \sigma(Z) \cdot (1 - \sigma(Z))$$

**步骤 4**: 链式法则

$$\frac{\partial L}{\partial Z} = \frac{\partial L}{\partial \hat{Y}} \cdot \frac{\partial \hat{Y}}{\partial Z}$$

$$= \left( -\frac{Y}{\hat{Y}} + \frac{1 - Y}{1 - \hat{Y}} \right) \cdot \hat{Y}(1 - \hat{Y})$$

$$= -Y(1 - \hat{Y}) + (1 - Y)\hat{Y}$$

$$= -Y + Y\hat{Y} + \hat{Y} - Y\hat{Y}$$

$$= \hat{Y} - Y$$

**结论**: 

$$\frac{\partial L}{\partial Z} = \hat{Y} - Y$$

这就是为什么在代码中可以直接写 `dZ2 = A2 - Y`！

## 附录 B: 数值梯度检查

为了验证反向传播实现是否正确，可以使用数值梯度：

$$\frac{\partial L}{\partial W} \approx \frac{L(W + \epsilon) - L(W - \epsilon)}{2\epsilon}$$

其中 $\epsilon$ 是一个很小的数（如 $10^{-5}$）。

**代码示例**:
```python
epsilon = 1e-5

# 解析梯度（你的实现）
grad_analytical = dW[i, j]

# 数值梯度
W[i, j] += epsilon
loss_plus = compute_loss()

W[i, j] -= 2 * epsilon
loss_minus = compute_loss()

grad_numerical = (loss_plus - loss_minus) / (2 * epsilon)

# 检查差异
diff = abs(grad_numerical - grad_analytical)
print(f"差异: {diff:.8f}")

# 如果差异 < 1e-5，说明反向传播实现正确
assert diff < 1e-5, "梯度检查失败！"
```

## 总结

反向传播的核心是**链式法则**。关键步骤：

1. 从输出层开始，计算 Loss 对激活前值的导数
2. 利用导数计算参数梯度
3. 将梯度反向传播到前一层
4. 重复步骤 2-3 直到输入层
5. 使用梯度下降更新所有参数

**为什么 BCE + Sigmoid 这么简洁？**

因为数学恰好简化为 $\frac{\partial L}{\partial Z} = \hat{Y} - Y$，这大大简化了实现！

**如何确保实现正确？**

1. 推导数学公式（如本文档）
2. 实现代码
3. 进行数值梯度检查
4. 观察 Loss 是否下降

祝你成功完成 Task 1！
