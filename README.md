# 机器学习大作业: 玻璃表面缺陷检测 (ML-project)

本项目旨在通过深度学习方法解决玻璃表面的缺陷检测问题。项目包含两个主要任务：
1.  **Task 1**: 从零实现卷积神经网络 (CNN)，完成**二分类**任务（判断是否存在缺陷）。
2.  **Task 2**: 使用预训练的 ResNet-50 模型进行迁移学习和多任务学习，完成**多标签分类**任务（识别具体的缺陷类型），并引入辅助分割任务以提升分类性能。

## 📁 目录结构

```
ML-project/
├── dataset/                # 数据集目录
│   ├── train/              # 训练集 (包含 img/ 和 txt/)
│   └── test/               # 测试集 (包含 img/ 和 txt/)
├── Student_ID/             # 学生代码目录
│   ├── requirements.txt    # 项目依赖
│   ├── Task1/              # 任务1：二分类
│   │   ├── main.py         # 训练主入口
│   │   ├── model.py        # 自定义模型结构
│   │   ├── dataset.py      # 数据加载与平衡策略
│   │   ├── evaluate.py     # 评估脚本
│   │   └── ...
│   └── Task2/              # 任务2：多标签分类
│       ├── main.py         # 训练主入口
│       ├── model_mtl.py    # 多任务 ResNet 模型
│       ├── dataset.py      # 数据加载 (GlassDataset)
│       ├── evaluate.py     # 评估脚本
│       └── ...
└── README.md               # 项目说明文档
```

## 🛠️ 环境依赖

请确保安装 Python 3.8+ 及以下依赖库：

```bash
pip install -r Student_ID/requirement.txt
```

主要依赖包括：
*   torch, torchvision (深度学习框架)
*   numpy, pandas (数据处理)
*   Pillow, opencv-python-headless (图像处理)
*   scikit-learn (评估指标)
*   matplotlib (绘图)
*   tqdm (进度条)

## 🚀 任务说明

### Task 1: 缺陷二分类 (Binary Classification)

*   **目标**: 区分输入图像是“无缺陷”还是“有缺陷”。
*   **模型**: 位于 `Student_ID/Task1/model.py`。
    *   通过 `Layer` 和 `Conv2d` 类手动模拟了卷积层的实现细节（包括前向传播和部分反向传播逻辑），外层使用 PyTorch 管理。
    *   采用 **Deep CNN** 架构。
*   **策略**:
    *   **样本平衡**: 在 `dataset.py` 中实现了正负样本平衡策略（正样本过采样 4 倍，负样本随机采样），解决类别不平衡问题。
*   **运行**:
    ```bash
    cd Student_ID/Task1
    python main.py
    ```
    训练完成后，模型将保存为 `model_final.pth`。

### Task 2: 多标签分类与辅助分割 (Multi-Label Classification with Aux Segmentation)

*   **目标**: 识别图像中包含的具体缺陷类型。
    *   类别: `[No Defect, Chipped, Scratch, Stain]`
*   **模型**: 位于 `Student_ID/Task2/model_mtl.py` 的 `MultiTaskResNet`。
    *   **Backbone**: ResNet-50 (使用 ImageNet 预训练权重)。
    *   **Head 1 (Classifier)**: 负责 4 类多标签分类。
    *   **Head 2 (Segmentation)**: 辅助分割头，用于生成缺陷掩码，通过多任务学习 (MTL) 辅助主分类任务提取更鲁棒的特征。
*   **策略**:
    *   **Auxiliary Loss**: 训练时结合分类损失和分割损失 (`SEG_LOSS_WEIGHT = 1.0`)。
    *   **Thresholding**:在此任务中使用了特定的阈值 `[0.5, 0.5, 0.4, 0.3]` 来判定各个类别的存在。
*   **运行**:
    ```bash
    cd Student_ID/Task2
    python main.py
    ```
    训练过程中会保存最佳模型 `best_model.pth` 和最终模型 `last_model.pth`，并生成损失曲线和指标曲线图。

## 📊 数据集

数据集位于项目根目录的 `dataset/` 文件夹中。
*   **img/**: 包含 `.png` 格式的玻璃表面扫描图像。
*   **txt/**: 包含对应的标注文件。如果某张图片对应的 `.txt` 文件存在，则认为其为正样本（有缺陷）；Task 2 会解析 `.txt` 内容以获取具体的缺陷类别。

## 📝 评估

每个任务目录下均包含 `evaluate.py` 或 `For_TA_test.py` 脚本，用于加载训练好的模型并在测试集上计算 F1-score、Precision 和 Recall 等指标。
