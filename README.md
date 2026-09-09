# CNN-ResNet-test

基于 PyTorch 从零手写实现的 ResNet18 网络，在 FashionMNIST 数据集上做图像分类实验。项目在残差块中加入了 Dropout 以尝试缓解过拟合，但实验效果不显著（详见文末）。

## 项目简介

- 数据集：FashionMNIST（10 类灰度图像，28×28，resize 到 224×224）
- 网络：ResNet18（输入 1 通道，输出 10 类）
- 优化器：Adam（lr=0.001）
- 损失函数：CrossEntropyLoss
- 超参数：batch_size=32，epochs=20，训练/验证按 8:2 划分

## 文件结构

- `model.py` — ResNet18 模型定义，含残差块 `Residual` 与 Dropout 层
- `model_train.py` — 数据加载、训练/验证流程、最优权重保存、loss/acc 曲线绘制
- `Figure_1.png` — 训练过程中 loss 与 accuracy 的变化曲线
- `best_model.pth` — 训练得到的最优模型权重（体积较大，未随仓库提交，可运行训练脚本重新生成）

## 环境依赖

- Python 3.x
- PyTorch
- torchvision
- torchsummary
- matplotlib
- numpy
- pandas

## 模型结构

- 输入：1 通道灰度图（224×224）
- 输出：10 类
- 主干：7×7 卷积（stride=2）+ MaxPool，四组残差块（通道数 64→128→256→512），自适应平均池化 + 全连接层（512→10）

## 使用方法

### 训练

```bash
python model_train.py
```

首次运行会自动下载 FashionMNIST 到 `./data`，训练结束后保存 `best_model.pth` 并弹出 loss/acc 曲线图。

### 加载权重进行推理

```python
import torch
from model import ResNet18, Residual

model = ResNet18(Residual)
model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
model.eval()
```

## 关于过拟合与 Dropout 实验

训练过程中观察到验证集与训练集精度之间存在明显差距（过拟合迹象）。为缓解过拟合，在 `model.py` 的 `Residual.forward` 中、第一个卷积（conv1 + BN + ReLU）之后加入了 Dropout：

```python
y = F.dropout(y, 0.5)
```

**实验结果：Dropout 对过拟合的缓解效果不显著。**

可能的原因：

1. **位置不当**：Dropout 加在卷积层之后，而非通常更有效的全连接层之后；卷积层参数共享，本身抗过拟合能力较强，Dropout 的额外收益有限。
2. **残差连接削弱了 Dropout 效果**：skip connection 保留了未经过 Dropout 的特征，信息可绕过 Dropout 路径，弱化了其正则作用。
3. **丢弃率偏高**：p=0.5 作用于卷积特征图可能偏大，容易导致训练不稳定甚至欠拟合。
4. **实现细节**：`F.dropout(y, 0.5)` 未显式传入 `training` 参数，其默认值为 `True`，因此在验证/测试阶段（`model.eval()`）Dropout 仍然激活，导致训练与验证行为不一致，验证精度被低估。

后续可尝试的方向：在全连接层前加 `nn.Dropout`、添加权重衰减（weight decay）、数据增强、学习率衰减或早停等。
