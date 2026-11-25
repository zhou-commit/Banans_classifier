# 香蕉成熟度分类器（Banana Ripeness Classifier）

基于 PyTorch 的四分类图像识别项目：判断香蕉处于 `overripe` / `ripe` / `rotten` / `unripe` 哪一阶段。

**技术栈**：PyTorch + torchvision（迁移学习）+ Gradio（Web 演示）
**环境**：Python 3.12.14，`torch` / `torchvision` / `gradio` / `tqdm` / `numpy` / `pillow`
**结果**：ResNet18 全网络微调，验证集 **98.49%**（1123 张）

```
pip install torch torchvision gradio tqdm numpy pillow
```

## 一、项目结构

```
banans_classifier/
├── bananas_dataset/                 # 数据集（不入库）
│   └── classifier/{train,val,test}/{overripe,ripe,rotten,unripe}/
│
├── checkpoints/                     # 权重（不入库），按模型名自动分目录
│   ├── resnet/  best.pth + cp0..cp19.pth
│   ├── mlp/
│   └── my_model/
│
├── dataloader.py                    # 数据集 + 预处理 + DataLoader
├── train.py                         # 训练/评估函数 + ResNet18 训练入口
├── mlp.py                           # 对照实验：纯全连接
├── My_model.py                      # 对照实验：手写小CNN
├── test_once.py                     # 测试集一次性评估
└── gr_demo.py                       # Gradio 演示
```

数据流：`dataloader` → `train_model`（train 训练 / val 选优）→ `best.pth` → `evaluate_model` 或 `gr_demo` 加载。

## 二、模块说明

### 1. `dataloader.py` — 数据入口

两套预处理，区别只在"确定性"：

```python
# 训练：带随机性（数据增强）
RandomResizedCrop(288, scale=(0.7, 1.0)), RandomHorizontalFlip()

# 验证/测试：完全确定性
Resize((288, 288)), CenterCrop(288)
```

`Normalize` 用 ImageNet 的六个统计量，因为骨干用的就是 ImageNet 预训练权重，输入分布必须对齐。

`DataLoader` 参数：`batch_size=32`、`shuffle/drop_last=is_train`、`num_workers=2`。训练集 `11793 // 32 = 368` 批（`drop_last` 每轮丢 17 张，避免末批 BN 统计量剧烈波动）。

**类别顺序由文件夹名字母序决定**：`overripe=0, ripe=1, rotten=2, unripe=3`。这个映射是隐式的，任何推理代码都应取 `train_dataset.classes` 而不是手写列表。

### 2. `train.py` — 训练核心

`train_model()` 一轮的骨架：

```python
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()

    val_acc, val_loss = evaluate_model(model, val_loader)
    if val_loss < best_loss:                       # 按 val_loss 挑最优
        best_loss = val_loss
        torch.save(model.state_dict(), 'best.pth')
    torch.save(model.state_dict(), f'cp{epoch}.pth')
    scheduler.step()
```

几个设计点：

| 设计 | 原因 |
|---|---|
| `best_loss = float('inf')` | 写成 `0.0` 会让 `val_loss < 0.0` 恒为假，`best.pth` 一次都不保存 |
| 按 `val_loss` 而非 `val_acc` 选优 | 1123 张时准确率的 SE ≈ 0.36pp，98.0 与 98.5 无统计差异；loss 是连续值，更灵敏 |
| 检查点分目录 | `type(model).__name__.lower()` → `checkpoints/resnet/`、`mlp/`、`my_model/`，避免互相覆盖 |
| 两种检查点 | `best.pth` 用于部署，`cp{epoch}.pth` 用于回溯训练过程 |

`evaluate_model()` 用 `model.eval() + no_grad()`，loss 按样本数加权（`loss.item() * labels.size(0)`），因为末批可能不满 32 张。

优化器是差异化学习率，配合余弦退火：

```python
Adam([
    {'params': 骨干参数, 'lr': 1e-5},   # 预训练权重只需轻推
    {'params': fc 参数,  'lr': 1e-3},   # 新建的分类头要大步走
])
CosineAnnealingLR(optimizer, T_max=20)
```

`FREEZE_FEATURES`、`SMOKE_TEST` 是模块级开关，用于"只训分类头"和"小数据冒烟测试"两组对照实验。

### 3. `gr_demo.py` — 推理与界面

`best.pth` 存的是 `state_dict`（参数名字典），不含网络结构，所以加载必须三步：

```python
model = models.resnet18(weights=None)          # 1. 造同架构空模型
model.fc = nn.Linear(model.fc.in_features, 4)  #    分类头也要一致
model.load_state_dict(torch.load(ckpt_path))   # 2. 填参数
model = model.to(device).eval()                # 3. 上设备 + 切评估模式
```

预测链路上每个环节都不能少：

```python
inputs = transform(image.convert("RGB")).unsqueeze(0).to(device)  # 补 batch 维
prob = torch.softmax(model(inputs), dim=1)                        # logits → 概率
```

| 遗漏 | 后果 |
|---|---|
| `.eval()` | BatchNorm 用单张图的统计量，结果乱跳 |
| `torch.no_grad()` | 白记梯度，显存翻倍 |
| `unsqueeze(0)` | 模型要 `(N,3,H,W)`，单图只有 `(3,H,W)` |
| `softmax` | 模型输出是原始分数，不是概率 |
| `transform` 与验证集对齐 | **不报错，精度悄悄下滑** |

### 4. `mlp.py` / `My_model.py` — 对照组

结构是 `train.py` 的复制品，只换模型，用于量化"卷积"和"预训练"各自的贡献：

| 模型 | 参数量级 | 起点 |
|---|---|---|
| `MLP` | 大（全连接） | 随机 |
| `My_model` | 小（3 层 conv） | 随机 |
| `ResNet18` | 11M | ImageNet 预训练 |

`My_model` 的卷积栈：`conv(3→16) → pool → conv(16→32) → pool → conv(32→64) → pool → flatten → fc(64*36*36 → 512) → fc(→4)`。
`36 = 288 / 8`，**改输入尺寸时必须同步改这行**。

### 5. `test_once.py` — 一次性测试

只在所有配置定稿后运行。`evaluate_model` 只读不写，不会影响 `best.pth`。

必须包在 `if __name__ == '__main__':` 里——`test_loader` 开了 `num_workers>0`，Windows 用 `spawn` 启动子进程会重新 import 主模块，模块级代码会导致无限递归。

## 三、实验结果

验证集 1123 张，`best.pth` 落在第 7 轮：

| epoch | val_acc | val_loss |
|:---:|:---:|:---:|
| 7 | 98.49% | **0.0657** ← 最低 |
| 9 | 98.58% | 0.0687 |
| 20 | 98.40% | 0.0852 |

第 7 轮之后 `train_loss` 从 0.0242 降到 0.0057，而 `val_loss` 从 0.0657 升到 0.0920（+40%），属过拟合，故按 loss 选优停在第 7 轮。准确率在这个区间内的波动（±0.5pp ≈ 6 张图）落在噪声内，不作为判据。

## 四、主要注意事项
以下都是我在实验当中出过的一些错误，仅作分享以示警
1. **`best_loss` 必须初始化为 `float('inf')`**。写成 0 会导致恒不保存，且不报错。症状：日志里每个 epoch 多打印一行（走了 `else` 分支）。
2. **对 `state_dict` 不能调 `.to()`**。只有 `torch.save(model)` 存的整个对象才可以。
3. **推理的 transform 必须与验证集一致**，不能与训练集一致。三项对齐：`Resize` 尺寸、`ToTensor`、`Normalize` 常数。
4. **改分辨率会连锁出错**：`dataloader` 改 288 后，`mlp.py` 的 `3*224*224`、`My_model.py` 的 `64*28*28`、`gr_demo.py` 的 `Resize(224)` 都要同步改。
5. **BatchNorm 定义了必须在 `forward` 里调用**。漏掉不报错，参数仍在 `state_dict` 里，只是永远是初始值。
6. **`BatchNorm2d` 用于卷积输出 `(N,C,H,W)`，`BatchNorm1d` 用于全连接输出 `(N,F)`**。
7. **`num_workers` 不要开太大**。16 GB 内存机器开 4 个 worker 会耗尽提交内存（物理 + 页面文件），表现为系统卡死。建议 2。
8. **检查点目录名全小写**。`My_model` 在 Windows 能访问 `my_model/`，Linux/macOS 区分大小写会 `FileNotFoundError`。
9. **相对路径依赖工作目录**。`dataloader.py` 用 `./bananas_dataset/classifier`，必须在项目根目录运行。
10. **测试集只用一次**。调参只看验证集，否则报出的数字有偏。

本项目仅用于学习用途。这个项目主要是专注于对代码的调优和精细化训练。 新人小白第一次在GitHub上进行项目开源部署，如果对你有帮助的话还请多多支持，点一个免费的star。
