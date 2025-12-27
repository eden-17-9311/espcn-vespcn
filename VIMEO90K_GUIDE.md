# Vimeo90K 标准格式支持 - 快速启动指南

## 概述

本项目现已完整支持标准 **Vimeo90K** 数据集格式，包括：

1. **嵌套目录结构**: `sequences/00001/0001/im1.png`, `sequences/00001/0002/im1.png`, ...
2. **列表文件支持**: `sep_trainlist.txt` 和 `sep_testlist.txt`
3. **自动下采样**: 生成 LR 版本用于模型输入
4. **灵活的训练/测试分割**: 基于列表文件的精确控制

## 快速设置（推荐）

如果你已有标准 Vimeo90K 数据格式的目录：

```bash
# 一键生成 LR 版本和列表文件（处理前 5 个序列用于测试）
python setup_vimeo90k_test.py --data_dir ./data/vimeo90k --max_seq 5
```

这个脚本将自动：
1. 生成 `sequences_lrx4/` 目录（训练集 LR）
2. 生成 `test/sequences_lrx4/` 目录（测试集 LR）
3. 生成 `sep_trainlist.txt` 和 `sep_testlist.txt`

## 分步设置

### 步骤 1: 下采样生成 LR 版本

**训练集**：
```bash
python downsample_vimeo90k.py \
    --input_dir ./data/vimeo90k/sequences \
    --output_dir ./data/vimeo90k/sequences_lrx4 \
    --downscale_factor 4 \
    --max_seq 5  # 测试时用，完整数据时删除此参数
```

**测试集**：
```bash
python downsample_vimeo90k.py \
    --input_dir ./data/vimeo90k/test/sequences \
    --output_dir ./data/vimeo90k/test/sequences_lrx4 \
    --downscale_factor 4
```

### 步骤 2: 生成列表文件

```bash
python generate_vimeo90k_lists.py \
    --input_dir ./data/vimeo90k/sequences \
    --output_dir ./data/vimeo90k \
    --train_ratio 0.8 \
    --max_seq 5  # 测试时用
```

这将生成：
- `./data/vimeo90k/sep_trainlist.txt` - 训练列表
- `./data/vimeo90k/sep_testlist.txt` - 测试列表

### 步骤 3: 验证目录结构

确保你的目录结构如下：

```
data/vimeo90k/
├── sequences/                    # GT 训练集
│   ├── 00001/
│   │   ├── 0001/
│   │   │   ├── im1.png
│   │   │   ├── im2.png
│   │   │   ├── ...
│   │   ├── 0002/
│   │   │   ├── im1.png
│   │   │   └── ...
│   │   └── ...
│   ├── 00002/
│   │   └── ...
│   └── ...
├── sequences_lrx4/               # LR 训练集（自动生成）
│   ├── 00001/0001/im*.png
│   ├── 00001/0002/im*.png
│   └── ...
├── test/
│   ├── sequences/                # GT 测试集
│   │   ├── 00001/0266/im*.png
│   │   ├── 00001/0268/im*.png
│   │   └── ...
│   └── sequences_lrx4/           # LR 测试集（自动生成）
│       ├── 00001/0266/im*.png
│       └── ...
├── sep_trainlist.txt             # 训练列表（自动生成）
│   # 00001/0001
│   # 00001/0002
│   # ...
└── sep_testlist.txt              # 测试列表（自动生成）
    # 00001/0266
    # 00001/0268
    # ...
```

## 开始训练

### 配置文件 (config.py)

确保以下配置已设置：

```python
# 数据集类型
dataset_type = "video"

# 视频数据集路径
train_gt_video_dir = "./data/vimeo90k/sequences"
test_gt_video_dir = "./data/vimeo90k/test/sequences"
test_lr_video_dir = "./data/vimeo90k/test/sequences_lrx4"

# 列表文件（关键！）
train_list_file = "./data/vimeo90k/sep_trainlist.txt"
test_list_file = "./data/vimeo90k/sep_testlist.txt"

# 测试时：1 个 epoch
epochs = 1

# 生产环境：改为 100 或更高
# epochs = 100
```

### 运行训练

```bash
python train.py
```

### 查看训练日志

```bash
tensorboard --logdir ./samples/logs/ESPCN_x4_EarlyFusion_Vimeo90K
```

## 数据集格式说明

### sep_trainlist.txt 格式

```
00001/0001
00001/0002
00001/0003
00001/0004
00001/0005
00002/0001
...
```

**说明**：
- 第一部分（`00001`）是序列号
- 第二部分（`0001`）是子序列号
- 对应目录：`sequences/00001/0001/`
- 包含的帧：`im1.png`, `im2.png`, `im3.png`, ..., `im7.png`（7 帧或更多）

### sep_testlist.txt 格式

```
00001/0266
00001/0268
00001/0275
00001/0278
00001/0285
...
```

**说明**：同上，但用于测试集

## 高级用法

### 只处理特定序列范围

```bash
python setup_vimeo90k_test.py \
    --data_dir ./data/vimeo90k \
    --filter_seq_start 00001 \
    --filter_seq_end 00005
```

### 跳过已完成的步骤

```bash
# 跳过下采样（已完成）
python setup_vimeo90k_test.py --data_dir ./data/vimeo90k --skip_downsample

# 跳过列表生成（已完成）
python setup_vimeo90k_test.py --data_dir ./data/vimeo90k --skip_lists
```

### 自定义训练/测试比例

```bash
python generate_vimeo90k_lists.py \
    --input_dir ./data/vimeo90k/sequences \
    --output_dir ./data/vimeo90k \
    --train_ratio 0.9  # 90% 训练，10% 测试
```

## 核心模块说明

### TrainValidVideoDataset (dataset.py)

读取标准 Vimeo90K 格式的多帧数据：

```python
dataset = TrainValidVideoDataset(
    gt_video_dir="./data/vimeo90k/sequences",
    gt_image_size=68,  # 17 * 4
    upscale_factor=4,
    mode="Train",
    num_frames=3,
    file_list="./data/vimeo90k/sep_trainlist.txt"  # 关键参数
)
```

**特点**：
- 自动发现嵌套目录结构
- 从列表文件加载特定序列
- 读取连续的 3 帧（前帧、当前帧、后帧）
- 相同的随机裁剪应用到所有帧

### TestVideoDataset (dataset.py)

读取测试集多帧数据：

```python
dataset = TestVideoDataset(
    gt_video_dir="./data/vimeo90k/test/sequences",
    lr_video_dir="./data/vimeo90k/test/sequences_lrx4",
    num_frames=3,
    file_list="./data/vimeo90k/sep_testlist.txt"
)
```

## 常见问题

### Q1: 我的数据已经有 LR 版本了，需要重新生成吗？

**A**: 如果你已有 LR 版本，可以使用 `--skip_downsample` 跳过下采样步骤：

```bash
python setup_vimeo90k_test.py --data_dir ./data/vimeo90k --skip_downsample
```

### Q2: 如何验证列表文件是否正确？

**A**: 检查生成的列表文件：

```bash
# 查看训练列表
head -10 ./data/vimeo90k/sep_trainlist.txt

# 查看测试列表
head -10 ./data/vimeo90k/sep_testlist.txt
```

### Q3: 训练时提示找不到某些文件？

**A**: 检查以下几点：

1. 列表文件中的路径是否存在：
   ```bash
   ls ./data/vimeo90k/sequences/00001/0001/
   # 应该包含 im1.png, im2.png, ...
   ```

2. LR 版本是否已生成：
   ```bash
   ls ./data/vimeo90k/sequences_lrx4/00001/0001/
   # 应该包含 im1.png, im2.png, ...
   ```

3. 配置文件中的路径是否正确（config.py）

### Q4: 如何只用前 5 个序列进行快速测试？

**A**: 使用 `--max_seq` 参数：

```bash
# 一键生成（只处理前 5 个序列）
python setup_vimeo90k_test.py --data_dir ./data/vimeo90k --max_seq 5
```

### Q5: 每个子序列应该包含多少帧？

**A**: 至少需要 `num_frames` 帧（默认为 3）。标准 Vimeo90K 每个子序列包含 7 帧，足以生成多个样本：

- 7 帧 → (7 - 3 + 1) = 5 个样本
- 15 帧 → (15 - 3 + 1) = 13 个样本

## 配置优化建议

### 对于快速测试（1 个 epoch）：

```python
epochs = 1
batch_size = 8  # 较小的 batch size
num_workers = 2
gt_image_size = 68  # 17 * 4
```

### 对于完整训练（100+ epochs）：

```python
epochs = 200
batch_size = 32
num_workers = 4
gt_image_size = 68

# 启用性能优化
use_amp = True  # 混合精度
gradient_accumulation_steps = 2  # 梯度累积
```

## 参考资源

- [Vimeo90K 官方网站](http://toflow.csail.mit.edu/)
- [Early Fusion ESPCN 论文](https://arxiv.org/abs/1609.05158)

## 支持的功能

✓ 标准 Vimeo90K 嵌套目录格式  
✓ sep_trainlist.txt / sep_testlist.txt 支持  
✓ 自动下采样生成 LR 版本  
✓ 自动列表生成  
✓ 灵活的序列过滤  
✓ GPU 加速训练  
✓ 混合精度（FP16）  
✓ TensorBoard 可视化  

## 许可证

Apache License 2.0

