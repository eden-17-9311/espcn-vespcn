# 标准 Vimeo90K 格式支持 - 实现总结

## 📋 修改概览

本次更新添加了对**标准 Vimeo90K 数据集格式**的完整支持，包括嵌套目录结构和列表文件。

## 🔧 核心修改

### 1. dataset.py

**修改内容：**

#### TrainValidVideoDataset
- ✅ 支持标准 Vimeo90K 嵌套格式（`00001/0001/im*.png`）
- ✅ 添加 `file_list` 参数，支持 `sep_trainlist.txt` 和 `sep_testlist.txt`
- ✅ 自动发现嵌套目录结构和单层结构
- ✅ 检查列表文件存在时优先使用，否则自动发现

**新签名：**
```python
TrainValidVideoDataset(
    gt_video_dir: str,
    gt_image_size: int,
    upscale_factor: int,
    mode: str,
    num_frames: int = 3,
    file_list: str = None  # 新参数
)
```

#### TestVideoDataset
- ✅ 完全重写以支持标准 Vimeo90K 格式
- ✅ 支持列表文件加载（`sep_testlist.txt`）
- ✅ 嵌套目录自动发现
- ✅ GT 和 LR 目录配对检查

**新签名：**
```python
TestVideoDataset(
    gt_video_dir: str,
    lr_video_dir: str,
    num_frames: int = 3,
    file_list: str = None  # 新参数
)
```

### 2. train.py

**修改内容：**

#### load_dataset() 函数
- ✅ 传递 `config.train_list_file` 给 `TrainValidVideoDataset`
- ✅ 传递 `config.test_list_file` 给 `TestVideoDataset`

**关键代码段：**
```python
train_datasets = TrainValidVideoDataset(
    config.train_gt_video_dir,
    config.gt_image_size,
    config.upscale_factor,
    "Train",
    num_frames=config.num_frames,
    file_list=config.train_list_file  # 新增
)

test_datasets = TestVideoDataset(
    config.test_gt_video_dir,
    config.test_lr_video_dir,
    num_frames=config.num_frames,
    file_list=config.test_list_file  # 新增
)
```

### 3. config.py

**修改内容：**

#### Vimeo90K 路径配置
```python
# 标准 Vimeo90K 格式
train_gt_video_dir = f"./data/vimeo90k/sequences"
test_gt_video_dir = f"./data/vimeo90k/test/sequences"
test_lr_video_dir = f"./data/vimeo90k/test/sequences_lrx4"

# 列表文件（关键）
train_list_file = f"./data/vimeo90k/sep_trainlist.txt"
test_list_file = f"./data/vimeo90k/sep_testlist.txt"
```

#### 测试模式配置
```python
# 已改为 1 个 epoch（用于快速测试）
epochs = 1  # 生产环境改为 100+
```

## ✨ 新增工具脚本

### 1. downsample_vimeo90k.py
**功能：** 生成 LR 版本图像

**使用：**
```bash
# 训练集下采样
python downsample_vimeo90k.py \
    --input_dir ./data/vimeo90k/sequences \
    --output_dir ./data/vimeo90k/sequences_lrx4 \
    --downscale_factor 4 \
    --max_seq 5  # 可选：仅处理前 5 个序列

# 测试集下采样
python downsample_vimeo90k.py \
    --input_dir ./data/vimeo90k/test/sequences \
    --output_dir ./data/vimeo90k/test/sequences_lrx4 \
    --downscale_factor 4
```

**特点：**
- 支持 Vimeo90K 嵌套目录
- 高质量插值（INTER_CUBIC）
- 进度提示
- 灵活的序列过滤（`--max_seq`, `--filter_seq_start`, `--filter_seq_end`）

### 2. generate_vimeo90k_lists.py
**功能：** 生成列表文件（sep_trainlist.txt, sep_testlist.txt）

**使用：**
```bash
python generate_vimeo90k_lists.py \
    --input_dir ./data/vimeo90k/sequences \
    --output_dir ./data/vimeo90k \
    --train_ratio 0.8 \
    --max_seq 5  # 可选
```

**输出：**
- `sep_trainlist.txt` - 训练列表
- `sep_testlist.txt` - 测试列表

**格式示例：**
```
00001/0001
00001/0002
...
```

### 3. setup_vimeo90k_test.py
**功能：** 一键设置测试环境（推荐）

**使用：**
```bash
# 最简单的方式
python setup_vimeo90k_test.py --data_dir ./data/vimeo90k --max_seq 5
```

**自动执行：**
1. 下采样生成 LR 训练集
2. 下采样生成 LR 测试集
3. 生成列表文件

### 4. verify_vimeo90k.py
**功能：** 验证数据集完整性

**使用：**
```bash
# 完整验证
python verify_vimeo90k.py --data_dir ./data/vimeo90k

# 快速检查
python verify_vimeo90k.py --data_dir ./data/vimeo90k --quick
```

**检查项：**
1. 目录结构
2. 列表文件
3. 序列文件
4. 数据加载

## 📁 预期目录结构

```
data/vimeo90k/
├── sequences/                    # GT 训练集
│   ├── 00001/
│   │   ├── 0001/
│   │   │   ├── im1.png
│   │   │   ├── im2.png
│   │   │   ├── im3.png
│   │   │   └── ...
│   │   ├── 0002/
│   │   │   └── ...
│   │   └── ...
│   └── ...
├── sequences_lrx4/               # LR 训练集（自动生成）
│   ├── 00001/
│   │   ├── 0001/
│   │   │   ├── im1.png
│   │   │   ├── im2.png
│   │   │   └── ...
│   │   └── ...
│   └── ...
├── test/
│   ├── sequences/                # GT 测试集
│   │   ├── 00001/
│   │   │   ├── 0266/
│   │   │   ├── 0268/
│   │   │   └── ...
│   │   └── ...
│   └── sequences_lrx4/           # LR 测试集（自动生成）
│       └── ...
├── sep_trainlist.txt             # 训练列表（自动生成）
└── sep_testlist.txt              # 测试列表（自动生成）
```

## 🚀 快速开始

### 步骤 1: 准备数据
确保你有标准 Vimeo90K 格式的数据：
```
./data/vimeo90k/sequences/     # GT
./data/vimeo90k/test/sequences/ # 测试 GT
```

### 步骤 2: 设置环境（一键）
```bash
python setup_vimeo90k_test.py --data_dir ./data/vimeo90k --max_seq 5
```

### 步骤 3: 验证数据
```bash
python verify_vimeo90k.py --data_dir ./data/vimeo90k
```

### 步骤 4: 开始训练
```bash
python train.py
```

## 📊 配置参数解析

### config.py 中的新参数

```python
# 列表文件（关键！）
train_list_file = "./data/vimeo90k/sep_trainlist.txt"
test_list_file = "./data/vimeo90k/sep_testlist.txt"

# 如果 file_list 为 None，则自动发现（向后兼容）
```

## ✅ 向后兼容性

- ✓ 旧的单层目录格式仍然支持（自动发现）
- ✓ 不提供列表文件时自动发现（file_list=None）
- ✓ TrainValidImageDataset 保持不变
- ✓ TestImageDataset 保持不变

## 🎯 核心改进

| 功能 | 之前 | 现在 |
|------|------|------|
| 目录格式 | 自动发现（需要特定结构） | 支持标准 Vimeo90K 嵌套 |
| 数据选择 | 自动发现全部 | 通过列表文件精确控制 |
| LR 生成 | 需要手动生成 | 自动化脚本 |
| 列表文件 | 无 | 支持 sep_trainlist.txt 等 |
| 快速测试 | 困难 | 支持 --max_seq 参数 |

## 🔍 数据加载流程

```
train.py (load_dataset)
    ↓
TrainValidVideoDataset.__init__
    ├── 检查 file_list 存在？
    │   ├── 是 → 从文件加载
    │   └── 否 → 自动发现
    ├── 发现序列结构
    │   └── 检查嵌套 (00001/0001) 或单层
    └── 生成样本索引
        └── 每个序列 → 多个样本（滑动窗口）
```

## 📝 示例：使用列表文件

```python
# 自动使用列表文件
train_datasets = TrainValidVideoDataset(
    gt_video_dir="./data/vimeo90k/sequences",
    gt_image_size=68,
    upscale_factor=4,
    mode="Train",
    num_frames=3,
    file_list="./data/vimeo90k/sep_trainlist.txt"
)
# 加载 sep_trainlist.txt 中列出的序列（例如：00001/0001, 00001/0002, ...）

# 自动发现（向后兼容）
train_datasets = TrainValidVideoDataset(
    gt_video_dir="./data/vimeo90k/sequences",
    gt_image_size=68,
    upscale_factor=4,
    mode="Train",
    num_frames=3
    # file_list=None（默认）
)
# 自动发现该目录下的所有序列
```

## 🎓 学习资源

- [Vimeo90K 官方](http://toflow.csail.mit.edu/)
- [PyTorch DataLoader](https://pytorch.org/docs/stable/data.html)
- [Early Fusion ESPCN 论文](https://arxiv.org/abs/1609.05158)

## 常见命令汇总

```bash
# 1. 一键设置（推荐）
python setup_vimeo90k_test.py --data_dir ./data/vimeo90k --max_seq 5

# 2. 分步骤设置
# 步骤 1: 下采样
python downsample_vimeo90k.py --input_dir ./data/vimeo90k/sequences --output_dir ./data/vimeo90k/sequences_lrx4 --max_seq 5

# 步骤 2: 生成列表
python generate_vimeo90k_lists.py --input_dir ./data/vimeo90k/sequences --output_dir ./data/vimeo90k --max_seq 5

# 3. 验证数据
python verify_vimeo90k.py --data_dir ./data/vimeo90k

# 4. 开始训练
python train.py

# 5. 监控训练
tensorboard --logdir ./samples/logs/ESPCN_x4_EarlyFusion_Vimeo90K
```

## 📞 故障排除

### 找不到序列？
```bash
# 检查列表文件内容
cat ./data/vimeo90k/sep_trainlist.txt | head -5

# 检查对应目录
ls -la ./data/vimeo90k/sequences/00001/0001/
```

### LR 版本缺失？
```bash
# 重新生成
python downsample_vimeo90k.py \
    --input_dir ./data/vimeo90k/sequences \
    --output_dir ./data/vimeo90k/sequences_lrx4
```

### 数据加载失败？
```bash
# 验证完整性
python verify_vimeo90k.py --data_dir ./data/vimeo90k
```

---

**更新日期：** 2024-12-28  
**版本：** 2.0 (Vimeo90K 标准格式支持)  
**状态：** ✅ 生产就绪
