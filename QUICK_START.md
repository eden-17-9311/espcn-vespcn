# 快速参考 - Vimeo90K 标准格式

## 🎯 三步启动

```bash
# 1️⃣  设置数据环境（自动下采样 + 生成列表）
python setup_vimeo90k_test.py --data_dir ./data/vimeo90k --max_seq 5

# 2️⃣  验证数据完整性
python verify_vimeo90k.py --data_dir ./data/vimeo90k

# 3️⃣  开始训练
python train.py
```

## 📁 必需的目录结构

```
data/vimeo90k/
├── sequences/                    # ✓ GT 训练集 (必需)
│   ├── 00001/0001/im*.png
│   ├── 00001/0002/im*.png
│   └── ...
├── test/sequences/               # ✓ GT 测试集 (必需)
│   ├── 00001/0266/im*.png
│   └── ...
├── sequences_lrx4/               # 自动生成
│   └── (自动创建的 LR 版本)
└── test/sequences_lrx4/          # 自动生成
    └── (自动创建的 LR 版本)
```

## 🔧 新增工具

| 工具 | 功能 | 命令 |
|------|------|------|
| `setup_vimeo90k_test.py` | 一键设置 | `python setup_vimeo90k_test.py --data_dir ./data/vimeo90k --max_seq 5` |
| `downsample_vimeo90k.py` | 下采样生成 LR | `python downsample_vimeo90k.py --input_dir ./sequences --output_dir ./sequences_lrx4 --max_seq 5` |
| `generate_vimeo90k_lists.py` | 生成列表文件 | `python generate_vimeo90k_lists.py --input_dir ./sequences --output_dir . --max_seq 5` |
| `verify_vimeo90k.py` | 数据验证 | `python verify_vimeo90k.py --data_dir ./data/vimeo90k` |

## ⚙️ 配置关键点 (config.py)

```python
# 数据集类型
dataset_type = "video"

# 路径配置
train_gt_video_dir = "./data/vimeo90k/sequences"
test_gt_video_dir = "./data/vimeo90k/test/sequences"
test_lr_video_dir = "./data/vimeo90k/test/sequences_lrx4"

# 列表文件（自动生成，指定使用）
train_list_file = "./data/vimeo90k/sep_trainlist.txt"
test_list_file = "./data/vimeo90k/sep_testlist.txt"

# 测试模式：1 个 epoch
epochs = 1

# 生产模式：改为 100+
# epochs = 100
```

## 📝 列表文件格式

**sep_trainlist.txt:**
```
00001/0001
00001/0002
00001/0003
00002/0001
...
```

**sep_testlist.txt:**
```
00001/0266
00001/0268
00001/0275
...
```

## 🚀 工作流

### 快速测试（推荐）

```bash
# 一步搞定（仅处理前 5 个序列）
python setup_vimeo90k_test.py --data_dir ./data/vimeo90k --max_seq 5

# 开始训练
python train.py

# 监控
tensorboard --logdir ./samples/logs
```

### 完整生产流程

```bash
# 步骤 1: 下采样训练集
python downsample_vimeo90k.py \
    --input_dir ./data/vimeo90k/sequences \
    --output_dir ./data/vimeo90k/sequences_lrx4

# 步骤 2: 下采样测试集
python downsample_vimeo90k.py \
    --input_dir ./data/vimeo90k/test/sequences \
    --output_dir ./data/vimeo90k/test/sequences_lrx4

# 步骤 3: 生成列表
python generate_vimeo90k_lists.py \
    --input_dir ./data/vimeo90k/sequences \
    --output_dir ./data/vimeo90k \
    --train_ratio 0.8

# 步骤 4: 验证
python verify_vimeo90k.py --data_dir ./data/vimeo90k

# 步骤 5: 修改 config.py epochs 为 100+，然后训练
python train.py
```

## 🔄 核心改动

| 文件 | 改动 | 说明 |
|------|------|------|
| `dataset.py` | TrainValidVideoDataset | 添加 `file_list` 参数支持列表文件 |
| `dataset.py` | TestVideoDataset | 完全重写支持嵌套目录 + 列表文件 |
| `train.py` | load_dataset() | 传递 `train_list_file` 和 `test_list_file` |
| `config.py` | 新增参数 | `train_list_file`, `test_list_file` |
| `config.py` | epochs | 改为 `1`（测试）；生产改为 `100+` |

## ✅ 检查清单

- [ ] 数据目录结构符合标准 Vimeo90K 格式
- [ ] 运行 `setup_vimeo90k_test.py` 自动生成 LR 版本
- [ ] 运行 `verify_vimeo90k.py` 验证数据完整性
- [ ] 确认 `sep_trainlist.txt` 和 `sep_testlist.txt` 已生成
- [ ] `config.py` 中 `epochs = 1`（快速测试）
- [ ] 运行 `python train.py` 开始训练

## ⚡ 性能提示

- 使用 `--max_seq 5` 进行快速测试（~5 分钟）
- 完整数据集训练时移除 `--max_seq` 参数
- 启用 GPU：`config.device = torch.device("cuda", 0)` ✓（默认）
- 启用混合精度：`use_amp = True` ✓（默认）

## 🎓 示例：自定义数据集

```python
# 只使用序列 00001 到 00005
python setup_vimeo90k_test.py \
    --data_dir ./data/vimeo90k \
    --filter_seq_start 00001 \
    --filter_seq_end 00005

# 自定义训练/测试比例
python generate_vimeo90k_lists.py \
    --input_dir ./data/vimeo90k/sequences \
    --output_dir ./data/vimeo90k \
    --train_ratio 0.9  # 90% 训练，10% 测试
```

## 🔍 故障排除

| 问题 | 解决方案 |
|------|---------|
| 找不到序列 | `python verify_vimeo90k.py` 检查目录 |
| LR 版本缺失 | 重新运行 `downsample_vimeo90k.py` |
| 列表文件为空 | 运行 `generate_vimeo90k_lists.py` |
| 数据加载失败 | 检查 config.py 中的路径配置 |

## 📊 数据统计

标准 Vimeo90K (前 5 个序列)：
- 序列数：5
- 子序列数：~500
- 总帧数：~3500
- 每个子序列：7 帧 → 5 个样本

## 🎯 预期结果

- ✓ GT 和 LR 版本自动生成
- ✓ 列表文件自动生成
- ✓ 数据加载正常（无 FileNotFoundError）
- ✓ 模型训练启动
- ✓ PSNR/SSIM 持续改进

## 📞 调试命令

```bash
# 检查列表文件
head -10 ./data/vimeo90k/sep_trainlist.txt

# 检查序列存在性
ls ./data/vimeo90k/sequences/00001/0001/

# 检查 LR 版本
ls ./data/vimeo90k/sequences_lrx4/00001/0001/

# 完整验证
python verify_vimeo90k.py --data_dir ./data/vimeo90k --check_limit 10

# 查看数据集信息
python -c "from dataset import TrainValidVideoDataset; d = TrainValidVideoDataset('./data/vimeo90k/sequences', 68, 4, 'Train', 3, './data/vimeo90k/sep_trainlist.txt'); print(f'总样本数: {len(d)}')"
```

## 📚 文档导航

- **详细指南**: [VIMEO90K_GUIDE.md](VIMEO90K_GUIDE.md)
- **实现总结**: [VIMEO90K_IMPLEMENTATION.md](VIMEO90K_IMPLEMENTATION.md)
- **代码参考**: dataset.py, train.py, config.py
- **工具脚本**: setup_vimeo90k_test.py, verify_vimeo90k.py

---

**快速开始（一行命令）：**
```bash
python setup_vimeo90k_test.py --data_dir ./data/vimeo90k --max_seq 5 && python verify_vimeo90k.py --data_dir ./data/vimeo90k && python train.py
```

