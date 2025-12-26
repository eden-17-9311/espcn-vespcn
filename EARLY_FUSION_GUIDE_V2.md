# Early Fusion ESPCN - 真正的多帧融合版本 (V2)

## 概述

这是**真正的 Early Fusion ESPCN** 实现，支持读取真实的连续视频帧，让模型学习帧间的实际关系。与 V1（使用高斯模糊模拟帧）不同，V2 支持直接使用 **Vimeo90K** 等视频数据集进行训练。

## 核心改进

### ❌ V1 的问题
```python
# 模拟帧（V1）- 实际上还是单帧去模糊
lr_frame_prev = cv2.GaussianBlur(lr_crop_y_image, (3, 3), 1.0)  # 模拟
lr_frame_center = lr_crop_y_tensor  # 原始
lr_frame_next = cv2.GaussianBlur(lr_crop_y_image, (5, 5), 1.0)  # 模拟

# 结果：模型学到的是 "如何去掉高斯模糊"，而不是帧间关系
```

### ✅ V2 的改进
```python
# 真实帧（V2）- 读取连续视频帧
frame_1 = cv2.imread("sequence_1/im1.png")  # 真实前一帧
frame_2 = cv2.imread("sequence_1/im2.png")  # 真实当前帧
frame_3 = cv2.imread("sequence_1/im3.png")  # 真实后一帧

# 结果：模型学到真实的帧间关系和时间一致性
```

## 数据集类

### 1. TrainValidImageDataset（原有，单帧）
用于传统图像数据集：
- T91, DIV2K, CelebA 等
- 输入：单张图像
- 输出：单帧 `[1, H, W]`

### 2. TrainValidVideoDataset（新增，多帧）
用于视频数据集（Vimeo90K 格式）：
- 输入：连续视频帧序列
- 输出：多帧张量 `[num_frames, H, W]`
- 自动生成所有帧间的有效样本

### 3. TestImageDataset（原有，单帧）

### 4. TestVideoDataset（新增，多帧）
用于视频测试集

## 支持的数据格式

### Vimeo90K 格式（推荐）
```
vimeo_septuplet/
├── sequences/
│   ├── 00001/
│   │   ├── im1.png (GT)
│   │   ├── im2.png (GT)
│   │   ├── im3.png (GT)
│   │   ├── im4.png (GT)
│   │   ├── im5.png (GT)
│   │   ├── im6.png (GT)
│   │   └── im7.png (GT)
│   ├── 00002/
│   │   └── im1.png - im7.png
│   └── ...
├── test/
│   ├── sequences/
│   │   ├── 00701/
│   │   └── ...
│   └── sequences_lrx4/  # LR 版本
│       ├── 00701/
│       └── ...
```

### 其他视频格式
只要按照以下规则组织即可：
```
video_dataset/
├── sequence_name_1/
│   ├── frame_001.png
│   ├── frame_002.png
│   ├── frame_003.png
│   └── ...
├── sequence_name_2/
│   └── ...
```

## 关键配置

### config.py

```python
# ==================== 数据集类型 ====================
dataset_type = "video"  # 或 "image"

# 多帧数量
num_frames = 3  # 必须与 in_channels 相同

# 输入通道数（= num_frames）
in_channels = 3
# ====================================================

# ==================== 视频数据集路径 ====================
train_gt_video_dir = "./data/Vimeo90K/vimeo_septuplet/sequences"
test_gt_video_dir = "./data/Vimeo90K/vimeo_septuplet/test/sequences"
test_lr_video_dir = "./data/Vimeo90K/vimeo_septuplet/test/sequences_lrx4"
# ====================================================

# ==================== 图像数据集路径 ====================
train_gt_images_dir = "./data/T91/ESPCN/train"
test_gt_images_dir = "./data/Set5/GTmod12"
test_lr_images_dir = "./data/Set5/LRbicx4"
# ====================================================
```

## 数据加载流程

### 训练时（video dataset）

1. **样本索引生成**
   ```
   每个视频序列（7帧）可产生 7 - 3 + 1 = 5 个样本
   样本 1: frames [1, 2, 3]
   样本 2: frames [2, 3, 4]
   样本 3: frames [3, 4, 5]
   样本 4: frames [4, 5, 6]
   样本 5: frames [5, 6, 7]
   ```

2. **数据增强（Train 模式）**
   - 对所有帧应用相同的 crop 和 resize
   - 保证多帧的对齐

3. **批处理**
   ```python
   输入 LR: [batch_size, num_frames, H, W]
   输出 GT: [batch_size, num_frames, H, W]
   
   # 但传递给模型时需要确保通道维度
   # 模型接收: [batch_size, num_frames, H, W]
   ```

## 模型架构（支持动态帧数）

```python
ESPCN(
    in_channels=3,      # 可支持任意帧数（3, 5, 7等）
    out_channels=1,
    channels=64,
    upscale_factor=4
)

# 第一层卷积自动适应 in_channels
nn.Conv2d(in_channels, channels, (5, 5), (1, 1), (2, 2))
```

## 训练命令

### 使用 Vimeo90K（视频）
```bash
# 编辑 config.py
dataset_type = "video"
num_frames = 3
in_channels = 3

# 运行训练
python train.py
```

### 使用 T91（图像）
```bash
# 编辑 config.py
dataset_type = "image"
num_frames = 1
in_channels = 1

# 运行训练
python train.py
```

## 测试命令

### 测试视频
```bash
python inference_video.py \
  --inputs_path video.mp4 \
  --model_weights_path model.pth \
  --upscale_factor 4
```

### 测试图像
```bash
python inference.py \
  --inputs_path image.png \
  --model_weights_path model.pth
```

## 性能对比

### V1（模拟帧）vs V2（真实帧）

| 指标 | V1 | V2 | 说明 |
|------|----|----|------|
| 学习内容 | 去模糊 | 帧间关系 | 本质差异 |
| Vimeo90K 兼容 | ❌ | ✅ | V2 支持直接使用 |
| 视频连贯性 | 低 | 高 | V2 更稳定 |
| 计算复杂度 | 低 | 低 | 都在 LR 空间 |

## 扩展方向

### 1. 增加帧数
```python
# config.py
num_frames = 5
in_channels = 5

# 自动支持，无需改代码
```

### 2. 非均匀帧采样
修改 `TrainValidVideoDataset.__getitem__`：
```python
# 当前：连续采样 [i, i+1, i+2]
# 改为：间隔采样 [i, i+2, i+4]
indices = [start_idx + i * stride for i in range(num_frames)]
```

### 3. 光流对齐
添加光流估计后处理帧（可选的预处理步骤）

### 4. 自适应融合权重
修改模型第一层，添加注意力机制学习帧间权重

## 常见问题

### Q: 如何从 Vimeo90K 视频文件生成帧？
```bash
# 使用 FFmpeg
ffmpeg -i sequence.mp4 "sequences/%05d.png"
```

### Q: 训练时内存不足？
降低 batch_size：
```python
batch_size = 8  # 或更小
```

### Q: 如何切换回图像数据集？
```python
# config.py
dataset_type = "image"
in_channels = 1
```

### Q: 能否混合使用图像和视频数据？
可以，创建混合数据集类继承两个类，或分别训练后进行迁移学习。

## 完整训练流程示例

```python
# 1. 准备数据
# 下载 Vimeo90K 或使用自己的视频数据
# 提取帧序列到指定目录

# 2. 配置
# config.py:
dataset_type = "video"
num_frames = 3
in_channels = 3
train_gt_video_dir = "./data/Vimeo90K/sequences"

# 3. 训练
python train.py

# 4. 评估
python test.py

# 5. 推理
python inference_video.py --inputs_path input.mp4
```

## 性能指标建议

对于 Vimeo90K x4 超分：
- PSNR: 目标 > 28 dB
- SSIM: 目标 > 0.8
- 帧间一致性: 更平稳的帧序列

---

**V2 是真正意义的 Early Fusion，充分利用视频的时间一致性。**
