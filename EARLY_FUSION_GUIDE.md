# Early Fusion ESPCN 修改指南

## 概述
本项目已修改为 **Early Fusion ESPCN**，即在**低分辨率空间**融合多帧信息进行超分辨率处理，无需运动补偿网络。

## 主要修改

### 1. 数据集处理 (dataset.py)
**修改内容**: 
- `TrainValidImageDataset.__getitem__`: 从单帧 [1, H, W] 改为多帧 [3, H, W]
- `TestImageDataset.__getitem__`: 同样支持多帧输入

**实现方式**:
```python
# 原始 LR 帧
lr_frame_center = lr_crop_y_tensor

# 使用高斯模糊模拟相邻帧
lr_frame_prev = imgproc.image_to_tensor(
    cv2.GaussianBlur(lr_crop_y_image, (3, 3), 1.0), 
    False, False
)

lr_frame_next = imgproc.image_to_tensor(
    cv2.GaussianBlur(lr_crop_y_image, (5, 5), 1.0), 
    False, False
)

# 在通道维度拼接: [3, H, W]
lr_crop_y_tensor = torch.cat([lr_frame_prev, lr_frame_center, lr_frame_next], dim=0)
```

### 2. 模型配置 (config.py)
**修改内容**:
```python
in_channels = 3      # 原为 1，现在支持 3 帧输入
out_channels = 1     # 保持不变
channels = 64        # 保持不变
upscale_factor = 4   # 保持不变
```

### 3. 模型架构 (model.py)
**修改内容**:
- 更新权重初始化逻辑，适应新的输入通道数
- 第一层卷积层现在接收 3 通道输入

**权重初始化**:
```python
if module.in_channels == channels:  # 第一层是 in_channels -> channels
    # 使用较小的初始化
    nn.init.normal_(module.weight.data, 0.0, 0.001)
else:
    # 其他层使用标准初始化
    nn.init.normal_(module.weight.data, 0.0, math.sqrt(...))
```

### 4. 推理脚本

#### inference_video.py (视频推理)
**修改内容**:
- 模型初始化: `in_channels=3`
- 推理循环: 同时读取前一帧、当前帧、后一帧，构建 [1, 3, H, W] 的输入张量

```python
# 堆叠成 [3, H, W]
tensor_multi_frame = torch.from_numpy(
    np.stack([prev_y_norm, curr_y_norm, next_y_norm], axis=0)
).unsqueeze(0).to(device, non_blocking=True)  # [1, 3, H, W]

# 推理
out_tensor = sr_model(tensor_multi_frame)
```

#### inference.py (图像推理)
**修改内容**:
- 模型初始化: `in_channels=3`
- 图像推理: 使用高斯模糊模拟相邻帧

```python
frame_prev = cv2.GaussianBlur(lr_y_np, (3, 3), 1.0)
frame_curr = lr_y_np
frame_next = cv2.GaussianBlur(lr_y_np, (5, 5), 1.0)

lr_multi_frame = np.stack([frame_prev, frame_curr, frame_next], axis=0)
lr_multi_frame = torch.from_numpy(lr_multi_frame).unsqueeze(0).to(device)
```

#### test.py (测试)
**修改内容**:
- 测试推理: 同样构建多帧输入

## 工作流程

### 训练
```bash
python train.py
```
- 数据加载时自动构建多帧输入
- 模型在低分辨率空间接收 3 通道输入并进行特征融合
- 输出单通道高分辨率图像

### 推理（图像）
```bash
python inference.py --inputs_path input.png --model_weights_path model.pth
```

### 推理（视频）
```bash
python inference_video.py \
  --inputs_path input.mp4 \
  --model_weights_path model.pth \
  --upscale_factor 4
```
- 自动读取相邻帧进行 Early Fusion
- 输出超分辨率视频

### 测试
```bash
python test.py
```

## 关键优势

1. **Low Resolution Fusion** (低分辨率融合)
   - 在低分辨率空间进行多帧融合，计算量低
   - 相比 Late Fusion，避免了高分辨率空间的重计算

2. **No Motion Compensation** (无运动补偿)
   - 使用高斯模糊简单模拟相邻帧
   - 避免复杂的光流估计，提高训练稳定性

3. **Temporal Information** (时间信息)
   - 充分利用视频的时间一致性
   - 可获得更稳定的超分辨率结果

## 权重兼容性

⚠️ **重要**: 新模型与原始 ESPCN (in_channels=1) 的权重**不兼容**

如需恢复原始 ESPCN:
1. 将 `config.py` 中的 `in_channels` 改为 `1`
2. 将 `dataset.py` 中的多帧堆叠改为单帧
3. 修改推理脚本回到单帧输入
4. 使用原始权重进行推理

## 建议的训练参数

```python
# config.py
batch_size = 16       # 根据显存调整
epochs = 3000
model_lr = 1e-2
lr_scheduler_milestones = [300, 2400]  # 10%, 80% of epochs
```

## 扩展方向

1. **多帧数量调整**
   - 修改数据集中的帧数（目前为 3 帧）
   - 对应修改 `in_channels` 为帧数

2. **光流对齐**
   - 可选：添加光流估计对齐相邻帧
   - 无需修改模型架构

3. **自适应融合**
   - 使用注意力机制学习帧间权重
   - 在现有架构基础上扩展第一层
