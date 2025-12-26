# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# When evaluating the performance of the SR model, whether to verify only the Y channel image data
only_test_y_channel = False
# Model architecture name
model_arch_name = "espcn_x4"
# Model arch config
# Early Fusion ESPCN: 多帧在低分辨率空间融合
# 对于视频数据：in_channels = 3（前帧、当前帧、后帧）
# 可调整为其他值以支持不同数量的帧
in_channels = 3
out_channels = 1
channels = 64
upscale_factor = 4
# 视频帧数配置（必须与 in_channels 一致）
num_frames = 3
# Current configuration parameter method
mode = "train"
# Experiment name, easy to save weights and log files
exp_name = "ESPCN_x4_EarlyFusion_Vimeo90K"

# ==================== 数据集类型选择 ====================
# "image": 使用 TrainValidImageDataset（单帧）- 用于 T91、DIV2K 等图像数据集
# "video": 使用 TrainValidVideoDataset（多帧）- 用于 Vimeo90K 等视频数据集
dataset_type = "video"
# ====================================================

if mode == "train":
    # ==================== 图像数据集配置 ====================
    # 使用 TrainValidImageDataset 时的路径配置
    train_gt_images_dir = f"./data/T91/ESPCN/train"
    test_gt_images_dir = f"./data/Set5/GTmod12"
    test_lr_images_dir = f"./data/Set5/LRbicx{upscale_factor}"
    # ====================================================
    
    # ==================== 视频数据集配置 ====================
    # 使用 TrainValidVideoDataset 时的路径配置
    # Vimeo90K 格式示例：
    # train_gt_video_dir/
    #   sequence_1/
    #     im1.png, im2.png, im3.png, ...
    #   sequence_2/
    #     im1.png, im2.png, im3.png, ...
    # ...
    train_gt_video_dir = f"./data/Vimeo90K/vimeo_septuplet/sequences"
    test_gt_video_dir = f"./data/Vimeo90K/vimeo_septuplet/test/sequences"
    test_lr_video_dir = f"./data/Vimeo90K/vimeo_septuplet/test/sequences_lrx{upscale_factor}"
    # ====================================================

    gt_image_size = int(17 * upscale_factor)
    batch_size = 16
    num_workers = 4

    # The address to load the pretrained model
    pretrained_model_weights_path = f""

    # Incremental training and migration training
    resume_model_weights_path = f""

    # Total num epochs
    epochs = 3000

    # loss function weights
    loss_weights = 1.0

    # Optimizer parameter
    model_lr = 1e-2
    model_momentum = 0.9
    model_weight_decay = 1e-4
    model_nesterov = False

    # EMA parameter
    model_ema_decay = 0.999

    # Dynamically adjust the learning rate policy
    lr_scheduler_milestones = [int(epochs * 0.1), int(epochs * 0.8)]
    lr_scheduler_gamma = 0.1

    # How many iterations to print the training result
    train_print_frequency = 100
    test_print_frequency = 1

if mode == "test":
    # Test data address
    lr_dir = f"./data/Set5/LRbicx{upscale_factor}"
    sr_dir = f"./results/test/{exp_name}"
    gt_dir = "./data/Set5/GTmod12"

    model_weights_path = "./results/pretrained_models/ESPCN_x4-T91-64bf5ee4.pth.tar"