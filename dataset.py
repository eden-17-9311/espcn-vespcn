# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
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
import os
import queue
import threading
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import imgproc

__all__ = [
    "TrainValidImageDataset", "TestImageDataset",
    "TrainValidVideoDataset", "TestVideoDataset",
    "PrefetchGenerator", "PrefetchDataLoader", "CPUPrefetcher", "CUDAPrefetcher",
]


class TrainValidImageDataset(Dataset):
    """Define training/valid dataset loading methods for single images.

    Args:
        gt_image_dir (str): Train/Valid ground-truth dataset address.
        gt_image_size (int): Ground-truth resolution image size.
        upscale_factor (int): Image up scale factor.
        mode (str): Data set loading method, the training data set is for data enhancement, and the
            verification dataset is not for data enhancement.
    """

    def __init__(
            self,
            gt_image_dir: str,
            gt_image_size: int,
            upscale_factor: int,
            mode: str,
    ) -> None:
        super(TrainValidImageDataset, self).__init__()
        self.image_file_names = [os.path.join(gt_image_dir, image_file_name) for image_file_name in
                                 os.listdir(gt_image_dir)]
        self.gt_image_size = gt_image_size
        self.upscale_factor = upscale_factor
        self.mode = mode

    def __getitem__(self, batch_index: int) -> [dict[str, Tensor], dict[str, Tensor]]:
        # Read a batch of image data
        gt_crop_image = cv2.imread(self.image_file_names[batch_index]).astype(np.float32) / 255.

        # Image processing operations
        if self.mode == "Train":
            gt_crop_image = imgproc.random_crop(gt_crop_image, self.gt_image_size)
        elif self.mode == "Valid":
            gt_crop_image = imgproc.center_crop(gt_crop_image, self.gt_image_size)
        else:
            raise ValueError("Unsupported data processing model, please use `Train` or `Valid`.")

        lr_crop_image = imgproc.image_resize(gt_crop_image, 1 / self.upscale_factor)

        # BGR convert Y channel
        gt_crop_y_image = imgproc.bgr_to_ycbcr(gt_crop_image, only_use_y_channel=True)
        lr_crop_y_image = imgproc.bgr_to_ycbcr(lr_crop_image, only_use_y_channel=True)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        gt_crop_y_tensor = imgproc.image_to_tensor(gt_crop_y_image, False, False)
        lr_crop_y_tensor = imgproc.image_to_tensor(lr_crop_y_image, False, False)

        return {"gt": gt_crop_y_tensor, "lr": lr_crop_y_tensor}

    def __len__(self) -> int:
        return len(self.image_file_names)


class TrainValidVideoDataset(Dataset):
    """Early Fusion Video Dataset: 读取真实的连续视频帧进行多帧融合训练
    
    支持标准 Vimeo90K 格式：
    sequences/
        00001/
            0001/
                im1.png, im2.png, im3.png, ...
            0002/
                im1.png, im2.png, im3.png, ...
            ...
        00002/
            ...
    
    Args:
        gt_video_dir (str): 包含视频帧序列的目录
        gt_image_size (int): 裁剪的GT分辨率
        upscale_factor (int): 超分倍率
        mode (str): "Train" 或 "Valid"
        num_frames (int): 每个样本使用的帧数（默认3）
        file_list (str): 列表文件路径（格式: 00001/0001）
    """

    def __init__(
            self,
            gt_video_dir: str,
            gt_image_size: int,
            upscale_factor: int,
            mode: str,
            num_frames: int = 3,
            file_list: str = None,
    ) -> None:
        super(TrainValidVideoDataset, self).__init__()
        self.gt_video_dir = gt_video_dir
        self.gt_image_size = gt_image_size
        self.upscale_factor = upscale_factor
        self.mode = mode
        self.num_frames = num_frames
        
        # 收集所有子序列
        self.sequence_paths = []
        
        if file_list and os.path.exists(file_list):
            # 从文件列表加载（标准 Vimeo90K 格式）
            print(f"从列表文件加载: {file_list}")
            with open(file_list, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # 格式: 00001/0001
                    seq_path = os.path.join(gt_video_dir, line)
                    if os.path.isdir(seq_path):
                        # 获取该序列中的所有帧
                        frames = sorted([f for f in os.listdir(seq_path) if f.endswith('.png')])
                        if len(frames) >= num_frames:
                            self.sequence_paths.append((seq_path, frames))
        else:
            # 自动发现（简单格式）
            print(f"自动发现序列目录")
            for seq_dir in sorted(os.listdir(gt_video_dir)):
                seq_path = os.path.join(gt_video_dir, seq_dir)
                if not os.path.isdir(seq_path):
                    continue
                
                # 检查是否为 Vimeo90K 嵌套格式（含有子目录）
                subdirs = [d for d in os.listdir(seq_path) if os.path.isdir(os.path.join(seq_path, d))]
                
                if subdirs:
                    # Vimeo90K 嵌套格式
                    for subdir in sorted(subdirs):
                        sub_seq_path = os.path.join(seq_path, subdir)
                        frames = sorted([f for f in os.listdir(sub_seq_path) if f.endswith('.png')])
                        if len(frames) >= num_frames:
                            self.sequence_paths.append((sub_seq_path, frames))
                else:
                    # 简单格式（单层）
                    frames = sorted([f for f in os.listdir(seq_path) if f.endswith('.png')])
                    if len(frames) >= num_frames:
                        self.sequence_paths.append((seq_path, frames))
        
        print(f"找到 {len(self.sequence_paths)} 个序列")
        
        # 计算总的训练样本数（每个序列产生多个样本）
        self.sample_indices = []
        for seq_idx, (seq_path, frame_list) in enumerate(self.sequence_paths):
            # 每个序列可以产生 (总帧数 - num_frames + 1) 个样本
            num_samples = len(frame_list) - num_frames + 1
            for start_idx in range(num_samples):
                self.sample_indices.append((seq_idx, start_idx))
    
    def __getitem__(self, batch_index: int) -> dict:
        seq_idx, start_idx = self.sample_indices[batch_index]
        seq_path, frame_list = self.sequence_paths[seq_idx]
        
        # 读取连续的 num_frames 帧
        frames_gt = []
        frames_lr = []
        
        for i in range(self.num_frames):
            frame_path = os.path.join(seq_path, frame_list[start_idx + i])
            gt_image = cv2.imread(frame_path).astype(np.float32) / 255.
            
            # 数据增强（仅在 Train 模式）
            if self.mode == "Train":
                gt_image = imgproc.random_crop(gt_image, self.gt_image_size)
            elif self.mode == "Valid":
                gt_image = imgproc.center_crop(gt_image, self.gt_image_size)
            else:
                raise ValueError("Unsupported data processing model, please use `Train` or `Valid`.")
            
            # 生成 LR 版本
            lr_image = imgproc.image_resize(gt_image, 1 / self.upscale_factor)
            
            # 转 Y 通道
            gt_y = imgproc.bgr_to_ycbcr(gt_image, only_use_y_channel=True)
            lr_y = imgproc.bgr_to_ycbcr(lr_image, only_use_y_channel=True)
            
            # 转 Tensor
            gt_y_tensor = imgproc.image_to_tensor(gt_y, False, False)  # [1, H, W]
            lr_y_tensor = imgproc.image_to_tensor(lr_y, False, False)  # [1, H, W]
            
            frames_gt.append(gt_y_tensor)
            frames_lr.append(lr_y_tensor)
        
        # 在通道维度拼接多帧
        # 从 [1, H, W] x num_frames 拼接为 [num_frames, H, W]
        gt_multi_frame = torch.cat(frames_gt, dim=0)  # [num_frames, H, W]
        lr_multi_frame = torch.cat(frames_lr, dim=0)  # [num_frames, H, W]
        
        return {"gt": gt_multi_frame, "lr": lr_multi_frame}
    
    def __len__(self) -> int:
        return len(self.sample_indices)


class TestImageDataset(Dataset):
    """Define Test dataset loading methods for single images.

    Args:
        test_gt_images_dir (str): ground truth image in test image
        test_lr_images_dir (str): low-resolution image in test image
    """

    def __init__(self, test_gt_images_dir: str, test_lr_images_dir: str) -> None:
        super(TestImageDataset, self).__init__()
        # Get all image file names in folder
        self.gt_image_file_names = [os.path.join(test_gt_images_dir, x) for x in os.listdir(test_gt_images_dir)]
        self.lr_image_file_names = [os.path.join(test_lr_images_dir, x) for x in os.listdir(test_lr_images_dir)]

    def __getitem__(self, batch_index: int) -> [torch.Tensor, torch.Tensor]:
        # Read a batch of image data
        image_path = self.gt_image_file_names[batch_index]
        # 尝试读取图片
        img = cv2.imread(image_path)

        # 检查是否读取成功
        if img is None:
            raise ValueError(f"【读取失败】无法找到或打开图片: {image_path}\n请检查路径是否存在，或路径中是否包含中文/特殊字符。")

        gt_image = img.astype(np.float32) / 255.
        lr_image = cv2.imread(self.lr_image_file_names[batch_index]).astype(np.float32) / 255.

        # BGR convert Y channel
        gt_y_image = imgproc.bgr_to_ycbcr(gt_image, only_use_y_channel=True)
        lr_y_image = imgproc.bgr_to_ycbcr(lr_image, only_use_y_channel=True)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        gt_y_tensor = imgproc.image_to_tensor(gt_y_image, False, False)
        lr_y_tensor = imgproc.image_to_tensor(lr_y_image, False, False)

        return {"gt": gt_y_tensor, "lr": lr_y_tensor}

    def __len__(self) -> int:
        return len(self.gt_image_file_names)


class TestVideoDataset(Dataset):
    """Early Fusion Video Dataset for testing: 支持标准 Vimeo90K 格式
    
    支持标准 Vimeo90K 格式：
    test/sequences/
        00001/
            0266/
                im1.png, im2.png, im3.png, ...
    test/sequences_lrx4/
        00001/
            0266/
                im1.png, im2.png, im3.png, ...
    
    Args:
        gt_video_dir (str): 包含视频帧序列的目录
        lr_video_dir (str): 包含LR视频帧序列的目录
        num_frames (int): 每个样本使用的帧数（默认3）
        file_list (str): 列表文件路径（格式: 00001/0266）
    """

    def __init__(
            self,
            gt_video_dir: str,
            lr_video_dir: str,
            num_frames: int = 3,
            file_list: str = None,
    ) -> None:
        super(TestVideoDataset, self).__init__()
        self.gt_video_dir = gt_video_dir
        self.lr_video_dir = lr_video_dir
        self.num_frames = num_frames
        
        # 收集所有子序列
        self.sequence_paths = []
        
        if file_list and os.path.exists(file_list):
            # 从文件列表加载
            print(f"从列表文件加载: {file_list}")
            with open(file_list, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # 格式: 00001/0266
                    gt_seq_path = os.path.join(gt_video_dir, line)
                    lr_seq_path = os.path.join(lr_video_dir, line)
                    
                    if os.path.isdir(gt_seq_path) and os.path.isdir(lr_seq_path):
                        gt_frames = sorted([f for f in os.listdir(gt_seq_path) if f.endswith('.png')])
                        lr_frames = sorted([f for f in os.listdir(lr_seq_path) if f.endswith('.png')])
                        
                        if len(gt_frames) >= self.num_frames and len(gt_frames) == len(lr_frames):
                            self.sequence_paths.append((gt_seq_path, lr_seq_path, gt_frames))
        else:
            # 自动发现
            print(f"自动发现序列目录")
            for seq_dir in sorted(os.listdir(gt_video_dir)):
                seq_path = os.path.join(gt_video_dir, seq_dir)
                if not os.path.isdir(seq_path):
                    continue
                
                # 检查是否为嵌套格式
                subdirs = [d for d in os.listdir(seq_path) if os.path.isdir(os.path.join(seq_path, d))]
                
                if subdirs:
                    # Vimeo90K 嵌套格式
                    for subdir in sorted(subdirs):
                        gt_sub_seq_path = os.path.join(seq_path, subdir)
                        lr_sub_seq_path = os.path.join(self.lr_video_dir, seq_dir, subdir)
                        
                        if os.path.isdir(lr_sub_seq_path):
                            gt_frames = sorted([f for f in os.listdir(gt_sub_seq_path) if f.endswith('.png')])
                            lr_frames = sorted([f for f in os.listdir(lr_sub_seq_path) if f.endswith('.png')])
                            
                            if len(gt_frames) >= self.num_frames and len(gt_frames) == len(lr_frames):
                                self.sequence_paths.append((gt_sub_seq_path, lr_sub_seq_path, gt_frames))
                else:
                    # 简单格式
                    lr_seq_path = os.path.join(self.lr_video_dir, seq_dir)
                    if os.path.isdir(lr_seq_path):
                        gt_frames = sorted([f for f in os.listdir(seq_path) if f.endswith('.png')])
                        lr_frames = sorted([f for f in os.listdir(lr_seq_path) if f.endswith('.png')])
                        
                        if len(gt_frames) >= self.num_frames and len(gt_frames) == len(lr_frames):
                            self.sequence_paths.append((seq_path, lr_seq_path, gt_frames))
        
        print(f"找到 {len(self.sequence_paths)} 个测试序列")
        
        # 计算总样本数
        self.sample_indices = []
        for seq_idx, (gt_seq_path, lr_seq_path, gt_frames) in enumerate(self.sequence_paths):
            num_samples = len(gt_frames) - self.num_frames + 1
            for start_idx in range(num_samples):
                self.sample_indices.append((seq_idx, start_idx))
    
    def __getitem__(self, batch_index: int) -> dict:
        seq_idx, start_idx = self.sample_indices[batch_index]
        gt_seq_path, lr_seq_path, gt_frames = self.sequence_paths[seq_idx]
        
        # 读取连续的 num_frames 帧
        frames_gt = []
        frames_lr = []
        
        for i in range(self.num_frames):
            gt_frame_path = os.path.join(gt_seq_path, gt_frames[start_idx + i])
            lr_frame_path = os.path.join(lr_seq_path, gt_frames[start_idx + i])
            
            gt_image = cv2.imread(gt_frame_path).astype(np.float32) / 255.
            lr_image = cv2.imread(lr_frame_path).astype(np.float32) / 255.
            
            # 转 Y 通道
            gt_y = imgproc.bgr_to_ycbcr(gt_image, only_use_y_channel=True)
            lr_y = imgproc.bgr_to_ycbcr(lr_image, only_use_y_channel=True)
            
            # 转 Tensor
            gt_y_tensor = imgproc.image_to_tensor(gt_y, False, False)
            lr_y_tensor = imgproc.image_to_tensor(lr_y, False, False)
            
            frames_gt.append(gt_y_tensor)
            frames_lr.append(lr_y_tensor)
        
        # 拼接多帧
        gt_multi_frame = torch.cat(frames_gt, dim=0)
        lr_multi_frame = torch.cat(frames_lr, dim=0)
        
        return {"gt": gt_multi_frame, "lr": lr_multi_frame}
    
    def __len__(self) -> int:
        return len(self.sample_indices)


class PrefetchGenerator(threading.Thread):
    """A fast data prefetch generator.

    Args:
        generator: Data generator.
        num_data_prefetch_queue (int): How many early data load queues.
    """

    def __init__(self, generator, num_data_prefetch_queue: int) -> None:
        threading.Thread.__init__(self)
        self.queue = queue.Queue(num_data_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self) -> None:
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):
    """A fast data prefetch dataloader.

    Args:
        num_data_prefetch_queue (int): How many early data load queues.
        kwargs (dict): Other extended parameters.
    """

    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None:
        self.num_data_prefetch_queue = num_data_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)


class CPUPrefetcher:
    """Use the CPU side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)


class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader: DataLoader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)
