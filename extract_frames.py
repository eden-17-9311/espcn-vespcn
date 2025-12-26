#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从视频文件中提取帧，用于 Early Fusion ESPCN 训练
支持批量处理多个视频文件

使用方法:
    python extract_frames.py --input_dir ./videos --output_dir ./data/vimeo90k/sequences
"""

import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm


def extract_frames_from_video(video_path, output_dir, start_frame=0, max_frames=7):
    """
    从视频文件中提取帧
    
    Args:
        video_path (str): 视频文件路径
        output_dir (str): 输出目录
        start_frame (int): 起始帧索引
        max_frames (int): 最多提取的帧数
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    
    frame_count = 0
    success = True
    
    while frame_count < max_frames and success:
        success, frame = cap.read()
        if not success:
            break
        
        # 从 start_frame 开始提取
        if frame_count >= start_frame:
            frame_idx = frame_count - start_frame + 1
            output_path = os.path.join(output_dir, f"im{frame_idx}.png")
            cv2.imwrite(output_path, frame)
        
        frame_count += 1
    
    cap.release()
    return True


def batch_extract_videos(input_dir, output_dir, max_frames=7):
    """
    批量从视频目录提取帧
    
    支持两种输入格式：
    1. 单个视频文件目录: 
       input_dir/
       ├── video1.mp4
       ├── video2.mp4
       └── ...
    
    2. 分类视频目录（Vimeo90K 格式）:
       input_dir/
       ├── train/
       │   ├── video1.mp4
       │   └── ...
       └── test/
           └── ...
    
    输出格式（Vimeo90K 兼容）:
       output_dir/
       ├── sequences/
       │   ├── 00001/
       │   │   ├── im1.png
       │   │   ├── im2.png
       │   │   └── ...
       │   └── ...
    """
    
    os.makedirs(output_dir, exist_ok=True)
    sequences_dir = os.path.join(output_dir, "sequences")
    os.makedirs(sequences_dir, exist_ok=True)
    
    # 收集所有视频文件
    video_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
                video_files.append(os.path.join(root, file))
    
    if not video_files:
        print(f"错误: 在 {input_dir} 中未找到视频文件")
        return False
    
    print(f"找到 {len(video_files)} 个视频文件")
    
    # 按 Vimeo90K 格式组织输出
    seq_idx = 1
    
    for video_path in tqdm(video_files, desc="提取帧"):
        seq_name = f"{seq_idx:05d}"  # 格式: 00001, 00002, ...
        seq_output_dir = os.path.join(sequences_dir, seq_name)
        
        if extract_frames_from_video(video_path, seq_output_dir, max_frames=max_frames):
            seq_idx += 1
        else:
            print(f"跳过: {video_path}")
    
    print(f"\n完成！共提取 {seq_idx - 1} 个序列")
    print(f"输出目录: {sequences_dir}")
    return True


def create_lr_frames(input_dir, output_dir, downscale_factor=4):
    """
    为现有的 GT 帧创建 LR 版本
    
    Args:
        input_dir (str): GT 帧目录（Vimeo90K 格式）
        output_dir (str): LR 帧输出目录
        downscale_factor (int): 缩放倍数
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    sequences_dir = os.path.join(input_dir, "sequences")
    lr_sequences_dir = os.path.join(output_dir, f"sequences_lrx{downscale_factor}")
    
    if not os.path.isdir(sequences_dir):
        print(f"错误: {sequences_dir} 不存在")
        return False
    
    os.makedirs(lr_sequences_dir, exist_ok=True)
    
    # 遍历所有序列
    for seq_name in tqdm(os.listdir(sequences_dir), desc="生成 LR 帧"):
        seq_path = os.path.join(sequences_dir, seq_name)
        if not os.path.isdir(seq_path):
            continue
        
        lr_seq_dir = os.path.join(lr_sequences_dir, seq_name)
        os.makedirs(lr_seq_dir, exist_ok=True)
        
        # 处理每一帧
        for frame_file in sorted(os.listdir(seq_path)):
            if not frame_file.endswith('.png'):
                continue
            
            frame_path = os.path.join(seq_path, frame_file)
            img = cv2.imread(frame_path)
            
            if img is None:
                continue
            
            # 缩放
            h, w = img.shape[:2]
            new_h, new_w = h // downscale_factor, w // downscale_factor
            lr_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # 保存
            lr_frame_path = os.path.join(lr_seq_dir, frame_file)
            cv2.imwrite(lr_frame_path, lr_img)
    
    print(f"完成！LR 帧已保存到: {lr_sequences_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="为 Early Fusion ESPCN 提取和准备视频帧数据"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["extract", "create_lr"],
        default="extract",
        help="执行模式: extract (提取帧) 或 create_lr (生成 LR 帧)"
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="输入目录 (视频文件 或 GT 帧目录)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录"
    )
    
    parser.add_argument(
        "--max_frames",
        type=int,
        default=7,
        help="每个序列的最大帧数 (默认 7)"
    )
    
    parser.add_argument(
        "--downscale_factor",
        type=int,
        default=4,
        help="LR 缩放倍数 (默认 4)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "extract":
        print(f"从视频提取帧...")
        print(f"输入目录: {args.input_dir}")
        print(f"输出目录: {args.output_dir}")
        print(f"每个序列帧数: {args.max_frames}")
        batch_extract_videos(args.input_dir, args.output_dir, args.max_frames)
    
    elif args.mode == "create_lr":
        print(f"生成 LR 帧...")
        print(f"输入目录: {args.input_dir}")
        print(f"输出目录: {args.output_dir}")
        print(f"缩放倍数: x{args.downscale_factor}")
        create_lr_frames(args.input_dir, args.output_dir, args.downscale_factor)


if __name__ == "__main__":
    main()
