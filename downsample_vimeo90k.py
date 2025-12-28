#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vimeo90K 下采样脚本：生成 LR 版本

支持标准 Vimeo90K 目录结构：
sequences/
    00001/
        0001/
            im1.png, im2.png, im3.png, ...
        0002/
            ...
    00002/
        ...

使用方法：
python downsample_vimeo90k.py --input_dir ./data/vimeo90k/sequences \
                               --output_dir ./data/vimeo90k/sequences_lrx4 \
                               --downscale_factor 4 \
                               --max_seq 5
"""

import os
import argparse
import cv2
from pathlib import Path
import shutil


def create_lr_frames(gt_dir, lr_dir, downscale_factor=4):
    """
    为 GT 目录创建对应的 LR 版本
    
    Args:
        gt_dir (str): GT 目录路径（包含 im*.png）
        lr_dir (str): LR 输出目录路径
        downscale_factor (int): 下采样倍率
    """
    # 创建输出目录
    os.makedirs(lr_dir, exist_ok=True)
    
    # 获取所有 im*.png 文件
    frame_files = sorted([f for f in os.listdir(gt_dir) if f.startswith('im') and f.endswith('.png')])
    
    for frame_file in frame_files:
        gt_path = os.path.join(gt_dir, frame_file)
        lr_path = os.path.join(lr_dir, frame_file)
        
        # 读取 GT 图像
        img = cv2.imread(gt_path)
        if img is None:
            print(f"警告：无法读取 {gt_path}")
            continue
        
        # 计算 LR 尺寸
        h, w = img.shape[:2]
        lr_h, lr_w = h // downscale_factor, w // downscale_factor
        
        # 使用高质量插值（INTER_CUBIC）进行下采样
        lr_img = cv2.resize(img, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
        
        # 保存 LR 图像
        cv2.imwrite(lr_path, lr_img)
        print(f"✓ {frame_file}: {w}x{h} → {lr_w}x{lr_h}")


def process_vimeo90k(input_dir, output_dir, downscale_factor=4, max_seq=None, filter_seq_start=None, filter_seq_end=None, seq_list=None, subseq_list=None):
    """
    处理 Vimeo90K 格式的目录结构
    
    Args:
        input_dir (str): 输入目录（GT）
        output_dir (str): 输出目录（LR）
        downscale_factor (int): 下采样倍率
        max_seq (int): 处理的最大序列数（用于测试）
        filter_seq_start (str): 过滤序列起始（如 '00001'）
        filter_seq_end (str): 过滤序列结束（如 '00005'）
        seq_list (list): 要处理的序列列表（如 ['00001', '00003', '00005']）
        subseq_list (list): 要处理的子序列列表（如 ['00001/0266', '00001/0268']）
    """
    seq_count = 0
    sub_count = 0
    
    # 遍历一级序列目录（00001, 00002, ...）
    seq_dirs = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    
    # 如果提供了子序列列表，直接处理这些子序列
    if subseq_list:
        print(f"处理指定的 {len(subseq_list)} 个子序列")
        processed_subseqs = set()
        
        for subseq_path in subseq_list:
            if subseq_path in processed_subseqs:
                continue
            processed_subseqs.add(subseq_path)
            
            # 分解路径：00001/0266 -> seq_dir=00001, sub_dir=0266
            parts = subseq_path.split('/')
            if len(parts) != 2:
                print(f"⚠ 跳过无效路径: {subseq_path}")
                continue
                
            seq_dir, sub_dir = parts
            gt_subseq_path = os.path.join(input_dir, seq_dir, sub_dir)
            lr_subseq_path = os.path.join(output_dir, seq_dir, sub_dir)
            
            if os.path.isdir(gt_subseq_path):
                # 检查帧文件
                frames = [f for f in os.listdir(gt_subseq_path) if f.startswith('im') and f.endswith('.png')]
                if frames:
                    print(f"  └─ 处理子序列: {subseq_path} ({len(frames)} 帧)")
                    create_lr_frames(gt_subseq_path, lr_subseq_path, downscale_factor)
                    sub_count += 1
                else:
                    print(f"⚠ 子序列无帧文件: {gt_subseq_path}")
            else:
                print(f"⚠ GT 子序列不存在: {gt_subseq_path}")
        
        seq_count = len(set(s.split('/')[0] for s in subseq_list))
    else:
        # 原来的逻辑：遍历所有目录
        for seq_dir in seq_dirs:
            # 过滤序列
            if seq_list and seq_dir not in seq_list:
                continue
            elif filter_seq_start and seq_dir < filter_seq_start:
                continue
            elif filter_seq_end and seq_dir > filter_seq_end:
                break
            
            seq_path = os.path.join(input_dir, seq_dir)
            output_seq_path = os.path.join(output_dir, seq_dir)
            
            print(f"\n处理序列: {seq_dir}")
            
            # 遍历二级子序列目录（0001, 0002, ...）
            sub_dirs = sorted([d for d in os.listdir(seq_path) if os.path.isdir(os.path.join(seq_path, d))])
            
            for sub_dir in sub_dirs:
                sub_seq_path = os.path.join(seq_path, sub_dir)
                output_sub_seq_path = os.path.join(output_seq_path, sub_dir)
                
                # 检查是否含有 im*.png 文件
                frames = [f for f in os.listdir(sub_seq_path) if f.startswith('im') and f.endswith('.png')]
                if not frames:
                    continue
                
                print(f"  └─ 处理子序列: {seq_dir}/{sub_dir} ({len(frames)} 帧)")
                
                # 创建 LR 版本
                create_lr_frames(sub_seq_path, output_sub_seq_path, downscale_factor)
                
                sub_count += 1
            
            seq_count += 1
            
            # 检查是否达到最大序列数
            if max_seq and seq_count >= max_seq:
                print(f"\n已达到最大序列数 {max_seq}，停止处理")
                break
    
    print(f"\n=============== 下采样完成 ===============")
    print(f"处理序列数: {seq_count}")
    print(f"处理子序列数: {sub_count}")
    print(f"输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Vimeo90K 数据集下采样脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 处理前 5 个序列
  python downsample_vimeo90k.py --input_dir ./data/vimeo90k/sequences \\
                                 --output_dir ./data/vimeo90k/sequences_lrx4 \\
                                 --max_seq 5
  
  # 处理 00001 到 00005
  python downsample_vimeo90k.py --input_dir ./data/vimeo90k/sequences \\
                                 --output_dir ./data/vimeo90k/sequences_lrx4 \\
                                 --filter_seq_start 00001 \\
                                 --filter_seq_end 00005
  
  # 处理测试集
  python downsample_vimeo90k.py --input_dir ./data/vimeo90k/test/sequences \\
                                 --output_dir ./data/vimeo90k/test/sequences_lrx4 \\
                                 --downscale_factor 4
        """
    )
    
    parser.add_argument('--input_dir', type=str, required=True,
                        help='输入目录（GT 图像）')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录（LR 图像）')
    parser.add_argument('--downscale_factor', type=int, default=4,
                        help='下采样倍率（默认 4）')
    parser.add_argument('--max_seq', type=int, default=None,
                        help='最多处理的序列数（用于测试，默认处理全部）')
    parser.add_argument('--filter_seq_start', type=str, default=None,
                        help='过滤序列起始（如 00001）')
    parser.add_argument('--filter_seq_end', type=str, default=None,
                        help='过滤序列结束（如 00005）')
    parser.add_argument('--seq_list', type=str, nargs='*', default=None,
                        help='要处理的序列列表（如 00001 00003 00005）')
    parser.add_argument('--subseq_list', type=str, nargs='*', default=None,
                        help='要处理的子序列列表（如 00001/0266 00001/0268）')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.isdir(args.input_dir):
        print(f"错误：输入目录不存在: {args.input_dir}")
        return
    
    print(f"=============== Vimeo90K 下采样 ===============")
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"下采样倍率: {args.downscale_factor}x")
    if args.max_seq:
        print(f"最大序列数: {args.max_seq}")
    if args.filter_seq_start or args.filter_seq_end:
        print(f"序列范围: {args.filter_seq_start or '开始'} ~ {args.filter_seq_end or '结束'}")
    if args.seq_list:
        print(f"指定序列: {', '.join(args.seq_list)}")
    if args.subseq_list:
        print(f"指定子序列: {', '.join(args.subseq_list)}")
    print("")
    
    # 执行下采样
    process_vimeo90k(
        args.input_dir,
        args.output_dir,
        downscale_factor=args.downscale_factor,
        max_seq=args.max_seq,
        filter_seq_start=args.filter_seq_start,
        filter_seq_end=args.filter_seq_end,
        seq_list=args.seq_list,
        subseq_list=args.subseq_list
    )


if __name__ == "__main__":
    main()
