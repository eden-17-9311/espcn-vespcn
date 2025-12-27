#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 Vimeo90K sep_trainlist.txt 和 sep_testlist.txt

根据目录结构自动生成列表文件，格式：
00001/0001
00001/0002
...

使用方法：
python generate_vimeo90k_lists.py --input_dir ./data/vimeo90k/sequences \
                                   --output_dir ./data/vimeo90k \
                                   --train_ratio 0.8
"""

import os
import argparse
from pathlib import Path


def generate_vimeo90k_lists(input_dir, output_dir, train_ratio=0.8, max_seq=None, filter_seq_start=None, filter_seq_end=None):
    """
    生成 Vimeo90K 训练列表和测试列表
    
    Args:
        input_dir (str): 输入目录（sequences）
        output_dir (str): 输出目录（放置列表文件）
        train_ratio (float): 训练集比例（0-1）
        max_seq (int): 最多处理的序列数
        filter_seq_start (str): 过滤序列起始
        filter_seq_end (str): 过滤序列结束
    """
    train_list = []
    test_list = []
    
    seq_count = 0
    total_sub_count = 0
    
    # 遍历一级序列目录
    seq_dirs = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    
    for seq_dir in seq_dirs:
        # 过滤序列范围
        if filter_seq_start and seq_dir < filter_seq_start:
            continue
        if filter_seq_end and seq_dir > filter_seq_end:
            break
        
        seq_path = os.path.join(input_dir, seq_dir)
        
        # 遍历二级子序列目录
        sub_dirs = sorted([d for d in os.listdir(seq_path) if os.path.isdir(os.path.join(seq_path, d))])
        
        for idx, sub_dir in enumerate(sub_dirs):
            sub_seq_path = os.path.join(seq_path, sub_dir)
            
            # 检查是否含有 im*.png 文件
            frames = [f for f in os.listdir(sub_seq_path) if f.startswith('im') and f.endswith('.png')]
            if not frames:
                continue
            
            # 根据比例分配到训练集或测试集
            entry = f"{seq_dir}/{sub_dir}"
            if idx / len(sub_dirs) < train_ratio:
                train_list.append(entry)
            else:
                test_list.append(entry)
            
            total_sub_count += 1
        
        seq_count += 1
        print(f"处理序列: {seq_dir} ({len(sub_dirs)} 个子序列)")
        
        # 检查是否达到最大序列数
        if max_seq and seq_count >= max_seq:
            print(f"已达到最大序列数 {max_seq}，停止处理")
            break
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存训练列表
    train_list_path = os.path.join(output_dir, "sep_trainlist.txt")
    with open(train_list_path, 'w') as f:
        for entry in train_list:
            f.write(entry + '\n')
    print(f"\n保存训练列表: {train_list_path}")
    print(f"训练样本数: {len(train_list)}")
    
    # 保存测试列表
    test_list_path = os.path.join(output_dir, "sep_testlist.txt")
    with open(test_list_path, 'w') as f:
        for entry in test_list:
            f.write(entry + '\n')
    print(f"保存测试列表: {test_list_path}")
    print(f"测试样本数: {len(test_list)}")
    
    print(f"\n=============== 统计信息 ===============")
    print(f"处理序列数: {seq_count}")
    print(f"处理子序列总数: {total_sub_count}")
    print(f"训练集比例: {train_ratio*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description='生成 Vimeo90K sep_trainlist.txt 和 sep_testlist.txt',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 生成完整列表（前 5 个序列）
  python generate_vimeo90k_lists.py --input_dir ./data/vimeo90k/sequences \\
                                    --output_dir ./data/vimeo90k \\
                                    --max_seq 5
  
  # 生成完整列表（序列 00001 到 00005）
  python generate_vimeo90k_lists.py --input_dir ./data/vimeo90k/sequences \\
                                    --output_dir ./data/vimeo90k \\
                                    --filter_seq_start 00001 \\
                                    --filter_seq_end 00005
  
  # 生成列表（自定义训练/测试比例）
  python generate_vimeo90k_lists.py --input_dir ./data/vimeo90k/sequences \\
                                    --output_dir ./data/vimeo90k \\
                                    --train_ratio 0.9
        """
    )
    
    parser.add_argument('--input_dir', type=str, required=True,
                        help='输入目录（序列目录）')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录（放置列表文件）')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例（默认 0.8）')
    parser.add_argument('--max_seq', type=int, default=None,
                        help='最多处理的序列数（用于测试）')
    parser.add_argument('--filter_seq_start', type=str, default=None,
                        help='过滤序列起始（如 00001）')
    parser.add_argument('--filter_seq_end', type=str, default=None,
                        help='过滤序列结束（如 00005）')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.isdir(args.input_dir):
        print(f"错误：输入目录不存在: {args.input_dir}")
        return
    
    print(f"=============== 生成 Vimeo90K 列表文件 ===============")
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"训练集比例: {args.train_ratio*100:.1f}%")
    if args.max_seq:
        print(f"最大序列数: {args.max_seq}")
    if args.filter_seq_start or args.filter_seq_end:
        print(f"序列范围: {args.filter_seq_start or '开始'} ~ {args.filter_seq_end or '结束'}")
    print("")
    
    # 生成列表
    generate_vimeo90k_lists(
        args.input_dir,
        args.output_dir,
        train_ratio=args.train_ratio,
        max_seq=args.max_seq,
        filter_seq_start=args.filter_seq_start,
        filter_seq_end=args.filter_seq_end
    )


if __name__ == "__main__":
    main()
