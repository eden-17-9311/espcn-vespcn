#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vimeo90K 测试环境快速设置脚本

一键生成：
1. 从 sequences 目录生成对应的 sequences_lrx4 下采样版本
2. 从 test/sequences 目录生成对应的 test/sequences_lrx4 下采样版本
3. 生成 sep_trainlist.txt 和 sep_testlist.txt

使用方法：
python setup_vimeo90k_test.py --data_dir ./data/vimeo90k --max_seq 5
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """运行命令并打印信息"""
    print(f"\n{'='*60}")
    print(f"[Step] {description}")
    print(f"{'='*60}")
    print(f"命令: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, shell=False)
    if result.returncode != 0:
        print(f"❌ 错误：命令执行失败")
        return False
    
    print(f"✓ 完成")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Vimeo90K 测试环境快速设置',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 设置完整测试环境（处理前 5 个序列）
  python setup_vimeo90k_test.py --data_dir ./data/vimeo90k --max_seq 5
  
  # 只对测试集进行下采样（00001-00005）
  python setup_vimeo90k_test.py --data_dir ./data/vimeo90k \\
                                --test_only \\
                                --filter_seq_start 00001 \\
                                --filter_seq_end 00005
  
  # 处理序列 00001 到 00005
  python setup_vimeo90k_test.py --data_dir ./data/vimeo90k \\
                                --filter_seq_start 00001 \\
                                --filter_seq_end 00005
        """
    )
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Vimeo90K 数据目录（应包含 sequences 和 test 子目录）')
    parser.add_argument('--max_seq', type=int, default=None,
                        help='最多处理的序列数（用于快速测试）')
    parser.add_argument('--filter_seq_start', type=str, default=None,
                        help='过滤序列起始（如 00001）')
    parser.add_argument('--filter_seq_end', type=str, default=None,
                        help='过滤序列结束（如 00005）')
    parser.add_argument('--skip_downsample', action='store_true',
                        help='跳过下采样步骤（如果已完成）')
    parser.add_argument('--skip_lists', action='store_true',
                        help='跳过列表生成步骤（如果已完成）')
    parser.add_argument('--test_only', action='store_true',
                        help='只对测试集进行下采样（跳过训练集）')
    
    args = parser.parse_args()
    
    # 检查数据目录
    if not os.path.isdir(args.data_dir):
        print(f"❌ 错误：数据目录不存在: {args.data_dir}")
        return
    
    sequences_dir = os.path.join(args.data_dir, "sequences")
    test_sequences_dir = os.path.join(args.data_dir, "test", "sequences")
    
    if not os.path.isdir(sequences_dir):
        print(f"❌ 错误：sequences 目录不存在: {sequences_dir}")
        return
    
    if not os.path.isdir(test_sequences_dir):
        print(f"❌ 错误：test/sequences 目录不存在: {test_sequences_dir}")
        return
    
    sequences_lr_dir = os.path.join(args.data_dir, "sequences_lrx4")
    test_sequences_lr_dir = os.path.join(args.data_dir, "test", "sequences_lrx4")
    
    print(f"\n{'='*60}")
    print(f"Vimeo90K 测试环境快速设置")
    print(f"{'='*60}")
    print(f"数据目录: {args.data_dir}")
    print(f"GT 序列: {sequences_dir}")
    print(f"LR 输出: {sequences_lr_dir}")
    print(f"测试 GT: {test_sequences_dir}")
    print(f"测试 LR: {test_sequences_lr_dir}")
    
    if args.max_seq:
        print(f"最大序列数: {args.max_seq}")
    if args.filter_seq_start or args.filter_seq_end:
        print(f"序列范围: {args.filter_seq_start or '开始'} ~ {args.filter_seq_end or '结束'}")
    if args.test_only:
        print("模式: 只处理测试集")
    
    success = True
    
    # ============ Step 1: 生成训练集 LR ============
    if not args.skip_downsample and not args.test_only:
        cmd = [
            sys.executable, "downsample_vimeo90k.py",
            "--input_dir", sequences_dir,
            "--output_dir", sequences_lr_dir,
            "--downscale_factor", "4"
        ]
        if args.max_seq:
            cmd.extend(["--max_seq", str(args.max_seq)])
        if args.filter_seq_start:
            cmd.extend(["--filter_seq_start", args.filter_seq_start])
        if args.filter_seq_end:
            cmd.extend(["--filter_seq_end", args.filter_seq_end])
        
        success = run_command(cmd, "生成训练集 LR 版本 (4x 下采样)") and success
    elif args.test_only:
        print("\n⊘ 跳过训练集下采样（测试模式）")
    else:
        print("\n⊘ 跳过下采样步骤")
    
    # ============ Step 2: 生成测试集 LR ============
    if not args.skip_downsample:
        cmd = [
            sys.executable, "downsample_vimeo90k.py",
            "--input_dir", test_sequences_dir,
            "--output_dir", test_sequences_lr_dir,
            "--downscale_factor", "4"
        ]
        if args.filter_seq_start:
            cmd.extend(["--filter_seq_start", args.filter_seq_start])
        if args.filter_seq_end:
            cmd.extend(["--filter_seq_end", args.filter_seq_end])
        
        success = run_command(cmd, "生成测试集 LR 版本 (4x 下采样)") and success
    else:
        print("\n⊘ 跳过测试集下采样步骤")
    
    # ============ Step 3: 生成列表文件 ============
    if not args.skip_lists:
        cmd = [
            sys.executable, "generate_vimeo90k_lists.py",
            "--input_dir", sequences_dir,
            "--output_dir", args.data_dir,
            "--train_ratio", "0.8"
        ]
        if args.max_seq:
            cmd.extend(["--max_seq", str(args.max_seq)])
        if args.filter_seq_start:
            cmd.extend(["--filter_seq_start", args.filter_seq_start])
        if args.filter_seq_end:
            cmd.extend(["--filter_seq_end", args.filter_seq_end])
        
        success = run_command(cmd, "生成训练/测试列表文件") and success
    else:
        print("\n⊘ 跳过列表生成步骤")
    
    # ============ 完成 ============
    print(f"\n{'='*60}")
    if success:
        print("✓ 所有步骤完成！")
        print("\n现在可以运行：")
        print(f"  python train.py")
        print("\n配置说明（config.py）：")
        print(f"  dataset_type = 'video'")
        print(f"  train_gt_video_dir = '{sequences_dir}'")
        print(f"  test_gt_video_dir = '{test_sequences_dir}'")
        print(f"  test_lr_video_dir = '{test_sequences_lr_dir}'")
        print(f"  train_list_file = '{os.path.join(args.data_dir, 'sep_trainlist.txt')}'")
        print(f"  test_list_file = '{os.path.join(args.data_dir, 'sep_testlist.txt')}'")
    else:
        print("❌ 某些步骤失败了")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
