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
  
  # 处理序列 00001 到 00005
  python setup_vimeo90k_test.py --data_dir ./data/vimeo90k \\
                                --filter_seq_start 00001 \\
                                --filter_seq_end 00005
        """
    )
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Vimeo90K 数据目录（应包含 sequences 子目录）')
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
    
    args = parser.parse_args()
    
    # 检查数据目录
    if not os.path.isdir(args.data_dir):
        print(f"❌ 错误：数据目录不存在: {args.data_dir}")
        return
    
    sequences_dir = os.path.join(args.data_dir, "sequences")
    
    if not os.path.isdir(sequences_dir):
        print(f"❌ 错误：sequences 目录不存在: {sequences_dir}")
        return
    
    sequences_lr_dir = os.path.join(args.data_dir, "sequences_lrx4")
    # 注意：标准Vimeo90K格式中，测试数据也从sequences目录读取，通过列表文件区分
    
    print(f"\n{'='*60}")
    print(f"Vimeo90K 测试环境快速设置")
    print(f"{'='*60}")
    print(f"数据目录: {args.data_dir}")
    print(f"GT 序列: {sequences_dir}")
    print(f"LR 输出: {sequences_lr_dir}")
    print(f"注: 测试数据也从 {sequences_dir} 读取，通过列表文件区分")
    
    if args.max_seq:
        print(f"最大序列数: {args.max_seq}")
    if args.filter_seq_start or args.filter_seq_end:
        print(f"序列范围: {args.filter_seq_start or '开始'} ~ {args.filter_seq_end or '结束'}")
    
    success = True
    
    # ============ Step 1: 生成测试集 LR ============
    if not args.skip_downsample:
        # 从测试列表文件中读取需要下采样的序列
        test_list_file = os.path.join(args.data_dir, "sep_testlist.txt")
        test_sequences = []
        
        if os.path.exists(test_list_file):
            with open(test_list_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # 格式: 00001/0266 -> 只取序列号 00001
                        seq_id = line.split('/')[0]
                        if seq_id not in test_sequences:
                            test_sequences.append(seq_id)
        
        print(f"发现 {len(test_sequences)} 个测试序列需要下采样")
        
        cmd = [
            sys.executable, "downsample_vimeo90k.py",
            "--input_dir", sequences_dir,
            "--output_dir", sequences_lr_dir,
            "--downscale_factor", "4"
        ]
        
        # 只对测试序列进行下采样
        if test_sequences:
            # 按序列排序并设置范围
            test_sequences.sort()
            cmd.extend(["--filter_seq_start", test_sequences[0]])
            cmd.extend(["--filter_seq_end", test_sequences[-1]])
        
        success = run_command(cmd, f"生成测试集 LR 版本 (4x 下采样) - {len(test_sequences)} 个序列") and success
    else:
        print("\n⊘ 跳过下采样步骤")
    
    # ============ Step 2: 生成列表文件 ============
    if not args.skip_lists:
        # 检查列表文件是否已存在
        train_list_file = os.path.join(args.data_dir, "sep_trainlist.txt")
        test_list_file = os.path.join(args.data_dir, "sep_testlist.txt")
        
        if os.path.exists(train_list_file) and os.path.exists(test_list_file):
            print(f"\n✓ 列表文件已存在，跳过生成步骤")
            print(f"  训练列表: {train_list_file}")
            print(f"  测试列表: {test_list_file}")
        else:
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
        print(f"  test_gt_video_dir = '{sequences_dir}'  # 测试也从同一目录")
        print(f"  test_lr_video_dir = '{sequences_lr_dir}'")
        print(f"  train_list_file = '{os.path.join(args.data_dir, 'sep_trainlist.txt')}'")
        print(f"  test_list_file = '{os.path.join(args.data_dir, 'sep_testlist.txt')}'")
    else:
        print("❌ 某些步骤失败了")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
