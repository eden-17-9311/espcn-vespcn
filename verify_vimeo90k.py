#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vimeo90K 数据集验证脚本

验证：
1. 目录结构是否正确
2. 列表文件是否有效
3. 帧文件是否完整
4. 数据是否可被加载
"""

import os
import sys
import argparse
from pathlib import Path


def verify_directory_structure(data_dir):
    """验证目录结构"""
    print("\n" + "="*60)
    print("1. 验证目录结构")
    print("="*60)
    
    required_dirs = [
        "sequences",
        "sequences_lrx4",
        "test/sequences",
        "test/sequences_lrx4"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        full_path = os.path.join(data_dir, dir_path)
        exists = os.path.isdir(full_path)
        status = "✓" if exists else "✗"
        print(f"{status} {dir_path:<25} {full_path}")
        all_exist = all_exist and exists
    
    return all_exist


def verify_list_files(data_dir):
    """验证列表文件"""
    print("\n" + "="*60)
    print("2. 验证列表文件")
    print("="*60)
    
    list_files = [
        ("sep_trainlist.txt", "训练列表"),
        ("sep_testlist.txt", "测试列表")
    ]
    
    all_valid = True
    for filename, description in list_files:
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"✗ {description:<15} 不存在: {filepath}")
            all_valid = False
            continue
        
        # 检查内容
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        if not lines:
            print(f"✗ {description:<15} 为空")
            all_valid = False
            continue
        
        print(f"✓ {description:<15} 包含 {len(lines)} 项")
        print(f"  首项: {lines[0]}")
        if len(lines) > 1:
            print(f"  末项: {lines[-1]}")
    
    return all_valid


def verify_sequences(data_dir, list_file, check_limit=5):
    """验证序列文件"""
    print("\n" + "="*60)
    print(f"3. 验证序列文件（检查前 {check_limit} 项）")
    print("="*60)
    
    list_path = os.path.join(data_dir, list_file)
    
    if not os.path.exists(list_path):
        print(f"✗ 列表文件不存在: {list_path}")
        return False
    
    with open(list_path, 'r') as f:
        entries = [line.strip() for line in f if line.strip()]
    
    all_valid = True
    
    for i, entry in enumerate(entries[:check_limit]):
        parts = entry.split('/')
        if len(parts) != 2:
            print(f"✗ 第 {i+1} 项格式错误: {entry}")
            all_valid = False
            continue
        
        seq_dir, sub_dir = parts
        
        # 检查 GT 目录
        gt_dir = os.path.join(data_dir, "sequences" if "sep_trainlist" in list_file else "test/sequences", 
                              seq_dir, sub_dir)
        if not os.path.isdir(gt_dir):
            print(f"✗ 第 {i+1} 项GT目录不存在: {gt_dir}")
            all_valid = False
            continue
        
        # 检查帧文件
        frames = sorted([f for f in os.listdir(gt_dir) if f.startswith('im') and f.endswith('.png')])
        if len(frames) < 3:
            print(f"✗ 第 {i+1} 项帧数不足 ({len(frames)} < 3): {gt_dir}")
            all_valid = False
            continue
        
        # 检查 LR 目录
        lr_dir = os.path.join(data_dir, "sequences_lrx4" if "sep_trainlist" in list_file else "test/sequences_lrx4",
                              seq_dir, sub_dir)
        lr_frames = []
        if os.path.isdir(lr_dir):
            lr_frames = sorted([f for f in os.listdir(lr_dir) if f.startswith('im') and f.endswith('.png')])
        
        if len(lr_frames) != len(frames):
            print(f"⚠ 第 {i+1} 项LR帧数不匹配 (GT:{len(frames)} vs LR:{len(lr_frames)}): {entry}")
            all_valid = False
            continue
        
        print(f"✓ 第 {i+1:2} 项正常: {entry:<15} ({len(frames)} 帧)")
    
    if len(entries) > check_limit:
        print(f"  ... 以及其他 {len(entries) - check_limit} 项 (未检查)")
    
    return all_valid


def verify_data_loading(data_dir):
    """验证数据加载（需要 PyTorch）"""
    print("\n" + "="*60)
    print("4. 验证数据加载 (仅训练集)")
    print("="*60)
    
    try:
        import torch
        from dataset import TrainValidVideoDataset
        
        # 尝试加载数据集
        dataset = TrainValidVideoDataset(
            gt_video_dir=os.path.join(data_dir, "sequences"),
            gt_image_size=68,
            upscale_factor=4,
            mode="Train",
            num_frames=3,
            file_list=os.path.join(data_dir, "sep_trainlist.txt")
        )
        
        print(f"✓ 数据集加载成功")
        print(f"  总样本数: {len(dataset)}")
        
        # 尝试获取一个样本
        try:
            sample = dataset[0]
            print(f"✓ 样本获取成功")
            print(f"  GT 形状: {sample['gt'].shape}")
            print(f"  LR 形状: {sample['lr'].shape}")
            return True
        except Exception as e:
            print(f"✗ 样本获取失败: {e}")
            return False
    
    except ImportError:
        print("⚠ 需要 PyTorch 才能完全验证（已跳过）")
        return True
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Vimeo90K 数据集验证脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 验证完整数据集
  python verify_vimeo90k.py --data_dir ./data/vimeo90k
  
  # 只验证目录结构（快速检查）
  python verify_vimeo90k.py --data_dir ./data/vimeo90k --quick
        """
    )
    
    parser.add_argument('--data_dir', type=str, default='./data/vimeo90k',
                        help='Vimeo90K 数据目录（默认: ./data/vimeo90k）')
    parser.add_argument('--quick', action='store_true',
                        help='快速检查（仅验证目录结构）')
    parser.add_argument('--check_limit', type=int, default=5,
                        help='检查的序列数量（默认: 5）')
    
    args = parser.parse_args()
    
    # 检查数据目录
    if not os.path.isdir(args.data_dir):
        print(f"❌ 错误：数据目录不存在: {args.data_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"Vimeo90K 数据集验证")
    print(f"{'='*60}")
    print(f"数据目录: {args.data_dir}\n")
    
    results = {}
    
    # 1. 验证目录结构
    results['structure'] = verify_directory_structure(args.data_dir)
    
    if args.quick:
        print(f"\n{'='*60}")
        print("快速检查完成")
        print(f"{'='*60}\n")
        return
    
    # 2. 验证列表文件
    results['lists'] = verify_list_files(args.data_dir)
    
    # 3. 验证序列文件
    results['sequences_train'] = verify_sequences(args.data_dir, "sep_trainlist.txt", args.check_limit)
    results['sequences_test'] = verify_sequences(args.data_dir, "sep_testlist.txt", args.check_limit)
    
    # 4. 验证数据加载
    results['loading'] = verify_data_loading(args.data_dir)
    
    # 总结
    print(f"\n{'='*60}")
    print("验证总结")
    print(f"{'='*60}")
    
    all_passed = all(results.values())
    
    for check_name, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}")
    
    print(f"{'='*60}\n")
    
    if all_passed:
        print("✓ 所有检查通过！数据集可以用于训练。")
        print("\n运行命令：")
        print("  python train.py")
    else:
        print("✗ 某些检查失败了。请根据上面的提示修复问题。")


if __name__ == "__main__":
    main()
