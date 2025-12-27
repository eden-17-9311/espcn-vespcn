#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速验证数据加载和模型的脚本
检查多帧数据是否能正确加载和处理
"""

import torch
import config
import model as model_module
from dataset import TrainValidVideoDataset, TestVideoDataset
from torch.utils.data import DataLoader


def main():
    print("=" * 60)
    print("Early Fusion ESPCN 数据验证脚本")
    print("=" * 60)
    
    # 1. 检查配置
    print(f"\n[配置检查]")
    print(f"  dataset_type: {config.dataset_type}")
    print(f"  num_frames: {config.num_frames}")
    print(f"  in_channels: {config.in_channels}")
    print(f"  upscale_factor: {config.upscale_factor}")
    
    if config.dataset_type != "video":
        print(f"  ⚠️  当前配置为 {config.dataset_type} 模式，跳过视频数据测试")
        return
    
    # 2. 测试数据加载
    print(f"\n[数据加载测试]")
    try:
        train_dataset = TrainValidVideoDataset(
            config.train_gt_video_dir,
            config.gt_image_size,
            config.upscale_factor,
            "Train",
            num_frames=config.num_frames
        )
        print(f"  ✓ 训练数据集加载成功")
        print(f"    - 序列数: {len(train_dataset.sequences)}")
        print(f"    - 总样本数: {len(train_dataset)}")
        
        if len(train_dataset) > 0:
            # 取第一个样本
            sample = train_dataset[0]
            print(f"    - 样本 LR 形状: {sample['lr'].shape}")
            print(f"    - 样本 GT 形状: {sample['gt'].shape}")
            
            # 检查形状
            assert sample['lr'].shape[0] == config.num_frames, f"LR 通道数不匹配！"
            assert sample['gt'].shape[0] == config.num_frames, f"GT 通道数不匹配！"
            print(f"  ✓ 数据形状验证通过")
        else:
            print(f"  ⚠️  训练数据集为空，请检查数据路径")
            return
            
    except Exception as e:
        print(f"  ✗ 数据加载失败: {e}")
        return
    
    # 3. 测试 DataLoader
    print(f"\n[DataLoader 测试]")
    try:
        dataloader = DataLoader(
            train_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        batch = next(iter(dataloader))
        print(f"  ✓ DataLoader 加载成功")
        print(f"    - 批次 LR 形状: {batch['lr'].shape}")
        print(f"    - 批次 GT 形状: {batch['gt'].shape}")
        
        # 验证
        assert batch['lr'].shape[1] == config.num_frames, "LR 帧数错误"
        assert batch['gt'].shape[1] == config.num_frames, "GT 帧数错误"
        print(f"  ✓ 批次形状验证通过")
        
    except Exception as e:
        print(f"  ✗ DataLoader 加载失败: {e}")
        return
    
    # 4. 测试模型
    print(f"\n[模型测试]")
    try:
        sr_model = model_module.__dict__[config.model_arch_name](
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            channels=config.channels
        )
        sr_model = sr_model.to(device=config.device)
        print(f"  ✓ 模型构建成功: {config.model_arch_name}")
        
        # 测试前向传播
        lr_batch = batch['lr'].to(config.device)
        with torch.no_grad():
            sr_output = sr_model(lr_batch)
        
        print(f"    - 输入形状: {lr_batch.shape}")
        print(f"    - 输出形状: {sr_output.shape}")
        
        # 验证输出
        assert sr_output.shape[0] == 2, "输出批次大小错误"
        assert sr_output.shape[1] == 1, f"输出通道应该是 1，得到 {sr_output.shape[1]}"
        print(f"  ✓ 模型前向传播验证通过")
        
    except Exception as e:
        print(f"  ✗ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. 测试中心帧提取
    print(f"\n[中心帧提取测试]")
    try:
        gt = batch['gt']
        if gt.shape[1] > 1:
            center_frame_idx = config.num_frames // 2
            gt_center = gt[:, center_frame_idx:center_frame_idx+1, :, :]
            print(f"  ✓ 中心帧提取成功")
            print(f"    - 原 GT 形状: {gt.shape}")
            print(f"    - 中心帧索引: {center_frame_idx}")
            print(f"    - 提取后形状: {gt_center.shape}")
            
            # 验证
            assert gt_center.shape[1] == 1, "提取后应该是单帧"
            print(f"  ✓ 中心帧维度正确")
        else:
            print(f"  ℹ️  单帧数据，无需提取")
            
    except Exception as e:
        print(f"  ✗ 中心帧提取失败: {e}")
        return
    
    print(f"\n" + "=" * 60)
    print("✓ 所有测试通过！可以开始训练")
    print("=" * 60)


if __name__ == "__main__":
    main()
