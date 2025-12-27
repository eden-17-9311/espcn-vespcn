#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
性能优化配置预设
根据你的硬件选择合适的配置
"""

import torch

# 检测 GPU 信息
def get_gpu_info():
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  显存: {props.total_memory / 1e9:.2f} GB")
            print(f"  计算能力: {props.major}.{props.minor}")
    else:
        print("❌ 未检测到 CUDA GPU")


def print_config_presets():
    """打印预设配置"""
    presets = {
        "快速实验": {
            "batch_size": 32,
            "epochs": 500,
            "use_amp": True,
            "num_workers": 4,
            "gradient_accumulation_steps": 1,
            "说明": "最快收敛，用于快速实验和调试"
        },
        "平衡配置": {
            "batch_size": 16,
            "epochs": 2000,
            "use_amp": True,
            "num_workers": 4,
            "gradient_accumulation_steps": 1,
            "说明": "速度与质量平衡，适合大多数场景"
        },
        "显存充足(>8GB)": {
            "batch_size": 64,
            "epochs": 1000,
            "use_amp": True,
            "num_workers": 8,
            "gradient_accumulation_steps": 1,
            "说明": "大batch size，最快收敛"
        },
        "显存紧张(<4GB)": {
            "batch_size": 8,
            "epochs": 3000,
            "use_amp": True,
            "num_workers": 2,
            "gradient_accumulation_steps": 2,
            "说明": "使用梯度累积模拟大batch size"
        },
    }
    
    print("\n" + "=" * 60)
    print("训练配置预设")
    print("=" * 60)
    
    for name, config in presets.items():
        print(f"\n📌 {name}:")
        print(f"   说明: {config.pop('说明')}")
        for key, value in config.items():
            print(f"   {key}: {value}")
    
    print("\n" + "=" * 60)
    print("使用方法: 将上述参数复制到 config.py 的训练部分")
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Early Fusion ESPCN - GPU 性能检测和优化")
    print("=" * 60)
    
    get_gpu_info()
    print_config_presets()
    
    print("\n📖 详细优化指南请查看: TRAINING_ACCELERATION_GUIDE.md\n")
