#!/usr/bin/env python3
"""
Vimeo90K 标准格式支持 - 使用说明

本脚本演示如何使用新的 Vimeo90K 标准格式支持。

标准格式目录结构：
data/vimeo90k/
├── sequences/                  # GT 训练集
│   ├── 00001/
│   │   ├── 0001/
│   │   │   ├── im1.png
│   │   │   ├── im2.png
│   │   │   └── ...
│   │   └── ...
│   └── ...
├── test/
│   └── sequences/              # GT 测试集
└── (LR 版本由工具自动生成)

使用步骤：

1️⃣ 一键设置（推荐）
   python setup_vimeo90k_test.py --data_dir ./data/vimeo90k --max_seq 5

2️⃣ 验证数据
   python verify_vimeo90k.py --data_dir ./data/vimeo90k

3️⃣ 开始训练
   python train.py

4️⃣ 监控训练
   tensorboard --logdir ./samples/logs/ESPCN_x4_EarlyFusion_Vimeo90K

详见文档：
- QUICK_START.md           # 快速参考（5分钟快速上手）
- VIMEO90K_GUIDE.md        # 详细指南（完整说明）
- VIMEO90K_IMPLEMENTATION.md  # 实现细节（技术参考）
- COMPLETION_SUMMARY.md    # 完成总结（变更说明）
"""

import os
import sys
import subprocess

def main():
    print("""
╔════════════════════════════════════════════════════════════════╗
║    Vimeo90K 标准格式支持 - 快速开始                          ║
║    ESPCN Early Fusion v2.0                                   ║
╚════════════════════════════════════════════════════════════════╝

📁 前提条件：
  ✓ 已有标准 Vimeo90K 格式的数据目录
  ✓ ./data/vimeo90k/sequences/          (所有序列都在这里)
  ✓ ./data/vimeo90k/sep_trainlist.txt   (训练序列列表)
  ✓ ./data/vimeo90k/sep_testlist.txt    (测试序列列表)

🚀 三步启动：

【步骤 1】一键设置数据环境
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
命令：
  python setup_vimeo90k_test.py --data_dir ./data/vimeo90k --max_seq 5

功能：
  ✓ 自动生成 LR 版本（4x 下采样）
  ✓ 自动生成列表文件 (sep_trainlist.txt, sep_testlist.txt)
  ✓ 进度实时提示

参数说明：
  --data_dir ./data/vimeo90k    # 数据目录
  --max_seq 5                   # 仅处理前 5 个序列（测试用）
  
生产环境请删除 --max_seq 参数以处理完整数据集。

【步骤 2】验证数据完整性
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
命令：
  python verify_vimeo90k.py --data_dir ./data/vimeo90k

功能：
  ✓ 检查目录结构
  ✓ 验证列表文件
  ✓ 测试数据加载

如果所有检查都通过（✓），可以继续训练。
如果某些检查失败（✗），请根据提示修复问题。

【步骤 3】开始训练
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
命令：
  python train.py

配置说明（config.py）：
  dataset_type = "video"                              # 使用视频数据集
  epochs = 1                                          # 测试模式（1个epoch）
  train_list_file = "./data/vimeo90k/sep_trainlist.txt"  # 训练列表
  test_list_file = "./data/vimeo90k/sep_testlist.txt"    # 测试列表

生产训练：
  1. 修改 config.py 中的 epochs = 200（或更高）
  2. （可选）修改 batch_size = 64 以加快训练
  3. 重新运行 python train.py

⚡ 性能监控
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
命令：
  tensorboard --logdir ./samples/logs/ESPCN_x4_EarlyFusion_Vimeo90K

然后在浏览器打开：http://localhost:6006

📊 预期输出
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
step 1: 数据加载
  ✓ Loading Video Dataset: video
  ✓ Load all datasets successfully.

step 2: 模型构建
  ✓ Build `espcn_x4` model successfully.

step 3: 损失函数和优化器
  ✓ Define all loss functions successfully.
  ✓ Define all optimizer functions successfully.

step 4: 训练
  ✓ Epoch 1/1
    - Training progress: ████████████████████ 100%
    - Train Loss: 0.0XXX
  ✓ Test PSNR: XX.XX dB
  ✓ Test SSIM: 0.XXXX

� 关于下采样
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

START_HERE.py 不会进行下采样，它只是使用说明文档。

Vimeo90K 标准格式说明：
  • 所有序列都在 ./data/vimeo90k/sequences/ 目录中
  • 通过 sep_trainlist.txt 和 sep_testlist.txt 区分训练/测试集
  • 只对测试集序列进行下采样（生成 LR 版本用于评估）

下采样策略：
  • 训练集：运行时动态生成 LR（不需要预先下采样）
  • 测试集：精确下采样 sep_testlist.txt 中列出的具体子序列

设置数据环境：
  python setup_vimeo90k_test.py --data_dir ./data/vimeo90k

results/
├── ESPCN_x4_EarlyFusion_Vimeo90K/
│   ├── g_best.pth.tar              # 最佳模型
│   └── g_last.pth.tar              # 最后一个模型

samples/
├── ESPCN_x4_EarlyFusion_Vimeo90K/
│   └── g_epoch_1.pth.tar
└── logs/
    └── ESPCN_x4_EarlyFusion_Vimeo90K/  # TensorBoard 日志

🔍 常用命令快速查询
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

一键完成所有（推荐）：
  python setup_vimeo90k_test.py --data_dir ./data/vimeo90k --max_seq 5

分步骤执行：
  # 1. 生成 LR 版本
  python downsample_vimeo90k.py --input_dir ./data/vimeo90k/sequences \\
    --output_dir ./data/vimeo90k/sequences_lrx4 --max_seq 5
  
  # 2. 生成列表文件
  python generate_vimeo90k_lists.py --input_dir ./data/vimeo90k/sequences \\
    --output_dir ./data/vimeo90k --max_seq 5

检查数据完整性：
  python verify_vimeo90k.py --data_dir ./data/vimeo90k

开始训练：
  python train.py

查看 TensorBoard：
  tensorboard --logdir ./samples/logs

📖 详细文档
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

选择适合你的文档：

⭐ 最快开始（5 分钟）
   QUICK_START.md
   - 三步启动
   - 常用命令
   - 快速参考

📚 完整指南（30 分钟）
   VIMEO90K_GUIDE.md
   - 详细步骤
   - 参数说明
   - 常见问题

🔧 实现细节（技术参考）
   VIMEO90K_IMPLEMENTATION.md
   - 代码修改
   - API 文档
   - 对比说明

✅ 完成总结（项目概览）
   COMPLETION_SUMMARY.md
   - 变更摘要
   - 功能对比
   - 项目展望

💡 关键配置
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

config.py 中的关键参数：

数据集类型选择：
  dataset_type = "video"          # 使用视频数据集

数据路径（标准 Vimeo90K 格式）：
  train_gt_video_dir = "./data/vimeo90k/sequences"
  test_gt_video_dir = "./data/vimeo90k/test/sequences"
  test_lr_video_dir = "./data/vimeo90k/test/sequences_lrx4"

列表文件（用于精确控制）：
  train_list_file = "./data/vimeo90k/sep_trainlist.txt"
  test_list_file = "./data/vimeo90k/sep_testlist.txt"

训练参数：
  epochs = 1                      # 测试模式
  batch_size = 32                 # 每批样本数
  num_workers = 4                 # 数据加载线程

GPU 和优化：
  device = torch.device("cuda", 0)  # 使用第一个 GPU
  use_amp = True                  # 启用混合精度（加速）
  cudnn.benchmark = True          # 启用 CUDNN 自动优化

🎯 工作流示例
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

快速测试（开发阶段）：
  $ python setup_vimeo90k_test.py --data_dir ./data/vimeo90k --max_seq 5
  $ python verify_vimeo90k.py --data_dir ./data/vimeo90k
  $ python train.py  # 运行 1 个 epoch

完整训练（生产阶段）：
  # 修改 config.py: epochs = 200
  $ python setup_vimeo90k_test.py --data_dir ./data/vimeo90k  # 移除 --max_seq
  $ python verify_vimeo90k.py --data_dir ./data/vimeo90k
  $ python train.py  # 运行 200 个 epoch

自定义配置：
  $ python setup_vimeo90k_test.py --data_dir ./data/vimeo90k \\
    --filter_seq_start 00001 --filter_seq_end 00005  # 只处理 00001-00005
  $ python train.py

🆘 故障排除
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

找不到数据？
  $ ls ./data/vimeo90k/sequences/00001/0001/
  # 应该看到：im1.png im2.png im3.png ...

LR 版本缺失？
  $ python downsample_vimeo90k.py \\
    --input_dir ./data/vimeo90k/sequences \\
    --output_dir ./data/vimeo90k/sequences_lrx4

列表文件为空？
  $ python generate_vimeo90k_lists.py \\
    --input_dir ./data/vimeo90k/sequences \\
    --output_dir ./data/vimeo90k

数据加载失败？
  $ python verify_vimeo90k.py --data_dir ./data/vimeo90k
  # 检查输出中是否有 ✗ 标记

GPU 不可用？
  $ nvidia-smi  # 检查 GPU 驱动
  $ python -c "import torch; print(torch.cuda.is_available())"

🎓 下一步
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 运行快速测试（5 分钟）
   确保整个流程工作正常

2. 调整 config.py 参数
   根据 GPU 内存调整 batch_size

3. 启动完整训练
   修改 epochs = 200+

4. 监控训练进度
   使用 TensorBoard 查看曲线

5. 评估模型效果
   使用 inference.py 进行推理测试

🔗 快速链接
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

工具脚本：
  • setup_vimeo90k_test.py       → 一键设置
  • downsample_vimeo90k.py       → 下采样 LR
  • generate_vimeo90k_lists.py   → 生成列表
  • verify_vimeo90k.py           → 验证数据

文档：
  • QUICK_START.md               → 快速开始 ⭐
  • VIMEO90K_GUIDE.md            → 详细指南
  • VIMEO90K_IMPLEMENTATION.md   → 实现细节
  • COMPLETION_SUMMARY.md        → 完成总结

训练和推理：
  • train.py                     → 开始训练
  • inference.py                 → 单图推理
  • inference_video.py           → 视频推理
  • test.py                      → 批量推理

═════════════════════════════════════════════════════════════════

立即开始：

  python setup_vimeo90k_test.py --data_dir ./data/vimeo90k --max_seq 5

祝你使用愉快！🚀

═════════════════════════════════════════════════════════════════
    """)

if __name__ == "__main__":
    main()
