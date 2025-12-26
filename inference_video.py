import argparse
import os
import time
import cv2
import torch
import numpy as np
import model
from utils import load_state_dict

def main(args):
    # ==================================================================
    # 1. 初始化模型与设备
    # ==================================================================
    if args.device_type == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda", 0)
        print("使用设备: CUDA (GPU)")
    else:
        device = torch.device("cpu")
        print("使用设备: CPU")

    # 构建模型
    # Early Fusion ESPCN: in_channels=3 (多帧融合)
    sr_model = model.__dict__[args.model_arch_name](in_channels=3, out_channels=1, channels=64)
    sr_model = sr_model.to(device=device)

    # 加载权重
    sr_model = load_state_dict(sr_model, args.model_weights_path)
    sr_model.eval()

    # [优化] 开启半精度推理 (FP16)，RTX 3060 建议开启
    if args.half and device.type == 'cuda':
        sr_model.half()
        print("已启用半精度 (FP16) 加速")

    # [优化] 开启 cudnn benchmark，针对固定尺寸输入有优化
    torch.backends.cudnn.benchmark = True

    # ==================================================================
    # 2. 视频流准备
    # ==================================================================
    cap = cv2.VideoCapture(args.inputs_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {args.inputs_path}")
        return

    # 读取视频参数
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 计算目标尺寸
    target_w = int(width * args.upscale_factor)
    target_h = int(height * args.upscale_factor)

    # 初始化视频写入器
    # 在 Linux 上推荐使用 'mp4v' 或 'avc1'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_path, fourcc, fps, (target_w, target_h))

    print(f"{'='*30}")
    print(f"原分辨率: {width}x{height}")
    print(f"目标分辨率: {target_w}x{target_h}")
    print(f"总帧数: {total_frames}")
    print(f"{'='*30}")

    # ==================================================================
    # 3. 高速推理循环（Early Fusion: 多帧处理）
    # ==================================================================
    frame_count = 0
    start_time = time.time()
    
    # 缓存前后帧用于 Early Fusion
    frame_buffer = []  # 存储 [前一帧, 当前帧, 后一帧]

    with torch.no_grad():
        # 预读第一帧
        ret, prev_frame = cap.read()
        if not ret:
            print("错误：无法读取第一帧")
            return
        
        # 转 Y 通道
        img_ycrcb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2YCrCb)
        prev_y, _, _ = cv2.split(img_ycrcb)
        
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break

            # 转 Y 通道并归一化
            img_ycrcb = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2YCrCb)
            curr_y, cr, cb = cv2.split(img_ycrcb)
            
            # 预读下一帧（用于 Early Fusion）
            ret_next, next_frame = cap.read()
            if ret_next:
                img_ycrcb_next = cv2.cvtColor(next_frame, cv2.COLOR_BGR2YCrCb)
                next_y, _, _ = cv2.split(img_ycrcb_next)
            else:
                # 如果没有下一帧，使用当前帧
                next_y = curr_y

            # ==================== Early Fusion ====================
            # 构建多帧输入：[前一帧, 当前帧, 后一帧]
            prev_y_norm = prev_y.astype(np.float32) / 255.0
            curr_y_norm = curr_y.astype(np.float32) / 255.0
            next_y_norm = next_y.astype(np.float32) / 255.0
            
            # 堆叠成 [3, H, W]
            tensor_multi_frame = torch.from_numpy(
                np.stack([prev_y_norm, curr_y_norm, next_y_norm], axis=0)
            ).unsqueeze(0).to(device, non_blocking=True)  # [1, 3, H, W]
            
            if args.half and device.type == 'cuda':
                tensor_multi_frame = tensor_multi_frame.half()

            # --- 模型推理 ---
            out_tensor = sr_model(tensor_multi_frame)
            # ==================== Early Fusion 结束 ====================

            # [优化3] 后处理：转回 CPU 并恢复为 uint8
            out_y = out_tensor.squeeze().float().cpu().numpy()
            out_y = (out_y * 255.0).clip(0, 255).astype(np.uint8)

            # [优化4] 色度通道缩放：直接使用 OpenCV 的双三次插值 (CPU 多核优化)
            out_cr = cv2.resize(cr, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            out_cb = cv2.resize(cb, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

            # 合并通道并转回 BGR
            out_img_ycrcb = cv2.merge([out_y, out_cr, out_cb])
            out_frame = cv2.cvtColor(out_img_ycrcb, cv2.COLOR_YCrCb2BGR)

            # 写入视频
            out.write(out_frame)

            frame_count += 1
            prev_y = curr_y  # 更新前一帧
            
            # 每 10 帧打印一次进度，减少 I/O 刷新
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed
                print(f"\r进度: {frame_count}/{total_frames} | FPS: {current_fps:.2f}", end="")
            
            # 如果没有下一帧，退出
            if not ret_next:
                break

    # ==================================================================
    # 4. 统计结果
    # ==================================================================
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time
    
    if device.type == 'cuda':
        max_mem = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        mem_info = f" | 显存峰值: {max_mem:.2f} MB"
    else:
        mem_info = ""

    print(f"\n\n{'='*30}")
    print(f"处理完成！")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均 FPS: {avg_fps:.2f}{mem_info}")
    print(f"文件保存至: {args.output_path}")
    print(f"{'='*30}")

    cap.release()
    out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESPCN 高速视频推理")
    parser.add_argument("--model_arch_name", type=str, default="espcn_x2", 
                        help="模型名称 (espcn_x2, espcn_x3, espcn_x4)")
    parser.add_argument("--upscale_factor", type=int, default=2, 
                        help="放大倍率")
    parser.add_argument("--inputs_path", type=str, required=True, 
                        help="输入视频路径")
    parser.add_argument("--output_path", type=str, default="output_video.mp4", 
                        help="输出视频路径")
    parser.add_argument("--model_weights_path", type=str, required=True, 
                        help="模型权重路径")
    parser.add_argument("--device_type", type=str, default="cuda", 
                        choices=["cpu", "cuda"], help="推理设备")
    parser.add_argument("--half", action="store_true", 
                        help="开启 FP16 半精度推理 (推荐 RTX 显卡开启)")
    
    args = parser.parse_args()
    main(args)