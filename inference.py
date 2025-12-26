# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import os

import cv2
import torch
import numpy as np
from torch import nn

import config
import imgproc
import model
from utils import load_state_dict

model_names = sorted(
    name for name in model.__dict__ if
    name.islower() and not name.startswith("__") and callable(model.__dict__[name]))


def choice_device(device_type: str) -> torch.device:
    # Select model processing equipment type
    if device_type == "cuda":
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")
    return device


def build_model(model_arch_name: str, device: torch.device) -> nn.Module:
    # Initialize the super-resolution model
    # Early Fusion ESPCN: in_channels=3 (多帧融合)
    sr_model = model.__dict__[model_arch_name](in_channels=3,
                                               out_channels=1,
                                               channels=64)
    sr_model = sr_model.to(device=device)

    return sr_model


def main(args):
    device = choice_device(args.device_type)

    # Initialize the model
    sr_model = build_model(args.model_arch_name, device)
    print(f"Build `{args.model_arch_name}` model successfully.")

    # Load model weights
    sr_model = load_state_dict(sr_model, args.model_weights_path)
    print(f"Load `{args.model_arch_name}` model weights `{os.path.abspath(args.model_weights_path)}` successfully.")

    # Start the verification mode of the model.
    sr_model.eval()

    lr_y_tensor, lr_cb_image, lr_cr_image = imgproc.preprocess_one_image(args.inputs_path, device)

    bic_cb_image = cv2.resize(lr_cb_image,
                              (int(lr_cb_image.shape[1] * args.upscale_factor),
                               int(lr_cb_image.shape[0] * args.upscale_factor)),
                              interpolation=cv2.INTER_CUBIC)
    bic_cr_image = cv2.resize(lr_cr_image,
                              (int(lr_cr_image.shape[1] * args.upscale_factor),
                               int(lr_cr_image.shape[0] * args.upscale_factor)),
                              interpolation=cv2.INTER_CUBIC)
    # ==================== Early Fusion ====================
    # 构建多帧输入用于推理
    # lr_y_tensor 形状: [1, 1, H, W]
    lr_y_np = lr_y_tensor.squeeze().cpu().numpy()  # [H, W]
    
    # 使用高斯模糊模拟相邻帧
    frame_prev = cv2.GaussianBlur(lr_y_np, (3, 3), 1.0)  # 前一帧
    frame_curr = lr_y_np  # 当前帧
    frame_next = cv2.GaussianBlur(lr_y_np, (5, 5), 1.0)  # 后一帧
    
    # 堆叠成 [1, 3, H, W]
    lr_multi_frame = np.stack([frame_prev, frame_curr, frame_next], axis=0)
    lr_multi_frame = torch.from_numpy(lr_multi_frame).unsqueeze(0).to(device)
    # ==================== Early Fusion 结束 ====================
    
    # Use the model to generate super-resolved images
    with torch.no_grad():
        sr_y_tensor = sr_model(lr_multi_frame)

    # Save image
    sr_y_image = imgproc.tensor_to_image(sr_y_tensor, range_norm=False, half=False)
    sr_y_image = sr_y_image.astype(np.float32) / 255.0

    sr_ycbcr_image = cv2.merge([sr_y_image[:, :, 0], bic_cb_image, bic_cr_image])
    sr_image = imgproc.ycbcr_to_bgr(sr_ycbcr_image)
    cv2.imwrite(args.output_path, sr_image * 255.0)

    print(f"SR image save to `{args.output_path}`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Using the model generator super-resolution images.")
    parser.add_argument("--model_arch_name",
                        type=str,
                        default="espcn_x4")
    parser.add_argument("--upscale_factor",
                        type=int,
                        default=4)
    parser.add_argument("--inputs_path",
                        type=str,
                        default="./figure/comic.png",
                        help="Low-resolution image path.")
    parser.add_argument("--output_path",
                        type=str,
                        default="./figure/sr_comic.png",
                        help="Super-resolution image path.")
    parser.add_argument("--model_weights_path",
                        type=str,
                        default="./results/pretrained_models/ESPCN_x4-T91-64bf5ee4.pth.tar",
                        help="Model weights file path.")
    parser.add_argument("--device_type",
                        type=str,
                        default="cpu",
                        choices=["cpu", "cuda"])
    args = parser.parse_args()

    main(args)
