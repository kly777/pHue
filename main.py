#!/usr/bin/env python3
"""
使用训练好的 YOLO 分割模型对图像进行分割，并将结果保存到 out/ 文件夹。
支持单张图像或整个目录。
"""

import argparse
import cv2
import os
import numpy as np
from pathlib import Path
from ultralytics.models import YOLO


def ensure_dir(path):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


def load_model(model_path):
    """加载 YOLO 模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)
    return model


def process_image(
    model,
    image_path,
    output_dir,
    conf_threshold=0.5,
    save_annotated=True,
    save_masks=True,
):
    """处理单张图像，保存分割结果"""
    print(f"处理图像: {image_path}")
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  无法读取图像: {image_path}")
        return

    # 推理
    results = model(img, conf=conf_threshold, verbose=False)

    # 获取基本文件名
    stem = Path(image_path).stem

    # 保存带标注的图像
    if save_annotated:
        annotated = results[0].plot()  # BGR 图像，包含掩码和边界框
        out_path = os.path.join(output_dir, f"{stem}_annotated.jpg")
        cv2.imwrite(out_path, annotated)
        print(f"  保存标注图像: {out_path}")

    # 保存每个检测到的掩码区域
    if save_masks and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()  # (N, H, W) 模型输出尺寸
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []
        classes = results[0].boxes.cls.cpu().numpy() if results[0].boxes else []
        confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes else []

        # 获取原始图像尺寸
        orig_h, orig_w = img.shape[:2]
        # 获取掩码尺寸（模型输出尺寸）
        mask_h, mask_w = masks.shape[1], masks.shape[2]

        for i, (mask, box, cls, conf) in enumerate(zip(masks, boxes, classes, confs)):
            # 将掩码调整到原始图像尺寸
            mask_resized = cv2.resize(
                mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
            )
            # 将掩码转换为二值图像 (0-255)
            mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255

            # 保存掩码图像
            mask_path = os.path.join(output_dir, f"{stem}_mask_{i}.png")
            cv2.imwrite(mask_path, mask_binary)

            # 创建掩码叠加图像（原始图像上高亮显示分割区域）
            # 这里我们生成一个彩色叠加层
            color_mask = np.zeros_like(img, dtype=np.uint8)
            color_mask[mask_resized > 0.5] = (0, 255, 0)  # 绿色
            overlay = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)
            overlay_path = os.path.join(output_dir, f"{stem}_overlay_{i}.jpg")
            cv2.imwrite(overlay_path, overlay)

            # 也可以裁剪出掩码区域（边界框内）
            x1, y1, x2, y2 = map(int, box)
            cropped = img[y1:y2, x1:x2]
            if cropped.size > 0:
                crop_path = os.path.join(output_dir, f"{stem}_crop_{i}.jpg")
                cv2.imwrite(crop_path, cropped)

            print(f"    对象 {i}: 类别 {int(cls)}, 置信度 {conf:.2f}, 掩码已保存")

    # 如果没有检测到任何对象
    if results[0].masks is None:
        print(f"  未检测到分割对象")


def main():
    parser = argparse.ArgumentParser(description="YOLO 图像分割与结果保存")
    parser.add_argument(
        "--model",
        type=str,
        default="seg/weights/best12072154.pt",
        help="模型权重路径 (默认: seg/weights/best12072154.pt)",
    )
    parser.add_argument("--input", type=str, required=True, help="输入图像路径或目录")
    parser.add_argument(
        "--output", type=str, default="out", help="输出目录 (默认: out)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.5, help="置信度阈值 (默认: 0.5)"
    )
    parser.add_argument(
        "--no-annotated", action="store_true", help="不保存带标注的图像"
    )
    parser.add_argument("--no-masks", action="store_true", help="不保存掩码和叠加图像")
    args = parser.parse_args()

    # 确保输出目录存在
    ensure_dir(args.output)

    # 加载模型
    model = load_model(args.model)

    # 处理输入
    input_path = Path(args.input)
    if input_path.is_file():
        # 单张图像
        process_image(
            model,
            input_path,
            args.output,
            args.conf,
            save_annotated=not args.no_annotated,
            save_masks=not args.no_masks,
        )
    elif input_path.is_dir():
        # 目录下的所有图像文件
        extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
        image_files = [
            f for f in input_path.iterdir() if f.suffix.lower() in extensions
        ]
        print(f"找到 {len(image_files)} 张图像")
        for img_file in image_files:
            process_image(
                model,
                img_file,
                args.output,
                args.conf,
                save_annotated=not args.no_annotated,
                save_masks=not args.no_masks,
            )
    else:
        print(f"输入路径不存在: {args.input}")
        return

    print("处理完成！")


if __name__ == "__main__":
    main()
