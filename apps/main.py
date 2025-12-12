#!/usr/bin/env python3
"""
使用训练好的 YOLO 分割模型对图像进行分割，并将结果保存到 out/ 文件夹。
支持单张图像或整个目录。
"""

import argparse
from calendar import c
from pathlib import Path
import sys

sys.path.insert(0, ".")
from src.segmentation import load_model, segment_image
from src.color_analysis import extract_colors_from_patch
from src.ph_measurement import calculate_ph_value


def ensure_dir(path):
    """确保目录存在"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="YOLO 图像分割与pH值计算")
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
    args = parser.parse_args()

    # 确保输出目录存在
    ensure_dir(args.output)

    # 加载模型
    model = load_model(args.model)

    # 处理输入
    input_path = Path(args.input)
    if input_path.is_file():
        # 单张图像
        for obj in segment_image(model, input_path, args.conf):
            colored_color, uncolored_color = extract_colors_from_patch(
                obj["cropped_bgra"]
            )
            if colored_color is not None and uncolored_color is not None:
                pH_value = calculate_ph_value(
                    colored_color, uncolored_color, color_space="hsv"
                )
            else:
                pH_value = "未知"

            print(
                f"对象 {obj['stem']}: 类别 {int(obj['cls'])}, 置信度 {obj['conf']:.2f}, pH 值: {pH_value}"
            )

    elif input_path.is_dir():
        # 目录下的所有图像文件
        extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
        image_files = [
            f for f in input_path.iterdir() if f.suffix.lower() in extensions
        ]
        print(f"找到 {len(image_files)} 张图像")
        for img_file in image_files:
            for obj in segment_image(model, img_file, args.conf):
                colored_color, uncolored_color = extract_colors_from_patch(
                    obj["cropped_bgra"]
                )

                if colored_color is not None and uncolored_color is not None:
                    pH_value = calculate_ph_value(
                        colored_color, uncolored_color, color_space="hsv"
                    )
                else:
                    pH_value = "未知"

                print(
                    f"./seg/data/images/train/{obj['stem']}.jpg : 置信度 {obj['conf']:.2f}, pH 值: {pH_value:.2f}"
                )
                # if "." in obj["stem"]:
                #     if colored_color is not None and uncolored_color is not None:
                #         print(
                #             f"({colored_color[0]}, {colored_color[1]}, {colored_color[2]},{uncolored_color[0]},{uncolored_color[1]},{uncolored_color[2]}) :{obj['stem']},"
                #         )
    else:
        print(f"输入路径不存在: {args.input}")
        return

    print("处理完成！")


if __name__ == "__main__":
    main()
