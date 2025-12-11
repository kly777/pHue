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


from pHmap import pH_color_map

import tkinter as tk


def show_color_window(color):
    """
    创建窗口显示指定颜色
    :param color: RGB元组 (BGR格式需转换)
    """
    # 转换BGR到RGB（OpenCV使用BGR，tkinter使用RGB）
    r, g, b = int(color[2]), int(color[1]), int(color[0])
    hex_color = f"#{r:02x}{g:02x}{b:02x}"

    root = tk.Tk()
    root.title("pH匹配颜色预览")
    root.geometry("300x200")

    # 颜色显示区域
    color_frame = tk.Frame(root, width=250, height=120, bg=hex_color)
    color_frame.pack(pady=15)

    # 颜色值标签
    tk.Label(
        root, text=f"RGB: ({r}, {g}, {b})\nHEX: {hex_color}", font=("Arial", 12)
    ).pack()

    # 关闭按钮
    tk.Button(root, text="确认", command=root.destroy).pack(pady=5)
    root.mainloop()


def ensure_dir(path):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


def get_ph_value_from_patch(image):
    """从pH试纸图像中获取pH值，通过颜色聚类获取最大的两种颜色，并取距离pH试纸较远的颜色匹配pH值"""
    # 确保输入图像是正确的格式
    if isinstance(image, np.ndarray):
        pixels = image.reshape(-1, 3).astype(np.float32)
    else:
        pixels = np.array(image).reshape(-1, 3).astype(np.float32)

    # 使用k-means聚类将像素分为2类（变色部分和未变色部分）
    n_colors = 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    labels = np.zeros((pixels.shape[0],), dtype=np.int32)
    _, labels, palette = cv2.kmeans(pixels, n_colors, labels, criteria, 10, flags)

    # 计算每个聚类的像素数量
    unique_labels, counts = np.unique(labels, return_counts=True)

    # 找到最大的两个聚类（按像素数量排序）
    sorted_indices = np.argsort(counts)[::-1]
    largest_cluster_idx = sorted_indices[0]
    second_largest_cluster_idx = (
        sorted_indices[1] if len(sorted_indices) > 1 else largest_cluster_idx
    )

    # 获取两个主要颜色
    main_color = palette[largest_cluster_idx]
    secondary_color = palette[second_largest_cluster_idx]
    print(f"  主要颜色: {main_color}")
    print(f"  次主要颜色: {secondary_color}")

    # 假设未变色的部分通常是白色或浅色，计算与白色的欧几里得距离
    white = np.array([255, 255, 255], dtype=np.float32)
    main_color_distance_to_white = np.sqrt(np.sum((main_color - white) ** 2))
    secondary_color_distance_to_white = np.sqrt(np.sum((secondary_color - white) ** 2))

    # 选择距离白色更远的颜色（即变色更明显的部分）
    if main_color_distance_to_white > secondary_color_distance_to_white:
        selected_color = main_color
    else:
        selected_color = secondary_color
    print(f"  匹配颜色: {selected_color}")

    show_color_window(selected_color)

    print(
        f" print for copy:\n ({selected_color[0]:.3f},{selected_color[1]:.3f},{selected_color[2]:.3f}):"
    )

    # 计算与所有已知颜色的距离
    color_distances = []
    for color, pH in pH_color_map.items():
        # 计算欧几里得距离
        distance = np.sqrt(sum((c - d) ** 2 for c, d in zip(selected_color, color)))
        color_distances.append((distance, pH))

    # 按距离排序并取最近的两个
    color_distances.sort()
    closest_two = color_distances[:2]

    # 计算加权平均 pH 值 (权重为距离的倒数)
    if len(closest_two) == 2:
        dist1, pH1 = closest_two[0]
        dist2, pH2 = closest_two[1]
        # 使用距离的倒数作为权重
        weight1 = 1.0 / (dist1 + 1e-8)  # 添加小常数避免除零
        weight2 = 1.0 / (dist2 + 1e-8)
        pH_value = (weight1 * pH1 + weight2 * pH2) / (weight1 + weight2)
    elif len(closest_two) == 1:
        pH_value = closest_two[0][1]  # 只有一个匹配
    else:
        pH_value = "未知"  # 没有匹配

    return pH_value


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

            # 将掩码调整到原始图像尺寸
            mask_resized = cv2.resize(
                mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
            )

            # 提取非零区域的边界框以进行精确裁剪
            coords = cv2.findNonZero((mask_resized > 0.5).astype(np.uint8))
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)

                # 裁剪原始图像和掩码到边界框区域
                cropped_img = img[y : y + h, x : x + w]
                cropped_mask = mask_resized[y : y + h, x : x + w]

                # 创建一个与裁剪后图像大小相同的BGRA图像
                cropped_bgra = np.zeros((h, w, 4), dtype=np.uint8)
                cropped_bgra[:, :, :3] = cropped_img  # 复制裁剪后的BGR图像
                cropped_bgra[:, :, 3] = (cropped_mask > 0.5) * 255  # 设置Alpha通道

                # 保存裁剪后的PNG图像（保留透明度）
                mask_area_path = os.path.join(output_dir, f"{stem}_mask_area_{i}.png")
                cv2.imwrite(mask_area_path, cropped_bgra)

                # 从裁剪后的图像中提取非透明像素用于聚类
                alpha_channel = cropped_bgra[:, :, 3]
                non_transparent_pixels = cropped_bgra[
                    alpha_channel > 0, :3
                ]  # 只取BGR通道

                # 获取pH值，使用非透明像素进行颜色聚类
                pH_value = get_ph_value_from_patch(non_transparent_pixels)
            else:
                pH_value = "未知"
                print(
                    f"    对象 {i}: 类别 {int(cls)}, 置信度 {conf:.2f}, 掩码已保存, pH 值: {pH_value} (无有效掩码区域)"
                )
            print(
                f"    对象 {i}: 类别 {int(cls)}, 置信度 {conf:.2f}, 掩码已保存, pH 值: {pH_value}"
            )

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
