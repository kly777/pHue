"""
模块2 & 3: 从mask图像中聚类, 获取两个颜色值, 并区分哪个是变色部分, 哪个是未变色部分。
"""

import numpy as np
import cv2


def extract_colors_from_patch(image):
    """
    从pH试纸的掩码区域图像中提取出变色和未变色两种主要颜色。

    Args:
        image: 包含透明背景的BGR图像 (np.ndarray with shape (H, W, 4))

    Returns:
        tuple: (变色部分颜色, 未变色部分颜色) 的BGR元组，如果失败则返回(None, None)
    """
    # 确保输入图像是正确的格式
    if not isinstance(image, np.ndarray) or image.shape[2] != 4:
        print("输入图像必须是包含Alpha通道的numpy数组")
        return None, None

    # 提取非透明像素 (忽略透明背景)
    alpha_channel = image[:, :, 3]
    non_transparent_pixels = image[alpha_channel > 0, :3].astype(np.float32)

    if len(non_transparent_pixels) == 0:
        print("没有找到非透明像素")
        return None, None

    # 使用k-means聚类将像素分为2类（变色部分和未变色部分）
    n_colors = 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    labels = np.zeros((non_transparent_pixels.shape[0],), dtype=np.int32)
    _, labels, palette = cv2.kmeans(
        non_transparent_pixels, n_colors, labels, criteria, 10, flags
    )

    # 计算每个聚类的像素数量
    unique_labels, counts = np.unique(labels, return_counts=True)

    # 找到最大的两个聚类（按像素数量排序）
    sorted_indices = np.argsort(counts)[::-1]
    largest_cluster_idx = sorted_indices[0]
    second_largest_cluster_idx = (
        sorted_indices[1] if len(sorted_indices) > 1 else largest_cluster_idx
    )

    # 获取两个主要颜色
    cluster1_color = palette[largest_cluster_idx]
    cluster2_color = palette[second_largest_cluster_idx]

    # --- 区分变色和未变色部分 ---
    # 方法：计算pH_color_map中所有“未变色”部分的颜色的平均值，作为动态的理想参考。
    from pHmap import pH_color_map

    # 提取pH_color_map中所有“未变色”部分的颜色并计算其平均值
    reference_colors = np.array(
        [color_ref for color_change, color_ref in pH_color_map.keys()]
    )
    ideal_reference_bgr = np.mean(reference_colors, axis=0).astype(np.float32)

    # 计算两个聚类颜色与动态理想参考色的距离
    diff1 = np.linalg.norm(cluster1_color - ideal_reference_bgr)
    diff2 = np.linalg.norm(cluster2_color - ideal_reference_bgr)

    # 距离更近的聚类被认为是“未变色”的参考区域
    if diff1 < diff2:
        uncolored_color = cluster1_color
        colored_color = cluster2_color
    else:
        uncolored_color = cluster2_color
        colored_color = cluster1_color

    print(f"识别结果:")
    print(f"  变色部分颜色: {colored_color}")
    print(f"  未变色部分颜色: {uncolored_color}")

    return tuple(colored_color.astype(int)), tuple(uncolored_color.astype(int))
