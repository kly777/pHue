"""
模块2 & 3: 从mask图像中聚类, 获取两个HSV颜色值, 并区分哪个是变色部分, 哪个是未变色部分。
"""

import numpy as np
import cv2

# 预计算参考HSV颜色（固定理想参考色）
_REFERENCE_BGR = np.array([105.971, 184.178, 214.027], dtype=np.float32)
_REFERENCE_HSV = cv2.cvtColor(
    np.array([[_REFERENCE_BGR]], dtype=np.float32) / 255.0, cv2.COLOR_BGR2HSV
)[0][0]
_REFERENCE_HSV[0] *= 2  # Hue从[0,180]转换到[0,360]


def extract_colors_from_patch(image):
    """
    从pH试纸的掩码区域图像中提取出变色和未变色两种主要颜色。

    Args:
        image: 包含透明背景的BGR图像 (np.ndarray with shape (H, W, 4))

    Returns:
        tuple: (变色部分颜色, 未变色部分颜色) 的HSV元组，如果失败则返回(None, None)
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

    # 下采样：如果像素过多，随机采样最多1000个像素
    max_pixels = 1000
    if len(non_transparent_pixels) > max_pixels:
        indices = np.random.choice(len(non_transparent_pixels), max_pixels, replace=False)
        sample_pixels = non_transparent_pixels[indices]
    else:
        sample_pixels = non_transparent_pixels

    # 使用k-means聚类将像素分为2类（变色部分和未变色部分）
    n_colors = 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)  # 减少迭代，增加epsilon
    flags = cv2.KMEANS_RANDOM_CENTERS
    labels = np.zeros((sample_pixels.shape[0],), dtype=np.int32)
    _, labels, palette = cv2.kmeans(
        sample_pixels, n_colors, labels, criteria, 5, flags  # 减少尝试次数
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
    # 使用预计算的参考HSV
    reference_hsv = _REFERENCE_HSV

    # 将聚类颜色转换为HSV
    cluster1_hsv = cv2.cvtColor(
        np.array([[cluster1_color]], dtype=np.float32) / 255.0, cv2.COLOR_BGR2HSV
    )[0][0]
    cluster1_hsv[0] *= 2  # Hue从[0,180]转换到[0,360]
    cluster2_hsv = cv2.cvtColor(
        np.array([[cluster2_color]], dtype=np.float32) / 255.0, cv2.COLOR_BGR2HSV
    )[0][0]
    cluster2_hsv[0] *= 2  # Hue从[0,180]转换到[0,360]

    # 计算两个聚类颜色与固定理想参考色的HSV距离
    diff1 = np.linalg.norm(cluster1_hsv - reference_hsv)
    diff2 = np.linalg.norm(cluster2_hsv - reference_hsv)

    # 距离更近的聚类被认为是“未变色”的参考区域
    if diff1 < diff2:
        uncolored_color = cluster1_color
        colored_color = cluster2_color
    else:
        uncolored_color = cluster2_color
        colored_color = cluster1_color

    # 将BGR颜色转换为HSV格式
    colored_bgr = np.array([[colored_color]], dtype=np.float32) / 255.0
    uncolored_bgr = np.array([[uncolored_color]], dtype=np.float32) / 255.0

    colored_hsv = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2HSV)[0][0]
    uncolored_hsv = cv2.cvtColor(uncolored_bgr, cv2.COLOR_BGR2HSV)[0][0]

    # 四舍五入到3位小数
    colored_hsv = tuple(round(float(x), 3) for x in colored_hsv)
    uncolored_hsv = tuple(round(float(x), 3) for x in uncolored_hsv)

    return colored_hsv, uncolored_hsv
