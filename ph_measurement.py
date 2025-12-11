"""
模块4: 利用被还原的变色部分的颜色, 匹配最佳的样板,得到pH值。
"""

import numpy as np


def calculate_ph_value(corrected_colored_color):
    """
    使用校正后的变色部分颜色，在pH_color_map中进行匹配，计算最终的pH值。

    Args:
        corrected_colored_color (tuple): 校正后的变色部分颜色 (B, G, R)

    Returns:
        float or str: 计算出的pH值，或"未知"
    """
    from pHmap import pH_color_map

    # 将输入转换为numpy数组
    sample_color = np.array(corrected_colored_color, dtype=np.float32)

    # --- 计算pH值 ---
    # 使用校正后的样本颜色与pH_color_map进行匹配
    color_distances = []
    for color_change, pH in pH_color_map.items():
        # 注意：我们现在只关心变色部分的颜色
        distance = np.linalg.norm(sample_color - color_change)
        color_distances.append((distance, pH))

    # 按距离排序并取最近的两个
    color_distances.sort()
    closest_two = color_distances[:2]

    # 计算加权平均 pH 值 (权重为距离的倒数)
    if len(closest_two) == 2:
        dist1, pH1 = closest_two[0]
        dist2, pH2 = closest_two[1]
        epsilon = 1e-8
        weight1 = 1.0 / (dist1 + epsilon)
        weight2 = 1.0 / (dist2 + epsilon)
        pH_value = (weight1 * pH1 + weight2 * pH2) / (weight1 + weight2)
    elif len(closest_two) == 1:
        pH_value = closest_two[0][1]
    else:
        pH_value = "未知"

    return pH_value
