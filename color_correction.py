"""
模块3: 利用未变色部分的固有色, 还原变色部分的颜色。
"""

import numpy as np


def correct_color_by_reference(colored_color, uncolored_color):
    """
    使用未变色部分的实际颜色来校正环境光，还原变色部分的真实颜色。

    Args:
        colored_color (tuple): 识别出的变色部分颜色 (B, G, R)
        uncolored_color (tuple): 识别出的未变色部分颜色 (B, G, R)

    Returns:
        tuple: 校正后的变色部分颜色 (B, G, R)
    """
    # 转换为numpy数组
    colored = np.array(colored_color, dtype=np.float32)
    uncolored = np.array(uncolored_color, dtype=np.float32)

    # --- 环境光校正 ---
    # 方法：计算pH_color_map中所有“未变色”部分的颜色的平均值，作为理想参考。
    from pHmap import pH_color_map

    # 提取pH_color_map中所有“未变色”部分的颜色并计算其平均值
    reference_colors = np.array(
        [color_ref for color_change, color_ref in pH_color_map.keys()]
    )
    ideal_reference_bgr = np.mean(reference_colors, axis=0).astype(np.float32)

    # 计算增益因子 (Gain Factor) 进行校正
    epsilon = 1e-8
    gain_factor = (ideal_reference_bgr + epsilon) / (uncolored + epsilon)

    # 对样本颜色应用增益校正
    corrected_colored = np.clip(colored * gain_factor, 0, 255).astype(np.float32)

    print(f"校正结果:")
    print(f"  原始变色颜色: {colored}")
    print(f"  实际未变色颜色: {uncolored}")
    print(f"  理想参考颜色: {ideal_reference_bgr}")
    print(f"  校正后变色颜色: {corrected_colored}")

    return tuple(corrected_colored.astype(int))
