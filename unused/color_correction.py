"""
模块3: 利用未变色部分的固有色(HSV), 还原变色部分的颜色(HSV)。
"""

import numpy as np
import cv2


def correct_color_by_reference(colored_color, uncolored_color):
    """
    使用未变色部分的实际颜色来校正环境光，还原变色部分的真实颜色。

    Args:
        colored_color (tuple): 识别出的变色部分颜色 (H, S, V)
        uncolored_color (tuple): 识别出的未变色部分颜色 (H, S, V)

    Returns:
        tuple: 校正后的变色部分颜色 (H, S, V)
    """
    # 转换为numpy数组
    colored = np.array(colored_color, dtype=np.float32)
    uncolored = np.array(uncolored_color, dtype=np.float32)

    # --- 环境光校正 ---
    # 将固定的理想参考色转换为HSV
    reference_bgr = np.array([105.971, 184.178, 214.027], dtype=np.float32)
    reference_hsv = cv2.cvtColor(
        np.array([[reference_bgr]], dtype=np.float32) / 255.0, cv2.COLOR_BGR2HSV
    )[0][0]
    reference_hsv[0] *= 2  # Hue从[0,180]转换到[0,360]

    # 在HSV空间中计算增益因子，但仅对S和V通道进行校正
    epsilon = 1e-8
    gain_s = (reference_hsv[1] + epsilon) / (uncolored[1] + epsilon)  # 饱和度(S)增益
    gain_v = (reference_hsv[2] + epsilon) / (uncolored[2] + epsilon)  # 明度(V)增益

    # 应用增益校正到变色部分的S和V通道，Hue保持不变
    corrected_colored = np.array(
        [
            colored[0],  # H通道保持不变
            np.clip(colored[1] * gain_s, 0, 1),  # S通道校正
            np.clip(colored[2] * gain_v, 0, 1),  # V通道校正
        ]
    )

    # 四舍五入到3位小数
    corrected_colored = tuple(round(float(x), 3) for x in corrected_colored)

    # print(f"校正结果:")
    # print(f"  原始变色颜色(HSV): {colored.tolist()}")
    # print(f"  实际未变色颜色(HSV): {uncolored.tolist()}")
    # print(f"  理想参考颜色(HSV): {reference_hsv.tolist()}")
    # print(f"  校正后变色颜色(HSV): {corrected_colored}")

    return corrected_colored
