"""
模块4: 利用被还原的变色部分的颜色(HSV), 使用神经网络模型预测pH值。
"""

import torch
import torch.nn as nn
import numpy as np
from pHmap import bgr_to_hsv
from ph_model import predict_ph


def calculate_ph_value(corrected_colored_color):
    """
    使用训练好的神经网络模型，根据校正后的变色部分颜色预测pH值。

    Args:
        corrected_colored_color (tuple): 校正后的变色部分颜色 (B, G, R)

    Returns:
        float: 预测的pH值
    """

    # 使用神经网络模型进行预测，直接传入BGR颜色和color_space参数
    predicted_ph = predict_ph(corrected_colored_color)

    return predicted_ph
