"""
模块4: 利用被还原的变色部分的颜色(HSV), 使用神经网络模型预测pH值。
"""

import torch
import torch.nn as nn
import numpy as np
from pHmap import bgr_to_hsv
from ph_model import predict_ph


def calculate_ph_value(colored_color, uncolored_color=None, color_space="hsv"):
    """
    使用训练好的神经网络模型，根据变色部分和未变色部分颜色预测pH值。
    如果未提供未变色颜色，则使用默认固定未变色颜色 (42.0, 0.71, 0.82)。

    Args:
        colored_color (tuple): 变色部分颜色 (H, S, V) 或 (B, G, R)
        uncolored_color (tuple, optional): 未变色部分颜色 (H, S, V) 或 (B, G, R)
            如果为None，则使用固定未变色颜色 (42.0, 0.71, 0.82)。
        color_space (str): 输入颜色空间，'hsv' 或 'bgr'，默认为 'hsv'

    Returns:
        float: 预测的pH值
    """
    from ph_model import predict_ph_two_colors
    from pHmap import UNCOLORED_HSV
    
    if uncolored_color is None:
        uncolored_color = UNCOLORED_HSV
        # 如果未变色颜色是HSV格式，而colored_color可能是BGR，需要统一颜色空间
        # 这里假设uncolored_color已经是HSV，colored_color根据color_space参数转换
        # predict_ph_two_colors 会处理转换
    
    # 使用双颜色模型预测
    predicted_ph = predict_ph_two_colors(colored_color, uncolored_color, color_space=color_space)
    return predicted_ph
