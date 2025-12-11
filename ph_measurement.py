"""
模块4: 利用被还原的变色部分的颜色, 使用神经网络模型预测pH值。
"""

import torch
import torch.nn as nn
import numpy as np
from pHmap import pH_color_map

# 定义神经网络模型结构
class pHNet(nn.Module):
    """
    神经网络模型，用于拟合颜色(BGR)到pH值的映射关系
    输入: B, G, R 三个通道的颜色值
    输出: 对应的pH值
    """
    def __init__(self, input_size=3, hidden_sizes=[64, 32, 16], output_size=1):
        super(pHNet, self).__init__()
        
        # 创建网络层
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # 添加dropout防止过拟合
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def load_model(model_path="best_ph_model.pth"):
    """加载训练好的模型"""
    # 由于PyTorch 2.6+默认启用weights_only=True，需要设置为False来兼容旧的保存格式
    checkpoint = torch.load(model_path, weights_only=False)
    model = pHNet(
        input_size=checkpoint["config"]["input_size"],
        hidden_sizes=checkpoint["config"]["hidden_sizes"],
        output_size=checkpoint["config"]["output_size"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return model, checkpoint["stats"]

def predict_ph(color, model=None, stats=None):
    """
    使用训练好的模型预测pH值

    Args:
        color (tuple): BGR颜色值 (B, G, R)
        model: 训练好的模型
        stats: 数据统计信息 (均值和标准差)

    Returns:
        float: 预测的pH值
    """
    if model is None or stats is None:
        model, stats = load_model()
    
    model.eval()
    
    # 解包统计信息
    X_mean, X_std, y_mean, y_std = stats
    
    # 转换输入数据
    color_array = np.array(color, dtype=np.float32).reshape(1, -1)
    color_normalized = (color_array - X_mean) / X_std
    
    # 转换为tensor
    color_tensor = torch.from_numpy(color_normalized)
    
    # 预测
    with torch.no_grad():
        pred_normalized = model(color_tensor)
        pred = pred_normalized.item() * y_std + y_mean
    
    return pred

def calculate_ph_value(corrected_colored_color):
    """
    使用训练好的神经网络模型，根据校正后的变色部分颜色预测pH值。

    Args:
        corrected_colored_color (tuple): 校正后的变色部分颜色 (B, G, R)

    Returns:
        float: 预测的pH值
    """
    # 直接使用神经网络模型进行预测
    predicted_ph = predict_ph(corrected_colored_color)
    
    return predicted_ph
