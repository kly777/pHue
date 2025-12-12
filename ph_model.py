import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pHmap import pH_color_map


class pHNet(nn.Module):
    """
    神经网络模型，用于拟合颜色(HSV)到pH值的映射关系
    输入: H, S, V 三个通道的颜色值 (经过sin/cos转换处理)
    输出: 对应的pH值
    """

    def __init__(self, input_size=4, hidden_sizes=[64, 32, 16], output_size=1):
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


def preprocess_hsv(hsv_color):
    """
    处理HSV颜色，特别是Hue通道的循环特性
    将Hue转换为sin/cos分量，消除循环不连续性
    输入: HSV颜色元组 (H, S, V)
    输出: 处理后的4维特征向量 [h_sin, h_cos, S, V]
    """
    h, s, v = hsv_color

    # 将Hue(0-360)转换为sin/cos分量 [0,360) → [-1,1]
    h_rad = np.radians(h)
    h_sin = np.sin(h_rad)
    h_cos = np.cos(h_rad)

    # 返回4维特征: [h_sin, h_cos, S, V]
    return (h_sin, h_cos, s, v)


def prepare_data():
    """准备训练数据（适配HSV空间）"""
    colors = list(pH_color_map.keys())  # 现在是HSV格式 (H, S, V)
    ph_values = list(pH_color_map.values())

    # 预处理所有颜色 - 转换Hue为sin/cos
    X = np.array([preprocess_hsv(color) for color in colors], dtype=np.float32)
    y = np.array(ph_values, dtype=np.float32).reshape(-1, 1)

    # 数据归一化（S/V已[0,1]，但sin/cos需特殊处理）
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)

    # 避免除零（sin/cos标准差可能很小）
    X_std[X_std < 1e-8] = 1.0

    X_normalized = (X - X_mean) / X_std
    y_mean = y.mean()
    y_std = y.std()
    y_normalized = (y - y_mean) / y_std

    return (X_normalized, y_normalized), (X_mean, X_std, y_mean, y_std)


def train_model():
    """训练神经网络模型"""
    # 准备数据
    (X_train, y_train), stats = prepare_data()

    # 转换为torch张量
    X_tensor = torch.from_numpy(X_train)
    y_tensor = torch.from_numpy(y_train)

    # 初始化模型
    model = pHNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # 训练参数
    epochs = 10000
    best_loss = float("inf")
    patience = 500
    patience_counter = 0

    print("开始训练神经网络模型...")

    for epoch in range(epochs):
        # 前向传播
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印进度
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")

        # 早停机制
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            # 保存最佳模型
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "stats": stats,
                    "config": {
                        "input_size": 3,
                        "hidden_sizes": [64, 32, 16],
                        "output_size": 1,
                    },
                },
                "best_ph_model.pth",
            )
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"早停触发，最佳损失: {best_loss:.6f}")
            break

    return model, stats


def load_model(model_path="best_ph_model.pth"):
    """加载训练好的模型"""
    # 由于PyTorch 2.6+默认启用weights_only=True，需要设置为False来兼容旧的保存格式
    checkpoint = torch.load(model_path, weights_only=False)
    model = pHNet(
        input_size=4,
        hidden_sizes=[64, 32, 16],
        output_size=1,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, checkpoint["stats"]


# 全局缓存模型和统计信息
_cached_model = None
_cached_stats = None

def get_cached_model():
    """获取缓存的模型和统计信息，如果未缓存则加载"""
    global _cached_model, _cached_stats
    if _cached_model is None or _cached_stats is None:
        _cached_model, _cached_stats = load_model()
    return _cached_model, _cached_stats

def predict_ph(color, color_space="hsv", model=None, stats=None):
    """
    使用训练好的模型预测pH值

    Args:
        color (tuple): 颜色值 - HSV格式(H,S,V)或BGR格式(B,G,R)
        color_space: 指定输入颜色空间 ('hsv' 或 'bgr')，默认为'hsv'
        model: 训练好的模型实例，如果为None则加载默认模型
        stats: 数据归一化统计信息，如果model为None则从检查点加载

    Returns:
        float: 预测的pH值
    """
    if model is None or stats is None:
        model, stats = get_cached_model()

    model.eval()
    X_mean, X_std, y_mean, y_std = stats

    # 修复BGR到HSV的转换问题
    if color_space.lower() == "bgr":
        # 创建正确的(1, 1, 3)形状数组（单像素BGR图像）
        bgr_normalized = np.array([[color]], dtype=np.float32) / 255.0
        hsv = cv2.cvtColor(bgr_normalized, cv2.COLOR_BGR2HSV)[0][0]

        # 调整Hue到[0,360]范围
        h = float(hsv[0] * 2)
        s = float(hsv[1])
        v = float(hsv[2])
        color = (h, s, v)

    # 预处理HSV颜色（处理Hue循环特性）
    processed_color = preprocess_hsv(color)

    # 归一化
    color_array = np.array(processed_color, dtype=np.float32).reshape(1, -1)
    color_normalized = (color_array - X_mean) / X_std

    # 预测
    color_tensor = torch.from_numpy(color_normalized)
    with torch.no_grad():
        pred_normalized = model(color_tensor)
        pred = pred_normalized.item() * y_std + y_mean

    return pred


# ============================================================================
# 双颜色神经网络模型（合并模块三和模块四）
# ============================================================================

class pHNetTwoColors(nn.Module):
    """
    神经网络模型，用于拟合两个颜色(HSV)到pH值的映射关系
    输入: 变色颜色 (H, S, V) + 未变色颜色 (H, S, V) -> 8个特征（每个颜色经过sin/cos转换）
    输出: 对应的pH值
    """
    def __init__(self, input_size=8, hidden_sizes=[64, 32, 16], output_size=1):
        super(pHNetTwoColors, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def preprocess_two_colors(colored_hsv, uncolored_hsv):
    """
    处理两个HSV颜色，将每个颜色的Hue转换为sin/cos分量。
    输入: 变色颜色 (H, S, V), 未变色颜色 (H, S, V)
    输出: 8维特征向量 [colored_h_sin, colored_h_cos, colored_s, colored_v,
                     uncolored_h_sin, uncolored_h_cos, uncolored_s, uncolored_v]
    """
    h1, s1, v1 = colored_hsv
    h2, s2, v2 = uncolored_hsv
    h1_rad = np.radians(h1)
    h2_rad = np.radians(h2)
    return (np.sin(h1_rad), np.cos(h1_rad), s1, v1,
            np.sin(h2_rad), np.cos(h2_rad), s2, v2)


def prepare_data_two_colors():
    """准备双颜色训练数据"""
    from pHmap import pH_color_map_two_colors
    colors = list(pH_color_map_two_colors.keys())  # 六元组 (c_h, c_s, c_v, u_h, u_s, u_v)
    ph_values = list(pH_color_map_two_colors.values())
    
    # 预处理所有颜色对
    X = np.array([preprocess_two_colors((c_h, c_s, c_v), (u_h, u_s, u_v))
                  for (c_h, c_s, c_v, u_h, u_s, u_v) in colors], dtype=np.float32)
    y = np.array(ph_values, dtype=np.float32).reshape(-1, 1)
    
    # 数据归一化
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    # 对于标准差极小的特征（如未变色颜色固定），避免除以接近零的值
    X_std[X_std < 1e-6] = 1.0
    X_normalized = (X - X_mean) / X_std
    y_mean = y.mean()
    y_std = y.std()
    y_normalized = (y - y_mean) / y_std
    
    return (X_normalized, y_normalized), (X_mean, X_std, y_mean, y_std)


def train_model_two_colors():
    """训练双颜色神经网络模型"""
    (X_train, y_train), stats = prepare_data_two_colors()
    X_tensor = torch.from_numpy(X_train)
    y_tensor = torch.from_numpy(y_train)
    
    model = pHNetTwoColors()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    epochs = 10000
    best_loss = float("inf")
    patience = 500
    patience_counter = 0
    
    print("开始训练双颜色神经网络模型...")
    for epoch in range(epochs):
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "stats": stats,
                    "config": {"input_size": 8, "hidden_sizes": [64, 32, 16], "output_size": 1},
                },
                "best_ph_model_two_colors.pth",
            )
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"早停触发，最佳损失: {best_loss:.6f}")
            break
    
    return model, stats


def load_model_two_colors(model_path="best_ph_model_two_colors.pth"):
    """加载训练好的双颜色模型"""
    checkpoint = torch.load(model_path, weights_only=False)
    model = pHNetTwoColors(
        input_size=8,
        hidden_sizes=[64, 32, 16],
        output_size=1,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint["stats"]


# 全局缓存双颜色模型
_cached_model_two_colors = None
_cached_stats_two_colors = None

def get_cached_model_two_colors():
    """获取缓存的双颜色模型和统计信息"""
    global _cached_model_two_colors, _cached_stats_two_colors
    if _cached_model_two_colors is None or _cached_stats_two_colors is None:
        _cached_model_two_colors, _cached_stats_two_colors = load_model_two_colors()
    return _cached_model_two_colors, _cached_stats_two_colors


def predict_ph_two_colors(colored_color, uncolored_color, color_space="hsv", model=None, stats=None):
    """
    使用训练好的双颜色模型预测pH值
    
    Args:
        colored_color (tuple): 变色颜色 (H, S, V) 或 (B, G, R)
        uncolored_color (tuple): 未变色颜色 (H, S, V) 或 (B, G, R)
        color_space: 指定输入颜色空间 ('hsv' 或 'bgr')，默认为'hsv'
        model: 训练好的模型实例，如果为None则加载默认模型
        stats: 数据归一化统计信息，如果model为None则从检查点加载
    
    Returns:
        float: 预测的pH值
    """
    if model is None or stats is None:
        model, stats = get_cached_model_two_colors()
    
    model.eval()
    X_mean, X_std, y_mean, y_std = stats
    
    # 如果输入是BGR，转换为HSV
    if color_space.lower() == "bgr":
        colored_color = _bgr_to_hsv(colored_color)
        uncolored_color = _bgr_to_hsv(uncolored_color)
    
    # 预处理两个颜色
    processed = preprocess_two_colors(colored_color, uncolored_color)
    color_array = np.array(processed, dtype=np.float32).reshape(1, -1)
    color_normalized = (color_array - X_mean) / X_std
    
    # 预测
    color_tensor = torch.from_numpy(color_normalized)
    with torch.no_grad():
        pred_normalized = model(color_tensor)
        pred = pred_normalized.item() * y_std + y_mean
    
    return pred


def _bgr_to_hsv(bgr_tuple):
    """将BGR颜色元组转换为HSV颜色元组（内部使用）"""
    bgr_normalized = np.array([[bgr_tuple]], dtype=np.float32) / 255.0
    hsv = cv2.cvtColor(bgr_normalized, cv2.COLOR_BGR2HSV)[0][0]
    h = float(hsv[0] * 2)
    s = float(hsv[1])
    v = float(hsv[2])
    return (h, s, v)


if __name__ == "__main__":
    # 训练模型
    model, stats = train_model()

    # 测试模型
    print("\n测试模型预测效果:")
    # 修改1: 使用HSV格式的测试颜色（原BGR已转换）
    test_colors = [
        (229.350, 0.550, 0.665),  # 对应原BGR (76.225, 85.511, 169.538)
        (198.420, 0.410, 0.475),  # 对应原BGR (69.042, 112.688, 121.125)
        (0.000, 0.000, 0.392),  # 对应原BGR (100, 100, 100)
    ]

    # 修改2: 添加颜色空间说明和BGR兼容测试
    for i, color in enumerate(test_colors):
        # HSV格式测试
        predicted_ph = predict_ph(color, color_space="hsv", model=model, stats=stats)
        print(f"HSV颜色 {color} -> 预测pH值: {predicted_ph:.2f}")
