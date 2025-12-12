import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.pHmap import pH_color_map_two_colors


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
    return (
        np.sin(h1_rad),
        np.cos(h1_rad),
        s1,
        v1,
        np.sin(h2_rad),
        np.cos(h2_rad),
        s2,
        v2,
    )


def prepare_data_two_colors():
    """准备双颜色训练数据"""
    colors = list(
        pH_color_map_two_colors.keys()
    )  # 六元组 (c_h, c_s, c_v, u_h, u_s, u_v)
    ph_values = list(pH_color_map_two_colors.values())

    # 预处理所有颜色对
    X = np.array(
        [
            preprocess_two_colors((c_h, c_s, c_v), (u_h, u_s, u_v))
            for (c_h, c_s, c_v, u_h, u_s, u_v) in colors
        ],
        dtype=np.float32,
    )
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
                    "config": {
                        "input_size": 8,
                        "hidden_sizes": [64, 32, 16],
                        "output_size": 1,
                    },
                },
                "data/models/best_ph_model_two_colors.pth",
            )
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"早停触发，最佳损失: {best_loss:.6f}")
            break

    return model, stats


def load_model_two_colors(model_path="data/models/best_ph_model_two_colors.pth"):
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


def predict_ph_two_colors(
    colored_color, uncolored_color, color_space="hsv", model=None, stats=None
):
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
    model, stats = train_model_two_colors()

    # 测试模型
    print("\n测试模型预测效果:")
    # 使用双颜色测试
    test_colored = [
        (355.218, 0.628, 0.569),
        (355.172, 0.551, 0.516),
        (50.443, 0.785, 0.648),
    ]
    test_uncolored = (42.0, 0.71, 0.82)  # 固定未变色颜色

    for i, colored in enumerate(test_colored):
        predicted_ph = predict_ph_two_colors(
            colored, test_uncolored, color_space="hsv", model=model, stats=stats
        )
        print(f"变色颜色 {colored} -> 预测pH值: {predicted_ph:.2f}")
