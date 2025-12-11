import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pHmap import pH_color_map


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


def prepare_data():
    """准备训练数据"""
    # 从pH_color_map提取数据
    colors = list(pH_color_map.keys())  # (B, G, R)
    ph_values = list(pH_color_map.values())

    # 转换为numpy数组
    X = np.array(colors, dtype=np.float32)
    y = np.array(ph_values, dtype=np.float32).reshape(-1, 1)

    # 数据归一化
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    y_mean = y.mean()
    y_std = y.std()

    X_normalized = (X - X_mean) / X_std
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
    epochs = 1000
    best_loss = float("inf")
    patience = 50
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


if __name__ == "__main__":
    # 训练模型
    model, stats = train_model()

    # 测试模型
    print("\n测试模型预测效果:")
    test_colors = [
        (76.225, 85.511, 169.538),  # 已知点
        (69.042, 112.688, 121.125),  # 已知点
        (100, 100, 100),  # 新颜色
    ]

    for color in test_colors:
        predicted_ph = predict_ph(color, model, stats)
        print(f"颜色 {color} -> 预测pH值: {predicted_ph:.2f}")
