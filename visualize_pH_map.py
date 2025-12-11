import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pHmap import pH_color_map

# 提取数据
colors = np.array(list(pH_color_map.keys()))  # (B, G, R) 值
ph_values = np.array(list(pH_color_map.values()))

# 对pH值进行排序，以便按顺序连接点
sorted_indices = np.argsort(ph_values)
colors_sorted = colors[sorted_indices]
ph_values_sorted = ph_values[sorted_indices]

# 创建3D图形
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection="3d")

# 绘制散点图，颜色由其自身的BGR值映射为RGB用于显示
scatter_colors = [
    (r / 255.0, g / 255.0, b / 255.0) for b, g, r in colors_sorted
]  # 转换为RGB并归一化
# 将numpy数组转换为列表以避免类型问题
x = colors_sorted[:, 0].tolist()
y = colors_sorted[:, 1].tolist()
z = colors_sorted[:, 2].tolist()
sc = ax.scatter(x, y, z, c=scatter_colors, s=100, depthshade=False)

# 连接相邻的点形成一条线
ax.plot(
    colors_sorted[:, 0],
    colors_sorted[:, 1],
    colors_sorted[:, 2],
    color="gray",
    alpha=0.5,
    linestyle="-",
    linewidth=2,
)

# 设置标签和标题
ax.set_xlabel("Blue Value")
ax.set_ylabel("Green Value")
ax.set_zlabel("Red Value")
ax.set_title("3D Visualization of pH Color Map with Connected Points")

# 添加图例说明
for i, (color, ph) in enumerate(zip(colors_sorted, ph_values_sorted)):
    ax.text(color[0], color[1], color[2], f" {ph}", fontsize=8)

# 显示图形
plt.tight_layout()
plt.show()
