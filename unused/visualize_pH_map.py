import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2

from pHmap import pH_color_map

# 提取数据
colors_hsv = np.array(list(pH_color_map.keys()))  # (H, S, V) 值
ph_values = np.array(list(pH_color_map.values()))


# 对pH值进行排序，以便按顺序连接点
sorted_indices = np.argsort(ph_values)
colors_sorted = colors_hsv[sorted_indices]
ph_values_sorted = ph_values[sorted_indices]

# 创建3D图形
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection="3d")

# 将HSV坐标转换为列表以避免类型问题
x = colors_sorted[:, 0].tolist()
y = colors_sorted[:, 1].tolist()
z = colors_sorted[:, 2].tolist()

# 使用HSV值作为颜色，按pH值映射到颜色范围
colors_normalized = (ph_values_sorted - ph_values_sorted.min()) / (
    ph_values_sorted.max() - ph_values_sorted.min()
)
cmap = plt.get_cmap('viridis')  # 使用viridis色图
scatter_colors = cmap(colors_normalized)

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
ax.set_xlabel("Hue (H)")
ax.set_ylabel("Saturation (S)")
ax.set_zlabel("Value (V)")
ax.set_title("3D Visualization of pH HSV Color Map with Connected Points")

# 添加图例说明
for i, (color, ph) in enumerate(zip(colors_sorted, ph_values_sorted)):
    ax.text(color[0], color[1], color[2], f" {ph}", fontsize=8)

# 显示图形
plt.tight_layout()
plt.show()
