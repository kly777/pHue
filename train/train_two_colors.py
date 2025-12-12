import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'src')
from src.ph_model import train_model_two_colors

print("开始训练双颜色模型...")
model, stats = train_model_two_colors()
print("训练完成。")
print("模型保存为 data/models/best_ph_model_two_colors.pth")