import cv2
import numpy as np
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'src')
from src.segmentation import load_model, segment_image
from src.color_analysis import extract_colors_from_patch
from src.ph_measurement import calculate_ph_value

model = load_model("seg/weights/best12072154.pt")
img = cv2.imread("seg/OIP-2867035697.jpg")
if img is None:
    print("无法加载图像")
    sys.exit(1)

for obj in segment_image(model, img, conf_threshold=0.5):
    cropped_bgra = obj["cropped_bgra"]
    colored_hsv, uncolored_hsv = extract_colors_from_patch(cropped_bgra)
    print("colored_hsv:", colored_hsv)
    print("uncolored_hsv:", uncolored_hsv)
    if colored_hsv is not None and uncolored_hsv is not None:
        ph = calculate_ph_value(colored_hsv, uncolored_hsv, color_space="hsv")
        print("预测 pH:", ph)
    else:
        print("无法提取颜色")
    break  # 只处理第一个对象