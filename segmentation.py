"""
模块1: 利用YOLO模型分割出mask部分图像
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics.models import YOLO


def load_model(model_path):
    """加载 YOLO 模型"""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)
    return model


def segment_image(model, image_path, conf_threshold=0.5):
    """
    对单张图像进行分割，返回所有检测到的掩码区域及其信息。

    Args:
        model: 加载的YOLO模型
        image_path: 图像路径
        conf_threshold: 置信度阈值

    Returns:
        一个生成器，每次产生一个字典，包含以下键：
            'image': 原始图像 (np.ndarray)
            'cropped_bgra': 裁剪并带有透明背景的掩码区域 (np.ndarray)
            'box': 边界框坐标 [x1, y1, x2, y2]
            'cls': 类别
            'conf': 置信度
            'stem': 文件名基础
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"无法读取图像: {image_path}")
        return

    # 推理
    results = model(img, conf=conf_threshold, verbose=False)
    if results[0].masks is None:
        print(f"未检测到分割对象: {image_path}")
        return

    # 获取原始图像尺寸
    orig_h, orig_w = img.shape[:2]
    masks = results[0].masks.data.cpu().numpy()  # (N, H, W) 模型输出尺寸
    boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []
    classes = results[0].boxes.cls.cpu().numpy() if results[0].boxes else []
    confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes else []

    stem = Path(image_path).stem

    for mask, box, cls, conf in zip(masks, boxes, classes, confs):
        # 将掩码调整到原始图像尺寸
        mask_resized = cv2.resize(
            mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )

        # 提取非零区域的边界框以进行精确裁剪
        coords = cv2.findNonZero((mask_resized > 0.5).astype(np.uint8))
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)

            # 裁剪原始图像和掩码到边界框区域
            cropped_img = img[y : y + h, x : x + w]
            cropped_mask = mask_resized[y : y + h, x : x + w]

            # 创建一个与裁剪后图像大小相同的BGRA图像
            cropped_bgra = np.zeros((h, w, 4), dtype=np.uint8)
            cropped_bgra[:, :, :3] = cropped_img  # 复制裁剪后的BGR图像
            cropped_bgra[:, :, 3] = (cropped_mask > 0.5) * 255  # 设置Alpha通道

            yield {
                "image": img,
                "cropped_bgra": cropped_bgra,
                "box": box,
                "cls": cls,
                "conf": conf,
                "stem": stem,
            }
