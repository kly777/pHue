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


def segment_image(model, image_input, conf_threshold=0.5):
    """
    对单张图像进行分割，返回所有检测到的掩码区域及其信息。

    Args:
        model: 加载的YOLO模型
        image_input: 图像路径 (str/Path) 或 numpy 数组 (BGR格式)
        conf_threshold: 置信度阈值

    Returns:
        一个生成器，每次产生一个字典，包含以下键：
            'image': 原始图像 (np.ndarray)
            'cropped_bgra': 裁剪并带有透明背景的掩码区域 (np.ndarray)
            'box': 边界框坐标 [x1, y1, x2, y2]
            'cls': 类别
            'conf': 置信度
            'stem': 文件名基础 (如果是文件路径) 或空字符串
    """
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            print(f"无法读取图像: {image_input}")
            return
        stem = Path(image_input).stem
    elif isinstance(image_input, np.ndarray):
        img = image_input
        stem = ""
    else:
        raise TypeError("image_input 必须是路径字符串、Path对象或numpy数组")

    if img.size == 0:
        print("图像为空")
        return

    # 推理
    results = model(img, conf=conf_threshold, verbose=False)
    if results[0].masks is None:
        print(
            f"未检测到分割对象: {image_input if isinstance(image_input, (str, Path)) else 'numpy array'}"
        )
        return

    # 获取原始图像尺寸
    orig_h, orig_w = img.shape[:2]
    masks = results[0].masks.data.cpu().numpy()  # (N, H, W) 模型输出尺寸
    boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []
    classes = results[0].boxes.cls.cpu().numpy() if results[0].boxes else []
    confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes else []

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
            # 在segment_image函数中应用（替换原有yield前的代码）
            cropped_bgra = erode_non_transparent(cropped_bgra, erosion_percent=20)

            yield {
                "image": img,
                "cropped_bgra": cropped_bgra,
                "box": box,
                "cls": cls,
                "conf": conf,
                "stem": stem,
            }


def erode_non_transparent(bgra_image, erosion_percent=20):
    """
    对BGRA图像的非透明区域进行迭代腐蚀，直到非透明像素数量减少到原始数量的指定百分比。

    参数:
        bgra_image: 形状为 (H, W, 4) 的numpy数组，BGRA格式。
        erosion_percent: 目标百分比（0-100），表示腐蚀后剩余非透明像素数量占原始数量的百分比。
                        默认20，即减少到原始数量的20%。

    返回:
        腐蚀后的BGRA图像，Alpha通道被更新。
    """
    # 提取Alpha通道
    alpha = bgra_image[:, :, 3].copy()
    # 创建二值掩码（非透明区域）
    mask = (alpha > 0).astype(np.uint8)

    # 如果没有非透明像素，直接返回原图
    original_count = cv2.countNonZero(mask)
    if original_count == 0:
        return bgra_image

    # 计算目标像素数量
    target_count = int(original_count * erosion_percent / 100.0)
    if target_count <= 0:
        target_count = 1  # 至少保留一个像素

    # 如果已经达到或低于目标，直接返回
    if original_count <= target_count:
        return bgra_image

    # 准备腐蚀核（3x3矩形核，每次腐蚀1像素）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # 添加1像素边框以确保边界正确处理
    border_size = 2
    bordered = cv2.copyMakeBorder(
        mask,
        border_size,
        border_size,
        border_size,
        border_size,
        cv2.BORDER_CONSTANT,
        value=(0,0,0,0),
    )

    current_mask = bordered
    current_count = original_count
    max_iterations = 1000  # 安全上限
    iteration = 0

    while current_count > target_count and iteration < max_iterations:
        # 腐蚀一次
        eroded = cv2.erode(current_mask, kernel)
        new_count = cv2.countNonZero(eroded)
        # 如果腐蚀后像素数量不再减少，跳出循环
        if new_count >= current_count:
            break
        current_mask = eroded
        current_count = new_count
        iteration += 1

    # 移除边框
    eroded_mask = current_mask[border_size:-border_size, border_size:-border_size]

    # 检查是否还有非透明像素
    if cv2.countNonZero(eroded_mask) == 0:
        # 如果腐蚀导致所有像素消失，则保留一个像素（原始掩码的中心点）
        center_y, center_x = np.array(mask.nonzero()).mean(axis=1).astype(int)
        eroded_mask = np.zeros_like(mask)
        eroded_mask[center_y, center_x] = 1

    # 更新Alpha通道：将腐蚀后的掩码乘以255（二值）
    result = bgra_image.copy()
    result[:, :, 3] = eroded_mask * 255
    return result
