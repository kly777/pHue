"""
屏幕pH检测：捕获屏幕区域，使用YOLO分割模型检测pH试纸，并计算pH值。
支持两种屏幕捕获方式：mss（优先）和pyautogui（备用）。
"""
import cv2
import numpy as np
import time
import argparse
import sys
import os
sys.path.insert(0, '.')

from segmentation import segment_image
from color_analysis import extract_colors_from_patch
from color_correction import correct_color_by_reference
from ph_measurement import calculate_ph_value
from ultralytics.models import YOLO

# 加载模型（与server.py相同）
MODEL_PATH = "seg/weights/best12072154.pt"
CONF_THRESHOLD = 0.5

def load_model():
    """加载YOLO分割模型"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")
    print(f"加载分割模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    return model

def process_screen_frame(frame, model, device_id="screen"):
    """
    处理单帧屏幕图像，返回与server.py相同格式的结果。
    参数:
        frame: BGR numpy数组
        model: 加载的YOLO模型
        device_id: 设备标识符
    返回:
        dict: 包含objects和metadata的字典
    """
    # 调整帧大小以加速推理（最大尺寸640）
    orig_height, orig_width = frame.shape[:2]
    max_dim = 640
    scale = 1.0
    if orig_width > max_dim or orig_height > max_dim:
        if orig_width >= orig_height:
            new_width = max_dim
            new_height = int(orig_height * max_dim / orig_width)
        else:
            new_height = max_dim
            new_width = int(orig_width * max_dim / orig_height)
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        scale_x = orig_width / new_width
        scale_y = orig_height / new_height
    else:
        resized_frame = frame
        scale_x = 1.0
        scale_y = 1.0
        new_width, new_height = orig_width, orig_height

    detected_objects = []

    for obj in segment_image(model, resized_frame, CONF_THRESHOLD):
        # 提取颜色
        colored_color, uncolored_color = extract_colors_from_patch(
            obj["cropped_bgra"]
        )

        # 计算pH值
        pH_value = "未知"
        if colored_color is not None and uncolored_color is not None:
            corrected_colored_color = correct_color_by_reference(
                colored_color, uncolored_color
            )
            pH_value = calculate_ph_value(corrected_colored_color)

        # 获取边界框信息（相对于调整大小后的帧）
        x1, y1, x2, y2 = obj["box"]
        # 缩放回原始尺寸
        x1 = x1 * scale_x
        y1 = y1 * scale_y
        x2 = x2 * scale_x
        y2 = y2 * scale_y

        # 归一化边界框坐标（相对于原始尺寸）
        x_center = ((x1 + x2) / 2) / orig_width
        y_center = ((y1 + y2) / 2) / orig_height
        bbox_width = (x2 - x1) / orig_width
        bbox_height = (y2 - y1) / orig_height

        # 添加到结果中
        detected_objects.append(
            {
                "label": model.names[int(obj["cls"])]
                if int(obj["cls"]) in model.names
                else str(int(obj["cls"])),
                "confidence": float(obj["conf"]),
                "x": float(x_center),
                "y": float(y_center),
                "width": float(bbox_width),
                "height": float(bbox_height),
                "ph_value": pH_value,
            }
        )

    # 构建结果字典
    result = {
        "objects": detected_objects,
        "metadata": {
            "device_id": device_id,
            "timestamp": time.time(),
            "frame_size": [orig_width, orig_height],
            "model": "YOLO11n-seg",
        },
    }
    return result

def get_screen_capturer(region):
    """
    返回一个捕获函数，该函数返回BGR帧。
    使用pyautogui进行屏幕捕获。
    """
    import pyautogui
    # 解析区域
    if region == "full":
        screen_width, screen_height = pyautogui.size()
        monitor = {"left": 0, "top": 0, "width": screen_width, "height": screen_height}
    else:
        x, y, w, h = map(int, region.split(","))
        monitor = {"left": x, "top": y, "width": w, "height": h}
    print(f"使用pyautogui捕获区域: {monitor}")
    def capture():
        # pyautogui.screenshot 返回 PIL Image
        screenshot = pyautogui.screenshot(region=(monitor["left"], monitor["top"], monitor["width"], monitor["height"]))
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame
    return capture, monitor

def main():
    parser = argparse.ArgumentParser(description="屏幕pH检测")
    parser.add_argument(
        "--region",
        type=str,
        default="full",
        help='屏幕区域，格式 "x,y,width,height" 或 "full"（默认全屏）',
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="捕获间隔（秒），默认2秒",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="保存结果到JSON文件（可选）",
    )
    args = parser.parse_args()

    # 加载模型
    model = load_model()

    # 获取捕获函数和区域信息
    capture_func, monitor = get_screen_capturer(args.region)

    print(f"开始屏幕pH检测，每 {args.interval} 秒处理一帧。按 Ctrl+C 停止。")
    try:
        while True:
            # 捕获屏幕
            frame = capture_func()

            # 处理帧
            start_time = time.perf_counter()
            result = process_screen_frame(frame, model, device_id="screen")
            processing_time = time.perf_counter() - start_time

            # 打印结果
            print(f"\n=== 处理完成，耗时 {processing_time*1000:.1f} ms ===")
            print(f"帧尺寸: {result['metadata']['frame_size']}")
            objects = result["objects"]
            if objects:
                for idx, obj in enumerate(objects):
                    print(f"对象 {idx}: 标签={obj['label']}, 置信度={obj['confidence']:.3f}, pH={obj['ph_value']}")
            else:
                print("未检测到pH试纸")

            # 如果需要，保存结果到JSON文件
            if args.output:
                import json
                with open(args.output, 'a') as f:
                    json.dump(result, f, indent=2)
                    f.write("\n")

            # 等待间隔
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n检测停止。")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()