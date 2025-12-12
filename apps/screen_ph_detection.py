"""
屏幕pH检测可视化：捕获屏幕，显示检测框、分割mask和颜色方块。
"""

import cv2
import numpy as np
import time
import argparse
import sys
import os

sys.path.insert(0, ".")

from src.segmentation import segment_image
from src.color_analysis import extract_colors_from_patch
from src.ph_measurement import calculate_ph_value
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


def process_frame_with_visualization(frame, model):
    """
    处理单帧屏幕图像，返回用于可视化的数据。
    参数:
        frame: BGR numpy数组
        model: 加载的YOLO模型
    返回:
        tuple: (vis_frame, objects_info)
        vis_frame: 用于显示的图像（包含边界框、pH值等）
        objects_info: 列表，每个元素为字典，包含：
            'box': (x1, y1, x2, y2) 原始坐标
            'cropped_bgra': 裁剪的BGRA图像 (H, W, 4)
            'colored_hsv': 变色部分HSV颜色 (tuple)
            'uncolored_hsv': 未变色部分HSV颜色 (tuple)
            'ph_value': pH值 (float 或 "未知")
            'confidence': 置信度
            'label': 标签
    """
    orig_height, orig_width = frame.shape[:2]
    max_dim = 640
    if orig_width > max_dim or orig_height > max_dim:
        if orig_width >= orig_height:
            new_width = max_dim
            new_height = int(orig_height * max_dim / orig_width)
        else:
            new_height = max_dim
            new_width = int(orig_width * max_dim / orig_height)
        resized_frame = cv2.resize(
            frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR
        )
        scale_x = orig_width / new_width
        scale_y = orig_height / new_height
    else:
        resized_frame = frame
        scale_x = 1.0
        scale_y = 1.0
        new_width, new_height = orig_width, orig_height

    objects_info = []

    for obj in segment_image(model, resized_frame, CONF_THRESHOLD):
        # 提取颜色
        colored_hsv, uncolored_hsv = extract_colors_from_patch(obj["cropped_bgra"])

        # 计算pH值
        ph_value = "未知"
        if colored_hsv is not None and uncolored_hsv is not None:
            ph_value = calculate_ph_value(colored_hsv, uncolored_hsv, color_space="hsv")

        # 边界框缩放回原始尺寸
        x1, y1, x2, y2 = obj["box"]
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        objects_info.append(
            {
                "box": (x1, y1, x2, y2),
                "cropped_bgra": obj["cropped_bgra"],
                "colored_hsv": colored_hsv,
                "uncolored_hsv": uncolored_hsv,
                "ph_value": ph_value,
                "confidence": float(obj["conf"]),
                "label": model.names[int(obj["cls"])]
                if int(obj["cls"]) in model.names
                else str(int(obj["cls"])),
            }
        )

    # 创建可视化帧（原始帧的副本）
    vis_frame = frame.copy()

    # 绘制每个检测对象
    for obj in objects_info:
        x1, y1, x2, y2 = obj["box"]
        # 绘制边界框
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 标签文本
        label_text = f"{obj['label']} {obj['confidence']:.2f} pH:{obj['ph_value']}"
        cv2.putText(
            vis_frame,
            label_text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    return vis_frame, objects_info


def create_visualization_panel(original_frame, objects_info, panel_width=400):
    """
    创建右侧可视化面板。
    参数:
        original_frame: 原始帧（用于参考尺寸）
        objects_info: 来自process_frame_with_visualization的对象列表
        panel_width: 右侧面板宽度
    返回:
        panel: BGR图像，高度与原始帧相同
    """
    orig_h, orig_w = original_frame.shape[:2]
    panel = np.zeros((orig_h, panel_width, 3), dtype=np.uint8)
    panel.fill(240)  # 浅灰色背景

    if not objects_info:
        # 无检测对象时显示提示
        cv2.putText(
            panel,
            "No pH strip detected",
            (50, orig_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (100, 100, 100),
            2,
        )
        return panel

    # 只显示第一个检测对象（假设只有一个）
    obj = objects_info[0]

    # 上半部分：分割出的mask（裁剪后的BGRA转换为BGR）
    cropped_bgra = obj["cropped_bgra"]
    # 将BGRA转换为BGR，并将alpha为0的像素设为黑色
    cropped_bgr = cropped_bgra[:, :, :3].copy()
    cropped_bgr[cropped_bgra[:, :, 3] == 0] = [0, 0, 0]
    # 调整大小以适应面板上半部分
    mask_height = orig_h // 2
    # 保持宽高比
    h, w = cropped_bgr.shape[:2]
    aspect = w / h
    new_w = int(mask_height * aspect)
    if new_w > panel_width:
        new_w = panel_width
        mask_height = int(new_w / aspect)
    resized_mask = cv2.resize(cropped_bgr, (new_w, mask_height))
    # 将mask放置在面板上半部分居中
    x_offset = (panel_width - new_w) // 2
    panel[0:mask_height, x_offset : x_offset + new_w] = resized_mask
    cv2.putText(
        panel, "Segmented Mask", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
    )

    # 下半部分：颜色方块
    square_size = 80
    margin = 20
    y_start = mask_height + margin
    # 变色部分颜色方块
    if obj["colored_hsv"] is not None:
        # 将HSV转换为BGR用于显示
        colored_hsv = obj["colored_hsv"]
        # 注意：colored_hsv是(H, S, V)格式，H在[0,360]，S,V在[0,1]
        # 转换为OpenCV HSV格式：H/2, S*255, V*255
        hsv_cv = np.array(
            [[[colored_hsv[0] / 2, colored_hsv[1] * 255, colored_hsv[2] * 255]]],
            dtype=np.uint8,
        )
        colored_bgr = cv2.cvtColor(hsv_cv, cv2.COLOR_HSV2BGR)[0][0]
        colored_bgr = tuple(map(int, colored_bgr))
        cv2.rectangle(
            panel,
            (margin, y_start),
            (margin + square_size, y_start + square_size),
            colored_bgr,
            -1,
        )
        cv2.putText(
            panel,
            "Colored",
            (margin, y_start + square_size + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

    # 未变色部分颜色方块
    if obj["uncolored_hsv"] is not None:
        uncolored_hsv = obj["uncolored_hsv"]
        hsv_cv = np.array(
            [[[uncolored_hsv[0] / 2, uncolored_hsv[1] * 255, uncolored_hsv[2] * 255]]],
            dtype=np.uint8,
        )
        uncolored_bgr = cv2.cvtColor(hsv_cv, cv2.COLOR_HSV2BGR)[0][0]
        uncolored_bgr = tuple(map(int, uncolored_bgr))
        x_start = margin + square_size + margin
        cv2.rectangle(
            panel,
            (x_start, y_start),
            (x_start + square_size, y_start + square_size),
            uncolored_bgr,
            -1,
        )
        cv2.putText(
            panel,
            "Uncolored",
            (x_start, y_start + square_size + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

    # 显示pH值
    ph_text = f"pH: {obj['ph_value']}"
    cv2.putText(
        panel,
        ph_text,
        (panel_width - 150, y_start + square_size + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )

    return panel


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
        screenshot = pyautogui.screenshot(
            region=(
                monitor["left"],
                monitor["top"],
                monitor["width"],
                monitor["height"],
            )
        )
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    return capture, monitor


def main():
    parser = argparse.ArgumentParser(description="屏幕pH检测可视化")
    parser.add_argument(
        "--region",
        type=str,
        default="full",
        help='屏幕区域，格式 "x,y,width,height" 或 "full"（默认全屏）',
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="捕获间隔（秒），默认0.5秒（实时）",
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="禁用可视化窗口，仅控制台输出",
    )
    args = parser.parse_args()

    # 加载模型
    model = load_model()

    # 获取捕获函数和区域信息
    capture_func, monitor = get_screen_capturer(args.region)

    print(f"开始屏幕pH检测可视化，每 {args.interval} 秒处理一帧。按 'q' 退出。")
    try:
        while True:
            # 捕获屏幕
            frame = capture_func()

            # 处理帧
            start_time = time.perf_counter()
            vis_frame, objects_info = process_frame_with_visualization(frame, model)
            processing_time = time.perf_counter() - start_time

            # 控制台输出
            print(f"\n处理耗时: {processing_time * 1000:.1f} ms")
            if objects_info:
                for idx, obj in enumerate(objects_info):
                    print(
                        f"对象 {idx}: 标签={obj['label']}, 置信度={obj['confidence']:.3f}, pH={obj['ph_value']}"
                    )
            else:
                print("未检测到pH试纸")

            if not args.no_vis:
                # 创建右侧面板
                panel = create_visualization_panel(frame, objects_info, panel_width=400)
                # 合并左右图像
                combined = np.hstack([vis_frame, panel])
                # 显示
                cv2.imshow("pH Strip Detection", combined)
                # 等待按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            else:
                # 无可视化时，等待间隔
                time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n检测停止。")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        if not args.no_vis:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
