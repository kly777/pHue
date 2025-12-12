# realtime.py
import sys
import cv2
import numpy as np
import asyncio
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from ultralytics.models import YOLO
import os

sys.path.insert(0, ".")
from src.segmentation import segment_image
from src.color_analysis import extract_colors_from_patch
from src.ph_measurement import calculate_ph_value

app = FastAPI()

# 全局连接管理 (设备ID -> WebSocket)
active_connections: Dict[str, WebSocket] = {}
executor = ThreadPoolExecutor(max_workers=4)  # CPU密集型任务线程池

# 加载分割模型
MODEL_PATH = "seg/weights/best12072154.pt"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")
print(f"加载分割模型: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
CONF_THRESHOLD = 0.5


def process_frame(frame: np.ndarray, device_id: str) -> Dict[str, Any]:
    """
    使用 YOLO 分割模型处理帧，并计算pH值
    返回格式: {
        "objects": [
            {
                "label": str,
                "confidence": float,
                "x": float,  # 归一化中心点x
                "y": float,  # 归一化中心点y
                "width": float,  # 归一化宽度
                "height": float,  # 归一化高度
                "ph_value": float  # 预测的pH值
            },
            ...
        ],
        "metadata": {
            "device_id": str,
            "timestamp": float,
            "frame_size": [width, height],
            "model": "YOLO11n-seg"
        }
    }
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

    # 使用segment_image函数处理帧，直接传递numpy数组
    detected_objects = []

    for obj in segment_image(model, resized_frame, CONF_THRESHOLD):
        # 提取颜色
        colored_color, uncolored_color = extract_colors_from_patch(obj["cropped_bgra"])

        # 计算pH值
        pH_value = "未知"
        if colored_color is not None and uncolored_color is not None:
            pH_value = calculate_ph_value(
                colored_color, uncolored_color, color_space="hsv"
            )

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

    objects = detected_objects

    # 确保所有数值类型可JSON序列化
    def convert_types(obj):
        if type(obj).__module__ == "numpy":
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        else:
            return obj

    # 使用原始尺寸
    width = orig_width
    height = orig_height

    result_dict = {
        "objects": objects,
        "metadata": {
            "device_id": device_id,
            "timestamp": time.time(),
            "frame_size": [width, height],
            "model": "YOLO11n-seg",
        },
    }

    converted_result = convert_types(result_dict)

    # 确保返回值是字典类型
    if isinstance(converted_result, dict):
        return converted_result
    else:
        # 如果转换后不是字典，构建默认结果
        return {
            "objects": [],
            "metadata": {
                "device_id": device_id,
                "timestamp": time.time(),
                "frame_size": [640.0, 480.0],
                "model": "YOLO11n-seg",
            },
        }


@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket, device_id: str):
    await websocket.accept()

    # 注册连接
    active_connections[device_id] = websocket
    print(f"设备 {device_id} 已连接，当前连接数: {len(active_connections)}")

    try:
        while True:
            # 接收二进制帧数据 (小程序发送的JPG)
            data = await websocket.receive_bytes()
            receive_time = time.time()

            # 异步处理 (避免阻塞WebSocket)
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                executor, lambda: process_frame_data(data, device_id, receive_time)
            )

            # 发送结果
            if result:
                await websocket.send_json(result)

    except WebSocketDisconnect:
        print(f"设备 {device_id} 断开连接")
    finally:
        active_connections.pop(device_id, None)


def process_frame_data(
    data: bytes, device_id: str, receive_time: float
) -> Optional[dict]:
    """处理单帧数据 (在独立线程中运行)"""
    try:
        # 1. 解码图像
        nparr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None or frame.size == 0:
            return None

        # 获取帧尺寸
        height, width = frame.shape[:2]

        # 2. 处理帧 (调用AI模型)
        start_time = time.time()
        result = process_frame(frame, device_id)
        processing_time = (time.time() - start_time) * 1000  # ms

        # 3. 构建响应
        server_latency = (time.time() - receive_time) * 1000
        # 确保所有数值类型可JSON序列化
        payload = result

        return {
            "type": "result",
            "payload": payload,
            "latency": {
                "server_processing": float(round(processing_time, 1)),
                "total": float(round(server_latency, 1)),
            },
            "timestamp": float(time.time()),
        }

    except Exception as e:
        print(f"处理帧错误: {str(e)}")
        return None


# 健康检查
@app.get("/health")
async def health_check():
    return {"status": "ok", "active_connections": len(active_connections)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
