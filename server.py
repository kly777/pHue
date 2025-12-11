# realtime.py
import cv2
import numpy as np
import asyncio
import time
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
from ultralytics.models import YOLO
import os

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


# 导入新模块
from segmentation import segment_image
from color_analysis import extract_colors_from_patch
from color_correction import correct_color_by_reference
from ph_measurement import calculate_ph_value


def process_frame(frame: np.ndarray, device_id: str) -> dict:
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
    # 使用segment_image函数处理帧
    from pathlib import Path
    import tempfile

    # 创建临时文件来保存帧
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        temp_path = tmp_file.name
        cv2.imwrite(temp_path, frame)

    try:
        # 处理图像
        img_path = Path(temp_path)
        detected_objects = []

        for obj in segment_image(model, img_path, CONF_THRESHOLD):
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

            # 获取边界框信息
            x1, y1, x2, y2 = obj["box"]
            width, height = frame.shape[1], frame.shape[0]

            # 归一化边界框坐标
            x_center = ((x1 + x2) / 2) / width
            y_center = ((y1 + y2) / 2) / height
            bbox_width = (x2 - x1) / width
            bbox_height = (y2 - y1) / height

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
    finally:
        # 清理临时文件
        if Path(temp_path).exists():
            Path(temp_path).unlink()

    objects = detected_objects

    return {
        "objects": objects,
        "metadata": {
            "device_id": device_id,
            "timestamp": time.time(),
            "frame_size": [width, height],
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
        return {
            "type": "result",
            "payload": result,
            "latency": {
                "server_processing": round(processing_time, 1),
                "total": round(server_latency, 1),
            },
            "timestamp": time.time(),
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
