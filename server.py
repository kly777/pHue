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


def process_frame(frame: np.ndarray, device_id: str) -> dict:
    """
    使用 YOLO 分割模型处理帧
    返回格式: {
        "objects": [
            {
                "label": str,
                "confidence": float,
                "x": float,  # 归一化中心点x
                "y": float,  # 归一化中心点y
                "width": float,  # 归一化宽度
                "height": float,  # 归一化高度
                "mask_polygons": List[List[float]]  # 可选，掩码多边形坐标 (归一化)
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
    height, width = frame.shape[:2]

    # 运行推理
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)

    objects = []
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()  # (N, H, W)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []
        classes = results[0].boxes.cls.cpu().numpy() if results[0].boxes else []
        confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes else []

        for i, (mask, box, cls, conf) in enumerate(zip(masks, boxes, classes, confs)):
            # 边界框归一化坐标 (x_center, y_center, width, height)
            x1, y1, x2, y2 = box
            x_center = ((x1 + x2) / 2) / width
            y_center = ((y1 + y2) / 2) / height
            bbox_width = (x2 - x1) / width
            bbox_height = (y2 - y1) / height

            # 获取掩码多边形 (可选)
            # 将掩码调整到原始尺寸并提取轮廓
            mask_resized = cv2.resize(
                mask, (width, height), interpolation=cv2.INTER_NEAREST
            )
            contours, _ = cv2.findContours(
                (mask_resized > 0.5).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            polygons = []
            for contour in contours:
                # 简化多边形，减少点数
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                # 归一化坐标
                normalized = approx.squeeze(1).astype(float)
                if len(normalized) > 0:
                    normalized[:, 0] /= width
                    normalized[:, 1] /= height
                    polygons.append(normalized.tolist())

            objects.append(
                {
                    "label": model.names[int(cls)]
                    if int(cls) in model.names
                    else str(int(cls)),
                    "confidence": float(conf),
                    "x": float(x_center),
                    "y": float(y_center),
                    "width": float(bbox_width),
                    "height": float(bbox_height),
                    "mask_polygons": polygons if polygons else None,
                }
            )

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
