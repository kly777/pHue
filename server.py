# realtime.py
import cv2
import numpy as np
import asyncio
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

# 全局连接管理 (设备ID -> WebSocket)
active_connections: Dict[str, WebSocket] = {}
executor = ThreadPoolExecutor(max_workers=4)  # CPU密集型任务线程池


def process_frame(frame: np.ndarray, device_id: str) -> dict:
    """
    实时帧处理函数
    返回格式: {
        "objects": [
            {"label": "person", "confidence": 0.95, "x":0.1, "y":0.2, "width":0.3, "height":0.4},
            ...
        ],
        "metadata": {"device_id": "abc123"}
    }
    """
    # 示例1: 简单边缘检测 (实际替换为YOLO等模型)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # 示例2: 模拟检测结果 (人脸区域)
    height, width = frame.shape[:2]
    return {
        "objects": [
            {
                "label": "face",
                "confidence": 0.88,
                "x": 0.4,  # 归一化中心点x
                "y": 0.3,
                "width": 0.2,  # 归一化宽度
                "height": 0.3,  # 归一化高度
            }
        ],
        "metadata": {
            "device_id": device_id,
            "timestamp": time.time(),
            "frame_size": [width, height],
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


def process_frame_data(data: bytes, device_id: str, receive_time: float) -> dict:
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

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
