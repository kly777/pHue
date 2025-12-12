# pHue 图像分割服务

本项目提供基于 YOLO 的实时图像分割服务，包括离线批量处理与实时 WebSocket 接口。

## 1. 批量图像分割脚本 (`main.py`)

### 功能

使用训练好的分割模型对单张图像或整个目录进行分割，将结果保存到 `out/` 文件夹。

### 使用方法

```bash
uv run python main.py --input <图像路径或目录> [选项]
```

### 参数说明

| 参数             | 说明                       | 默认值                        |
| ---------------- | -------------------------- | ----------------------------- |
| `--model`        | 模型权重路径               | `seg/weights/best12072154.pt` |
| `--input`        | 输入图像路径或目录（必需） | 无                            |
| `--output`       | 输出目录                   | `out`                         |
| `--conf`         | 置信度阈值                 | `0.5`                         |
| `--no-annotated` | 不保存带标注的图像         | `False`                       |
| `--no-masks`     | 不保存掩码和叠加图像       | `False`                       |

### 输出文件

对于每张输入图像，在输出目录中生成：

- `{图像名}_annotated.jpg`：带标注的图像（边界框 + 掩码）
- `{图像名}_mask_{索引}.png`：二值掩码图像
- `{图像名}_overlay_{索引}.jpg`：掩码叠加图像（绿色高亮）
- `{图像名}_crop_{索引}.jpg`：裁剪出的对象区域

### 示例

```bash
# 分割单张图像
uv run python main.py --input seg/OIP-2867035697.jpg

# 分割目录下所有图像
uv run python main.py --input seg/data/images/train/ --output results
```

## 2. 实时分割 WebSocket 服务 (`server.py`)

### 功能

提供 WebSocket 接口，接收客户端发送的图像帧（JPEG 格式），实时进行分割并返回检测结果。

### 启动服务

```bash
uv run python server.py
```

服务启动后监听 `0.0.0.0:8000`。

### 接口说明

#### 2.1 WebSocket 连接

- **URL**: `ws://<服务器地址>:8000/ws/realtime?device_id=<设备ID>`
- **参数**：
  - `device_id`：客户端唯一标识符（字符串），用于连接管理。
- **连接流程**：
    1. 客户端建立 WebSocket 连接。
    2. 服务端接受连接后，客户端可连续发送图像帧数据（JPEG 二进制流）。
    3. 服务端对每帧进行处理并返回 JSON 格式的结果。
    4. 客户端可随时断开连接。

#### 2.2 数据格式

##### 客户端发送

- 二进制数据：JPEG 编码的图像字节流（推荐尺寸不超过 1280x720，以降低延迟）。

##### 服务端返回

返回 JSON 对象，结构如下：

```json
{
  "type": "result",
  "payload": {
    "objects": [
      {
        "label": "pH",
        "confidence": 0.93,
        "x": 0.45,
        "y": 0.32,
        "width": 0.15,
        "height": 0.18,
        "mask_polygons": [[[0.1,0.2],[0.2,0.3],...]]
      }
    ],
    "metadata": {
      "device_id": "device123",
      "timestamp": 1733915296.123,
      "frame_size": [640, 480],
      "model": "YOLO11n-seg"
    }
  },
  "latency": {
    "server_processing": 45.2,
    "total": 67.8
  },
  "timestamp": 1733915296.124
}
```

##### 字段说明

- `objects`：检测到的对象数组，每个对象包含：
  - `label`：类别名称（字符串）。
  - `confidence`：置信度（0~1）。
  - `x`, `y`：边界框中心点归一化坐标（相对于图像宽度/高度）。
  - `width`, `height`：边界框归一化尺寸。
  - `mask_polygons`：可选，掩码多边形坐标列表，每个多边形为 `[[x1,y1],[x2,y2],...]` 归一化坐标。
- `metadata`：元数据，包含设备 ID、时间戳、图像尺寸、模型名称。
- `latency`：延迟信息（毫秒）。
- `timestamp`：服务器返回时间戳。

#### 2.3 健康检查

- **HTTP GET** `http://<服务器地址>:8000/health`
- 返回当前活跃连接数及服务状态：

```json
{ "status": "ok", "active_connections": 2 }
```

### 客户端示例（Python）

```python
import asyncio
import websockets
import cv2

async def send_frame():
    uri = "ws://localhost:8000/ws/realtime?device_id=test123"
    async with websockets.connect(uri) as websocket:
        # 读取图像并编码为 JPEG
        img = cv2.imread("test.jpg")
        _, jpeg_data = cv2.imencode(".jpg", img)
        await websocket.send(jpeg_data.tobytes())
        response = await websocket.recv()
        print(response)

asyncio.run(send_frame())
```

## 3. 模型训练 (`seg/train.py`)

如需训练模型，使用 `seg/train.py` 脚本。训练配置位于 `seg/configs/data.yaml`。

## 4. 环境配置

### 安装依赖

```bash
uv sync
```

## 5. 注意事项

- 模型加载需要一定时间，启动时请等待“加载分割模型”提示。
- 默认置信度阈值为 0.5，可在代码中修改 `CONF_THRESHOLD`。
- 掩码多边形提取会增加计算开销，若不需要可注释相关代码。
- 建议客户端控制发送频率，避免服务器过载。

## 6. 项目结构

```
pHue2/
├── src/                     # 核心模块
│   ├── segmentation.py      # 图像分割与腐蚀
│   ├── color_analysis.py    # 颜色分析
│   ├── ph_model.py          # 双颜色神经网络模型
│   ├── ph_measurement.py    # pH值计算
│   └── pHmap.py             # pH-颜色映射数据
├── apps/                    # 应用脚本
│   ├── main.py              # 批量分割脚本
│   ├── server.py            # 实时 WebSocket 服务
│   └── screen_ph_detection.py # 屏幕pH检测
├── train/                   # 训练脚本
│   └── train_two_colors.py  # 双颜色模型训练
├── tests/                   # 测试脚本
│   └── test_integration.py  # 集成测试
├── data/                    # 数据与模型
│   └── models/              # 训练好的模型权重
├── seg/                     # YOLO分割相关
│   ├── weights/             # 分割模型权重
│   ├── configs/             # 训练配置文件
│   ├── train.py             # 分割训练脚本
│   └── predict.py           # 分割预测示例
├── out/                     # 批量分割输出目录（自动创建）
├── pyproject.toml           # 项目依赖
├── uv.lock                  # 依赖锁文件
└── README.md                # 本文档
```

## 7. 许可证

MIT
