from ultralytics.models import YOLO
import os


def main():
    # Force offline mode to prevent download attempts
    os.environ["ULTRALYTICS_OFFLINE"] = "1"

    # Load a pretrained segmentation model
    # Use local file path to avoid network download
    model_path = "./yolo11n-seg.pt"
    if not os.path.exists(model_path):
        model_path = "yolo11n-seg.pt"
    model = YOLO(model_path)  # you can change to 'yolov8s-seg.pt' etc.

    # Build absolute path to config file
    config_path = os.path.join(os.path.dirname(__file__), "configs/data.yaml")
    # Train the model with only 1 epoch for quick test
    results = model.train(
        data=config_path,
        epochs=100,
        imgsz=640,
        batch=8,
        device="cpu",
        workers=0,
        seed=42,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.0003,
        lrf=0.1,
        cos_lr=True,
        save=True,
        save_period=10,
        project="runs",
        name="seg",
        exist_ok=True,
        verbose=True,
        augment=True,
        # 几何变换增强
        degrees=90.0,  # 旋转角度范围(-deg,+deg)，推荐10-30
        translate=0.1,  # 平移比例，推荐0.1-0.2
        scale=0.5,  # 缩放比例(1-scale, 1+scale)，推荐0.5-0.9
        shear=2.0,  # 剪切强度，推荐1.0-5.0
        perspective=0.001,  # 透视变换，推荐0.000-0.001
        # 高级增强（分割任务特别推荐）
        mosaic=0.15,  # 马赛克增强概率，0-3.0（3.0=100%）
        mixup=0.1,  # MixUp增强概率，推荐0.0-0.2
        copy_paste=0.2,  # 复制粘贴增强（分割专用），推荐0.1-0.5
        # 颜色增强（可选加强）
        hsv_h=0.015,  # 色调增强
        hsv_s=0.7,  # 饱和度增强
        hsv_v=0.4,  # 明度增强
    )

    # Evaluate on validation set
    metrics = model.val()
    # print(f"mAP50-95: {metrics.box.map}")  # segmentation metrics
    # print(f"Metrics: {metrics}")

    # Export to ONNX if needed
    # model.export(format="onnx", simplify=True)


if __name__ == "__main__":
    main()
