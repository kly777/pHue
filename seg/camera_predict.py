import cv2
import time
from ultralytics.models import YOLO
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Real-time segmentation with camera using BEST model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="runs/seg/weights/best.pt",
        help="Path to model weights (default: runs/seg/weights/best.pt)",
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.5, help="Confidence threshold (default: 0.5)"
    )
    parser.add_argument("--show-fps", action="store_true", help="Display FPS on frame")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model}...")
    model = YOLO(args.model)
    print("Model loaded.")

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {args.camera}")
        return

    print("Press 'q' to quit, 's' to save current frame.")
    fps_history = []
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Run inference
            start_time = time.perf_counter()
            results = model(frame, conf=args.conf, verbose=False)
            inference_time = time.perf_counter() - start_time

            # Calculate FPS
            fps = 1.0 / inference_time if inference_time > 0 else 0
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history) if fps_history else fps

            # Annotate frame
            annotated_frame = results[0].plot()  # BGR image with masks and boxes

            # Display FPS if requested
            if args.show_fps:
                cv2.putText(
                    annotated_frame,
                    f"FPS: {avg_fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

            # Show result
            cv2.imshow("BEST Segmentation", annotated_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"Saved frame to {filename}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released.")


if __name__ == "__main__":
    main()
