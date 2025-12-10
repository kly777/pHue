import cv2
import time
import numpy as np
from ultralytics.models import YOLO
import argparse
import mss
import mss.tools


def main():
    parser = argparse.ArgumentParser(
        description="Real-time segmentation on screen capture using BEST model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="runs/seg/weights/best.pt",
        help="Path to model weights (default: runs/seg/weights/best.pt)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.5, help="Confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--region",
        type=str,
        default="full",
        help='Screen region to capture in format "x,y,width,height" or "full" (default: full screen)',
    )
    parser.add_argument("--show-fps", action="store_true", help="Display FPS on frame")
    parser.add_argument(
        "--output", type=str, default=None, help="Save video to file (optional)"
    )
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model}...")
    model = YOLO(args.model)
    print("Model loaded.")

    # Determine capture region
    with mss.mss() as sct:
        if args.region == "full":
            monitor = sct.monitors[1]  # primary monitor (index 1 is the entire screen)
        else:
            try:
                x, y, w, h = map(int, args.region.split(","))
                monitor = {"left": x, "top": y, "width": w, "height": h}
            except Exception as e:
                print(f"Invalid region format: {args.region}. Using full screen.")
                monitor = sct.monitors[1]
        print(f"Capturing region: {monitor}")

    # Video writer if needed
    out = None
    if args.output:
        fourcc = cv2.VideoWriter.fourcc("m", "p", "4", "v")  # 替换原第56行代码
        out = cv2.VideoWriter(
            args.output, fourcc, 20.0, (monitor["width"], monitor["height"])
        )

    print("Press 'q' to quit, 's' to save current frame.")
    fps_history = []
    try:
        with mss.mss() as sct:
            while True:
                # Capture screen
                screenshot = sct.grab(monitor)
                # Convert to numpy array (BGRA)
                frame = np.array(screenshot)
                # Convert BGRA to BGR (drop alpha)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

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
                cv2.imshow("BEST Screen Segmentation", annotated_frame)

                # Write to video if needed
                if out is not None:
                    out.write(annotated_frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"screen_capture_{timestamp}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"Saved frame to {filename}")

    finally:
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        print("Screen capture stopped.")


if __name__ == "__main__":
    main()
