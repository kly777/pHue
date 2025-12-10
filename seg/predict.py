import cv2
from ultralytics import YOLO

import os


def main():
    # Load the trained model
    model_path = "runs/seg/weights/best.pt"
    if not os.path.exists(model_path):
        model_path = "runs/seg/weights/last.pt"
    model = YOLO(model_path)

    # Choose an image (you can change this)
    img_path = "OIP-2867035697.jpg"
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return

    # Run prediction
    results = model(img_path, conf=0.5)

    # Visualize results
    for result in results:
        # Plot with masks
        annotated_img = result.plot()  # returns BGR image with masks and boxes

        # Save the annotated image
        output_path = "prediction_result.jpg"
        cv2.imwrite(output_path, annotated_img)
        print(f"Saved result to {output_path}")

        # Print detection info
        if result.masks is not None:
            print(f"Detected {len(result.masks)} masks")
        if result.boxes is not None:
            print(f"Detected {len(result.boxes)} boxes")
            for box in result.boxes:
                print(
                    f"  Class: {model.names[int(box.cls)]}, Confidence: {box.conf.item():.2f}"
                )


if __name__ == "__main__":
    main()
