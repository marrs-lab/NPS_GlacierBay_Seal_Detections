from Scripts import yolov8m_seg_detection_workflow as detection_workflow
from Scripts import draw_detections as draw_detections

import time
import os

def Full_System_Run():
    start_time = time.time()
    print("Current working directory:", os.getcwd())

    # Parameters
    IMAGE_DIRECTORY = "Sample_Images/"
    MODEL_PATH = "Models/seal-segmentation-v2-1/weights/best.pt"
    CONFIDENCE = 0.8
    DRAW = True

    # Create REPROC folder inside image directory
    reproc_base = os.path.join(IMAGE_DIRECTORY, "REPROC")
    os.makedirs(reproc_base, exist_ok=True)

    # Run detection workflow inside REPROC
    csv_output_path = detection_workflow.process_images(
        img_dir=IMAGE_DIRECTORY,
        model_dir=MODEL_PATH,
        conf_threshold=CONFIDENCE,
        draw=False,
        output_dir=reproc_base
    )

    # Run drawing on results if enabled
    if DRAW:
        draw_detections.read_csv_and_draw(csv_path=csv_output_path, image_dir=IMAGE_DIRECTORY)

    # Runtime summary
    end_time = time.time()
    elapsed = end_time - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Completed Full System Run in {int(hours)}h {int(minutes)}m {int(seconds)}s.")

if __name__ == "__main__":
    Full_System_Run()
