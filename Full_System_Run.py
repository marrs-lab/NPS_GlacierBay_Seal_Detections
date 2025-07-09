from Scripts import yolov8m_seg_detection_workflow as detection_workflow
from Scripts import draw_detections as draw_detections
from Scripts import seal_ice_analysis as seal_ice_analysis
import time
import os

def Full_System_Run():
    start_time = time.time()
    print("Current working directory:", os.getcwd())

    # Parameters
    IMAGE_DIRECTORY = "Sample_Images/"
    MODEL_PATH = "Models/seal-segmentation-v2-1/weights/best.pt"
    CONFIDENCE = 0.8
    DRAW_SEALS = True
    DRAW_ICE = True

    # Create REPROC folder inside image directory
    reproc_base = os.path.join(IMAGE_DIRECTORY, "REPROC")
    os.makedirs(reproc_base, exist_ok=True)

    # Run detection workflow inside REPROC
    csv_output_path = detection_workflow.process_images(
        img_dir=IMAGE_DIRECTORY,
        model_dir=MODEL_PATH,
        output_dir=None,
        conf_threshold=CONFIDENCE,
        draw=False
        )

    # Run drawing on results if enabled
    if DRAW_SEALS:
        draw_detections.read_csv_and_draw(
            csv_path=csv_output_path, 
            image_dir=IMAGE_DIRECTORY,
            output_dir=None
            )
    
    # Run seal ice analysis
    seal_ice_analysis.analyze_seal_ice(
        csv_path=csv_output_path,
        image_dir=IMAGE_DIRECTORY,
        output_dir=None,
        save_images=DRAW_ICE
        )

    # Runtime summary
    end_time = time.time()
    elapsed = end_time - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Completed Full System Run in {int(hours)}h {int(minutes)}m {int(seconds)}s.")

if __name__ == "__main__":
    Full_System_Run()
