from Scripts import yolov8m_seg_detection_workflow as detection_workflow
from Scripts import draw_detections as draw_detections
from Scripts import seal_ice_analysis as seal_ice_analysis
import time
import multiprocessing
import os


def Full_System_Run(IMAGE_DIRECTORY):
    start_time = time.time()
    print("Running on:", IMAGE_DIRECTORY)

    # Parameters
    MODEL_PATH = "Models/seal-segmentation-v2-1/weights/best.pt"    # Path to YOLO model
    CONFIDENCE = 0.7                                               # Confidence of detections
    DRAW_SEALS = False                                              # Draw only seals after detections
    DRAW_SEALS_ON_ICE = True                                        # Draw seals and trace ice
    DATA_SAMPLING = True                                            # Enable Data-sampling
    SAMPLING_SIZE = 20                                              # Choose x random images after drawing seals and tracing ice

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
        save_images=DRAW_SEALS_ON_ICE,
        enable_sampling=DATA_SAMPLING,
        sample_size=SAMPLING_SIZE
        )

    # Runtime summary
    end_time = time.time()
    elapsed = end_time - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Completed run for {IMAGE_DIRECTORY} in {int(hours)}h {int(minutes)}m {int(seconds)}s.")


if __name__ == "__main__":
    BASE_DIR = r"Z:\Projects\Clients\NPS_GlacierBay\2023\WingtraPilotProjects"   # change this to your top-level directory

    for day in os.listdir(BASE_DIR):
        day_path = os.path.join(BASE_DIR, day)
        if os.path.isdir(day_path):
            for flight in os.listdir(day_path):
                output_path = os.path.join(day_path, flight, "OUTPUT")
                if os.path.isdir(output_path):
                    Full_System_Run(output_path)

#    dir = r""
#    Full_System_Run(dir)
