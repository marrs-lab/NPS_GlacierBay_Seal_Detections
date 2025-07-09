import torch
import random
import numpy as np
from ultralytics import YOLO
import time

# =========================================
MODEL_SIZE = "yolov8m-seg.pt"
BATCH_SIZE = 16
EPOCHS = 100
PATIENCE = 20
IMAGE_SIZE = 640
LEARNING_RATE = 1e-4
DATA_YAML = r"C:\Users\sa553\Desktop\NPS\NPS_GlacierBay_Seal_Detections\Roboflow_V2\data.yaml"
PROJECT_DIR = r"C:\Users\sa553\Desktop\NPS\NPS_GlacierBay_Seal_Detections\Models"
EXPERIMENT_NAME = "seal-segmentation-v2-4"
# =========================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def main():
    start_time = time.time()
    print("CUDA Available:", torch.cuda.is_available())
    print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")
    set_seed()

    model = YOLO(MODEL_SIZE)

    # First: Hyperparameter evolution using mutate (via tune)
    print("Starting hyperparameter tuning with mutation...")
    model.tune(
        data=DATA_YAML,
        epochs=20,  # shorter tuning cycle
        patience=10,
        iterations=100,  # number of mutation attempts
        optimizer="AdamW",
        val=True,
        plots=False,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        device=0
    )

    # Second: Train final model using best evolved hyperparameters
    print("Training final model with evolved hyperparameters...")
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT_DIR,
        name=EXPERIMENT_NAME,
        exist_ok=True,
        save=True,
        val=True,
        plots=True,
        patience=PATIENCE,
        optimizer="AdamW",
        device=0,
        amp=True, # Turn to false for next attempt
        augment=True,
        verbose=True,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total time = {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds.")

if __name__ == "__main__":
    main()
