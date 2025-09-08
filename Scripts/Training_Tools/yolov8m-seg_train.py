import torch
import random
import numpy as np
from ultralytics import YOLO
import time

# =========================================
MODEL_SIZE = "yolov8m.pt"
BATCH_SIZE = 16
EPOCHS = 300
PATIENCE = 20
IMAGE_SIZE = 640
LEARNING_RATE = 1e-4
EXPERIMENT_NAME = "seal-box-v3-1"
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

    model.train(
        data=r"C:\Users\sa553\Desktop\NPS\NPS_GlacierBay_Seal_Detections\Roboflow_V3\data.yaml",
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project=r"C:\Users\sa553\Desktop\NPS\NPS_GlacierBay_Seal_Detections\Models",
        name=EXPERIMENT_NAME,
        exist_ok=True,
        save=True,
        save_period=-1,
        val=True,
        plots=True,
        patience=PATIENCE,
        optimizer="AdamW",
        lr0=LEARNING_RATE,
        device=0,
        amp=True,
        augment=True,
        verbose=True
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Time script took to run = {int(hours)} hours, {int(minutes)} minutes, and {int(seconds)} seconds.")

if __name__ == "__main__":
    main()
