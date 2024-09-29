# -*- coding: utf-8 -*-
"""
   Created on Mon Nov  6 10:46:53 2023

   @author: Saagar
   
   Meant for training the model. Create data with crops using 
   labelme2yolo --json_dir /path/to/labelme_json_dir/
   data=dataset.yaml

   """

from ultralytics import YOLO
import time
start_time = time.time()

# create the yaml dataset for the crops
# labelme2yolo --json_dir /path/to/labelme_json_dir/

# resume training
# # Load a model
# model = YOLO('path/to/last.pt')  # load a partially trained model

# # Resume training
# results = model.train(resume=True)

# load a model
ColorAndInvert = YOLO('yolov8s-seg.yaml')  # new model from scratch
Color = YOLO('yolov8s-seg.yaml')
Inverse = YOLO('yolov8s-seg.yaml')

if __name__ == '__main__':
  
    resultsOne = MODEL.train(
    data=r'',
    epochs=200,           # Increase the number of epochs for more training
    imgsz=640,
    batch=64,             # Adjust the batch size based on available GPU memory (start with 16)
    name="",
    project=r'',
    workers=8,            # Adjust the number of worker threads for data loading
    device='0',           # Utilize the GPU for training
    lr0=0.01,            # Experiment with different initial learning rates
    lrf=0.01,              # Experiment with different final learning rate factors
    cos_lr=False,         # Enable cosine learning rate scheduler
    warmup_epochs=5.0,    # Experiment with different warm-up epochs
    box=7.5,              # Adjust the box loss gain
    cls=0.5,              # Adjust the cls loss gain
    dfl=1.5,              # Adjust the dfl loss gain
    pose=12.0,             # Adjust the pose loss gain
    kobj=2.0,             # Adjust the keypoint obj loss gain
    label_smoothing=0.0,  # Experiment with label smoothing
    nbs=64,               # Adjust nominal batch size based on GPU memory
    overlap_mask=True,    # Masks should overlap during training
    mask_ratio=4,         # Experiment with mask downsample ratio
    dropout=0.5,          # Experiment with dropout regularization
    val=True,             # Enable validation during training
    plots=False,            # Disable saving plots and images during train/val
)

end_time = time.time()
elapsed_time = end_time - start_time
hours, remainder = divmod(elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Time script took to run = {int(hours)} hours, {int(minutes)} minutes, and {int(seconds)} seconds.")
