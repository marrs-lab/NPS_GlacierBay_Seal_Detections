# -*- coding: utf-8 -*-
"""
   Created on Mon Nov  6 10:46:53 2023

   @author: Saagar
   
   Meant for training the model. Create data with crops using 
   labelme2yolo --json_dir /path/to/labelme_json_dir/
   labelme2yolo --json_dir /path/to/labelme_json_dir/ --val_size 0.2 --test_size 0.05
   data=dataset.yaml

   """

from ultralytics import YOLO
import time
start_time = time.time()

# create the yaml dataset for the crops
# labelme2yolo --json_dir /path/to/labelme_json_dir/
# labelme2yolo --json_dir /path/to/labelme_json_dir/ --val_size 0.2 --test_size 0.1


# resume training
# # Load a model
# model = YOLO('path/to/last.pt')  # load a partially trained model

# # Resume training
# results = model.train(resume=True)

# load a model
Combined = YOLO('yolov8s-seg.yaml')  # new model from scratch
Color = YOLO('yolov8s-seg.yaml')
Inverse = YOLO('yolov8s-seg.yaml')

if __name__ == '__main__':
  
    resultsOne = Combined.train(
    data=r'',
    epochs=200,           # Increase the number of epochs for more training
    imgsz=640,
    batch=64,             # Adjust the batch size based on available GPU memory (start with 16)
    name="",     #V10 cos True. V11 cos False. 
    project=r'',
    workers=8,            # Adjust the number of worker threads for data loading
    device='0',           # Utilize the GPU for training
    lr0=0.002,            # Experiment with different initial learning rates
    lrf=0.1,              # Experiment with different final learning rate factors
    cos_lr=True,         # Enable cosine learning rate scheduler
    warmup_epochs=5.0,    # Experiment with different warm-up epochs
    box=6.25,              # Adjust the box loss gain
    cls=0.3,              # Adjust the cls loss gain
    dfl=1.5,              # Adjust the dfl loss gain
    pose=12.0,             # Adjust the pose loss gain
    kobj=2.0,             # Adjust the keypoint obj loss gain
    label_smoothing=0.0,  # Experiment with label smoothing
    nbs=64,               # Adjust nominal batch size based on GPU memory
    overlap_mask=True,    # Masks should overlap during training
    mask_ratio=3,         # Experiment with mask downsample ratio
    val=True,             # Enable validation during training
    plots=False,            # Disable saving plots and images during train/val
    patience=30,
    cache=True,
    optimizer='auto',
    single_cls = True
)

end_time = time.time()
elapsed_time = end_time - start_time
hours, remainder = divmod(elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Time script took to run = {int(hours)} hours, {int(minutes)} minutes, and {int(seconds)} seconds.")
