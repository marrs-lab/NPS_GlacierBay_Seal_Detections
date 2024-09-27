# -*- coding: utf-8 -*-
"""
   @author: Saagar
   
   Workflow for passing in a directory of pictures, creating crops,
   running the models, and logging results. No recombination of crops.
"""

import os
import shutil
import time
from PIL import Image, ImageOps
from ultralytics import YOLO
from itertools import product
from pathlib import Path


# Directory setup and cleanup functions
def clear_directory(directory):
    if os.path.exists(directory):
        for file_name in os.listdir(directory):
            file_path = os.path.join(directory, file_name)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Error: {e}")

def create_directory(directory, clear=False):
    if clear:
        clear_directory(directory)
    os.makedirs(directory, exist_ok=True)

# Image manipulation functions
def tile(filename, directory, crops_directory, d=640):
    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(directory, filename))

    # Check image orientation and rotate if necessary
    if hasattr(img, '_getexif'):
        exif = img._getexif()
        orientation = exif.get(274) if exif else None

        if orientation == 3:  # Rotate 180 degrees
            img = img.rotate(180, expand=True)
        elif orientation == 6:  # Rotate 270 degrees clockwise
            img = img.rotate(270, expand=True)
        elif orientation == 8:  # Rotate 90 degrees clockwise
            img = img.rotate(90, expand=True)

    w, h = img.size
    grid = product(range(0, h, d), range(0, w, d))
    for j, i in grid:
        box = (i, j, i + d, j + d)
        out = os.path.join(crops_directory, f'{name}_{j}_{i}{ext}')
        img.crop(box).save(out)

def invert_and_save_directory(input_directory, output_directory):
    for filename in os.listdir(input_directory):
        if filename.lower().endswith('.jpg') or filename.endswith(".JPG"):
            original_image = Image.open(os.path.join(input_directory, filename))
            inverted_image = ImageOps.invert(original_image)
            inverted_image.save(os.path.join(output_directory, filename), format='JPEG')

# Model functions
def run_model_on_crop(model, crop_path, confidence):
    results = model.predict(crop_path, conf=confidence, show_labels=False, show_conf=True, show_boxes=False)
    return results

def boxes_overlap(box1, box2, overlap_threshold=0.9):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate overlap area
    overlap_x1 = max(x1_1, x1_2)
    overlap_y1 = max(y1_1, y1_2)
    overlap_x2 = min(x2_1, x2_2)
    overlap_y2 = min(y2_1, y2_2)

    overlap_width = max(0, overlap_x2 - overlap_x1)
    overlap_height = max(0, overlap_y2 - overlap_y1)

    overlap_area = overlap_width * overlap_height

    # Calculate box areas
    area_box1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area_box2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Calculate overlap ratio
    overlap_ratio = overlap_area / min(area_box1, area_box2)

    return overlap_ratio > overlap_threshold

def combine_results(resultOne, resultTwo, resultThree, resultFour, overlap_threshold=0.9):
    boxesOne = resultOne[0].boxes.xyxy.cpu().numpy()
    boxesTwo = resultTwo[0].boxes.xyxy.cpu().numpy()
    boxesThree = resultThree[0].boxes.xyxy.cpu().numpy()
    boxesFour = resultFour[0].boxes.xyxy.cpu().numpy()

    unique_boxes = []
    
    for box_set in [boxesOne, boxesTwo, boxesThree, boxesFour]:
        for box in box_set:
            overlaps = any(boxes_overlap(box, existing_box['boxes'], overlap_threshold) for existing_box in unique_boxes)
            if not overlaps:
                unique_boxes.append({'boxes': box})
    
    return unique_boxes

def analyze_crop(filename, models, confidence, crops_directory_color, crops_directory_inverse, log_path):
    inverseImagePath = os.path.join(crops_directory_inverse, filename)
    colorImagePath = os.path.join(crops_directory_color, filename)

    resultCombinedInverse = run_model_on_crop(models[0], inverseImagePath, confidence)
    resultCombinedColor = run_model_on_crop(models[1], colorImagePath, confidence)
    resultInverse = run_model_on_crop(models[2], inverseImagePath, confidence)
    resultColor = run_model_on_crop(models[3], colorImagePath, confidence)

    combined_results = combine_results(resultCombinedInverse, resultCombinedColor, resultInverse, resultColor)

    # Extract crop coordinates from filename (similar to script 1)
    base_name = Path(filename).stem
    crop_x, crop_y = map(int, base_name.split('_')[-2:])

    # Open the image to check for orientation
    image = Image.open(colorImagePath)
    if hasattr(image, '_getexif'):
        exif = image._getexif()
        orientation = exif.get(274) if exif else None
    else:
        orientation = None

    # Log results to file with adjusted coordinates
    with open(log_path, 'a') as log:
        for box in combined_results:
            x1, y1, x2, y2 = box['boxes']

            # Adjust coordinates based on orientation (similar to script 1)
            if orientation == 3:  # Rotated 180 degrees
                x1, x2 = image.width - x2, image.width - x1
                y1, y2 = image.height - y2, image.height - y1
            elif orientation == 6:  # Rotated 270 degrees clockwise
                x1, y1, x2, y2 = y1, image.width - x2, y2, image.width - x1
            elif orientation == 8:  # Rotated 90 degrees clockwise
                x1, y1, x2, y2 = image.height - y2, x1, image.height - y1, x2

            # Adjust based on crop coordinates
            x1_adjusted = x1 + crop_y
            y1_adjusted = y1 + crop_x
            x2_adjusted = x2 + crop_y
            y2_adjusted = y2 + crop_x

            # Log the adjusted coordinates
            log.write(f", {x1_adjusted} {y1_adjusted} {x2_adjusted} {y2_adjusted}")
    
    return len(combined_results)


# Main processing function
def process_raw_directory(raw_directory, confidence, legacy_folder_n, models):
    recom_directory = os.path.join(raw_directory, "Recomb")
    create_directory(recom_directory, False)
    
    existing_legacy_folders = [folder for folder in os.listdir(recom_directory) if folder.startswith(legacy_folder_n)]
    legacy_number = len(existing_legacy_folders) + 1
    
    log_path = os.path.join(recom_directory, f"{legacy_folder_n} ({legacy_number}) ({confidence}) LOG.txt")
    
    crops_directory = os.path.join(raw_directory, "Crops")
    create_directory(crops_directory, True)
    crops_directory_color = os.path.join(crops_directory, "ColorCrops")
    create_directory(crops_directory_color, True)
    crops_directory_inverse = os.path.join(crops_directory, "InverseCrops")
    create_directory(crops_directory_inverse, True)

    print("Crop Directories Set Up")

    imageList = {}
    for filename in os.listdir(raw_directory):
        if os.path.isfile(os.path.join(raw_directory, filename)) and (filename.lower().endswith('.jpg') or filename.endswith(".JPG")):
            imageList[filename] = 0

            clear_directory(crops_directory_color)
            clear_directory(crops_directory_inverse)
            with open(log_path, 'a') as log:
                log.write("\n"+filename)
            tile(filename, raw_directory, crops_directory_color)
            invert_and_save_directory(crops_directory_color, crops_directory_inverse)

            for file in os.listdir(crops_directory_color):
                if os.path.isfile(os.path.join(crops_directory_color, file)) and (file.lower().endswith('.jpg') or file.endswith(".JPG")):
                    seals = analyze_crop(file, models, confidence, crops_directory_color, crops_directory_inverse, log_path)
                    imageList[filename] += seals

    return log_path

def run_workflow(combinedPath, colorPath, inversePath, image_directory_path, conf):
    Image.MAX_IMAGE_PIXELS = None
    start_time = time.time()
    
    combinedInverse = YOLO(combinedPath)
    combinedColor = YOLO(combinedPath)
    inverse = YOLO(inversePath)
    color = YOLO(colorPath)
    models = [combinedInverse, combinedColor, inverse, color]
    
    confidence = conf
    log_path = process_raw_directory(image_directory_path, confidence, "FinalFlightAnalysis", models)

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Time script took to run = {int(hours)} hours, {int(minutes)} minutes, and {int(seconds)} seconds.")
    return log_path

if __name__ == '__main__':
    combinedPath = r"C:\Users\sa553\Desktop\NPS\Models\AugmentAndInverse\ColorAndInverseV9\weights\best.pt"
    colorPath = r"C:\Users\sa553\Desktop\NPS\Models\Color\ColorV9\weights\best.pt"
    inversePath = r"C:\Users\sa553\Desktop\NPS\Models\Inverse\InverseV9\weights\best.pt"
    parent_directory = r"C:\Users\sa553\Desktop\NPS\JHI_FullSurvey_75m_Survey1 Flight 01\IMAGES"

    run_workflow(combinedPath, colorPath, inversePath, parent_directory, .85)
