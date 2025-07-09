import os
import cv2
import shutil
import time 
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def is_valid_image(file_path):
    """Check if image is valid and has non-zero dimensions."""
    try:
        img = Image.open(file_path)
        img.verify()
        return img.size[0] > 0 and img.size[1] > 0
    except:
        return False

def draw_detections(image, boxes):
    """Draw bounding boxes on the image."""
    for box in boxes:
        x1, y1 = box[0]
        x2, y2 = box[2]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return image

def read_csv_and_draw(image_dir, csv_path, output_dir = None):
    """Read detection CSV and annotate images with bounding boxes."""
    start_time = time.time()
    Image.MAX_IMAGE_PIXELS = None

    df = pd.read_csv(csv_path)

    # Create output directory for annotated images
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(csv_path), "Detect_Images")
    else:
        output_dir = os.path.join(output_dir, "Detect_Images")
    os.makedirs(output_dir, exist_ok=True)

    grouped = df.groupby("Image")

    for image_name, group in tqdm(grouped, desc="Draw Detections"):
        src_image_path = os.path.join(image_dir, image_name)
        dst_image_path = os.path.join(output_dir, image_name)

        if not os.path.exists(src_image_path) or not is_valid_image(src_image_path):
            print(f"Skipping invalid or missing image: {src_image_path}")
            continue

        # Copy original image with metadata
        shutil.copy2(src_image_path, dst_image_path)

        # Load and draw detections
        img = cv2.imread(dst_image_path)
        boxes = []

        for _, row in group.iterrows():
            try:
                corners = []
                for key in ["Top Left", "Bottom Left", "Bottom Right", "Top Right"]:
                    val = row[key].strip("()")
                    x_str, y_str = val.split(",")
                    corners.append((int(x_str), int(y_str)))
                boxes.append(corners)
            except Exception as e:
                print(f"Error parsing corners for {image_name}: {e}")


        if boxes:
            img = draw_detections(img, boxes)
            cv2.imwrite(dst_image_path, img)

    print(f"Annotated images saved to: {output_dir}")

    end_time = time.time()
    elapsed = end_time - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Completed Draw Detections in {int(hours)}h {int(minutes)}m {int(seconds)}s.")

if __name__ == "__main__":
    image_dir = "Sample_Images/"
    csv_path = "Sample_Images/REPROC/2025_07_09_00_00_CONF_80/detections.csv"
    output_dir = None
    read_csv_and_draw(image_dir, csv_path, output_dir)
