import os
import cv2
import csv
import shutil
import time 
import numpy as np
import pandas as pd
import random
from pathlib import Path
from tqdm import tqdm

# --- Utility Functions ---
def parse_bbox(row):
    try:
        corners = []
        for key in ["Top Left", "Bottom Left", "Bottom Right", "Top Right"]:
            val = row[key].strip("()")
            x_str, y_str = val.split(",")
            corners.append((int(float(x_str)), int(float(y_str))))
        return corners
    except:
        return None

def get_bbox_center(bbox):
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return (sum(xs) // 4, sum(ys) // 4)

def bbox_to_rect(bbox):
    x1 = min([p[0] for p in bbox])
    y1 = min([p[1] for p in bbox])
    x2 = max([p[0] for p in bbox])
    y2 = max([p[1] for p in bbox])
    return x1, y1, x2, y2

def mask_seal_area(image, bbox, mask_color=(200, 200, 200)):
    x1, y1, x2, y2 = bbox_to_rect(bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), mask_color, thickness=cv2.FILLED)
    return (x2 - x1) * (y2 - y1)

def trace_ice_contours(image, lower_thresh, upper_thresh, kernel_size):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.inRange(img_rgb, lower_thresh, upper_thresh)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# --- Main Function ---
def analyze_seal_ice(csv_path, image_dir, output_dir=None, save_images=True,
                     lower_thresh=(150, 150, 150), upper_thresh=(245, 245, 245), kernel_size=2, enable_sampling=False, sample_size=10):
    df = pd.read_csv(csv_path)
    df_grouped = df.groupby("Image")
    start_time = time.time()

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(csv_path))
    base_output = Path(output_dir)
    analysis_dir = base_output / "Seal_Ice_Analysis"

    if analysis_dir.exists():
        shutil.rmtree(analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    seal_output = []
    ice_output = []

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for image_name in tqdm(image_files, desc="Seal Ice Analysis"):
        image_path = Path(image_dir) / image_name
        if not image_path.exists():
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        image_masked = image.copy()
        bboxes = []
        seal_areas = []

        # Only process seals if image is in CSV
        if image_name in df_grouped.groups:
            group = df_grouped.get_group(image_name)
            for _, row in group.iterrows():
                bbox = parse_bbox(row)
                if bbox:
                    area = mask_seal_area(image_masked, bbox)
                    bboxes.append((bbox, row['Latitude'], row['Longitude'], row['Confidence'], area))
                    seal_areas.append(area)

        contours = trace_ice_contours(image_masked, np.array(lower_thresh), np.array(upper_thresh), kernel_size)
        ice_data = []

        for contour in contours:
            area = cv2.contourArea(contour)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                ice_data.append(((cX, cY), area, contour))

        # Remove ice chunks that match seal areas
        valid_ice_data = []
        for (cX, cY), area, contour in ice_data:
            if not any(abs(area - seal_area) < 100 for seal_area in seal_areas):
                valid_ice_data.append(((cX, cY), area, contour))

        if save_images:
            display_img = image.copy()
            for bbox, _, _, _, _ in bboxes:
                x1, y1, x2, y2 = bbox_to_rect(bbox)
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.drawContours(display_img, [contour for (_, _, contour) in valid_ice_data], -1, (255, 0, 0), 2)
            cv2.imwrite(str(analysis_dir / image_name), display_img)

        # Per-seal CSV
        for bbox, lat, lon, conf, bbox_area in bboxes:
            center = get_bbox_center(bbox)
            matched_area = 0
            for (cX, cY), area, contour in valid_ice_data:
                if cv2.pointPolygonTest(contour, center, False) >= 0:
                    matched_area = area
                    break
            top_left, bottom_left, bottom_right, top_right = bbox
            seal_output.append([
                image_name, lat, lon, conf,
                f"({top_left[0]},{top_left[1]})",
                f"({bottom_left[0]},{bottom_left[1]})",
                f"({bottom_right[0]},{bottom_right[1]})",
                f"({top_right[0]},{top_right[1]})",
                matched_area
            ])

        # Ice chunk CSV
        per_image_ice = [image_name]
        for (cX, cY), area, _ in valid_ice_data:
            per_image_ice.extend([(cX, cY), area])
        if len(per_image_ice) > 1:
            ice_output.append(per_image_ice)

    # Save CSVs
    seal_csv = base_output / "seal_ice_chunk_analysis.csv"
    with open(seal_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Image", "Latitude", "Longitude", "Confidence",
            "Top Left", "Bottom Left", "Bottom Right", "Top Right",
            "Ice Chunk Area"
        ])
        writer.writerows(seal_output)

    ice_csv = base_output / "ice_chunk_summary.csv"
    with open(ice_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ["Image"] + [f"Ice Chunk {i} Center / Area" for i in range((max(len(r)-1 for r in ice_output))//2)]
        writer.writerow(header)
        writer.writerows(ice_output)

    if enable_sampling:
        perform_data_sampling(seal_csv, analysis_dir, image_dir, sample_size)

    print(f"Seal analysis saved to: {seal_csv}")
    print(f"Ice chunk summary saved to: {ice_csv}")
    
    end_time = time.time()
    elapsed = end_time - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Completed Seal Ice Analysis in {int(hours)}h {int(minutes)}m {int(seconds)}s.")

    return analysis_dir, seal_csv

# --- Data Sampling Function ---
def perform_data_sampling(seal_csv, analysis_dir, image_dir, sample_size):
    base_output = Path(analysis_dir).parent
    sampling_dir = base_output / "Data_Sampling"
    if sampling_dir.exists():
        shutil.rmtree(sampling_dir)
    sampling_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(seal_csv)
    seal_counts = df.groupby("Image").size().to_dict()

    all_images = [f for f in os.listdir(analysis_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    # Build full image list with 0 seal fallback
    full_df = pd.DataFrame({
        "Image": all_images,
        "Seal_Count": [seal_counts.get(img, 0) for img in all_images]
    })

    sampled = full_df.sample(
        n=min(sample_size, len(full_df)),
        replace=False,
        random_state=None
    )

    for image_name in sampled["Image"]:
        src = analysis_dir / image_name
        dst = sampling_dir / image_name
        if src.exists():
            shutil.copy(src, dst)

    sampled_csv = sampling_dir / "sampled_data.csv"
    sampled[["Image", "Seal_Count"]].to_csv(sampled_csv, index=False)

    print(f"Data sampling completed. Results saved to: {sampled_csv}")


if __name__ == "__main__":
    # --- Set Parameters Here ---
    IMAGE_DIR = "Sample_Images"
    CSV_PATH = "Sample_Images/REPROC/2025_08_26_11_48_CONF_60/detections.csv"
    OUTPUT_DIR = None
    SAVE_IMAGES = True
    LOWER_THRESH = (150, 150, 150)
    UPPER_THRESH = (245, 245, 245)
    KERNEL_SIZE = 2

    # --- Data Sampling Config ---
    ENABLE_SAMPLING = True
    SAMPLE_SIZE = 5

    analysis_dir, seal_csv = analyze_seal_ice(
        csv_path=CSV_PATH,
        image_dir=IMAGE_DIR,
        output_dir=OUTPUT_DIR,
        save_images=SAVE_IMAGES,
        lower_thresh=LOWER_THRESH,
        upper_thresh=UPPER_THRESH,
        kernel_size=KERNEL_SIZE,
        enable_sampling=ENABLE_SAMPLING,
        sample_size=SAMPLE_SIZE
    )
