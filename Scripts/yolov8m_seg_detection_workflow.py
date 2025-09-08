import os
import cv2
import torch
import shutil
import datetime
import time
import numpy as np
import pandas as pd
import uuid
import multiprocessing
from PIL import Image, ExifTags
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
from functools import partial
import imageio.v3 as iio

def create_output_folders(base_dir, output_dir, draw, conf_threshold):
    if output_dir is None:
        output_dir = base_dir
    reproc_dir = os.path.join(output_dir, "REPROC")
    target_base = reproc_dir if os.path.isdir(reproc_dir) else output_dir
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    timestamp_conf = f"{timestamp}_CONF_{str(int(conf_threshold*100))}"
    run_dir = os.path.join(target_base, timestamp_conf)
    tile_dir = os.path.join(run_dir, "TILES")
    os.makedirs(tile_dir, exist_ok=True)

    detect_dir = None
    if draw:
        detect_dir = os.path.join(run_dir, "Detect_Images")
        os.makedirs(detect_dir, exist_ok=True)

    return run_dir, detect_dir, tile_dir

def is_valid_image(file_path):
    try:
        img = Image.open(file_path)
        img.verify()
        return img.size[0] > 0 and img.size[1] > 0
    except:
        return False

def tile_image(image_path, tile_size=(640, 640), overlap=(320, 320)):
    img = cv2.imread(image_path)
    tiles = []
    h, w, _ = img.shape
    stride_x, stride_y = tile_size[0] - overlap[0], tile_size[1] - overlap[1]

    for y in range(0, h, stride_y):
        for x in range(0, w, stride_x):
            tile = np.zeros((tile_size[1], tile_size[0], 3), dtype=img.dtype)
            img_tile = img[y:min(y + tile_size[1], h), x:min(x + tile_size[0], w)]
            tile[0:img_tile.shape[0], 0:img_tile.shape[1]] = img_tile
            tiles.append(((x, y), tile))

    return tiles, img.shape

def compute_overlap_metrics(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = areaA + areaB - interArea

    iou = interArea / unionArea if unionArea else 0
    containment = interArea / min(areaA, areaB) if min(areaA, areaB) else 0

    return iou, containment, areaA, areaB

def merge_detections(detections, iou_thresh=0.6, containment_thresh=0.92):
    detections = sorted(detections, key=lambda d: (
        (d['bbox'][2] - d['bbox'][0]) * (d['bbox'][3] - d['bbox'][1])
    ), reverse=True)

    merged = []

    while detections:
        current = detections.pop(0)
        current_box = current['bbox']
        to_remove = []

        for other in detections:
            iou, containment, areaA, areaB = compute_overlap_metrics(current_box, other['bbox'])
            if iou > iou_thresh or containment > containment_thresh:
                to_remove.append(other)

        for r in to_remove:
            if r in detections:
                detections.remove(r)

        merged.append(current)

    return merged

def draw_detections(image, detections):
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return image

def preserve_exif_and_copy(src, dst):
    shutil.copy2(src, dst)

def convert_to_degrees(value):
    d = float(value[0])
    m = float(value[1])
    s = float(value[2])
    return d + (m / 60.0) + (s / 3600.0)

def get_lat_lon(image_path):
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        if exif_data is not None:
            gps_info = None
            for tag, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                if tag_name == "GPSInfo":
                    gps_info = value
                    break
            if gps_info:
                lat_data = gps_info[2]
                lon_data = gps_info[4]
                lat_ref = gps_info[1]
                lon_ref = gps_info[3]

                latitude = convert_to_degrees(lat_data)
                if lat_ref != 'N':
                    latitude = -latitude
                longitude = convert_to_degrees(lon_data)
                if lon_ref != 'E':
                    longitude = -longitude
                return latitude, longitude
        return None, None
    except Exception as e:
        print(f"Error extracting GPS data from {image_path}: {e}")
        return None, None

def save_image(path, array, exif_data=None):
    """Always save as high-quality JPEG (virtually lossless), preserves EXIF if available."""
    if exif_data:
        iio.imwrite(
            str(path.with_suffix(".jpg")),
            array,
            quality=100,
            chroma_subsampling=False,
            exif=exif_data
        )
    else:
        iio.imwrite(
            str(path.with_suffix(".jpg")),
            array,
            quality=100,
            chroma_subsampling=False
        )

def process_single_image(img_path, model_path, conf_threshold, draw, run_dir, detect_dir):
    img_name = os.path.basename(img_path)
    tile_dir = os.path.join(run_dir, "TILES", f"{Path(img_name).stem}_{uuid.uuid4().hex[:8]}")
    os.makedirs(tile_dir, exist_ok=True)

    model = YOLO(model_path)
    detections_csv = []

    if not is_valid_image(img_path):
        return []

    latitude, longitude = get_lat_lon(img_path)
    tiles, _ = tile_image(img_path)
    detections = []

    for idx, ((x_off, y_off), tile) in enumerate(tiles):
        tile_path = Path(tile_dir) / f"{Path(img_name).stem}_tile_{idx}"
        save_image(tile_path, tile)  # always JPG
        results = model(str(tile_path.with_suffix(".jpg")), verbose=False)[0]

        for box in results.boxes:
            if box.conf < conf_threshold:
                continue
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            full_coords = [x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off]

            detections.append({
                'image': img_name,
                'confidence': float(box.conf),
                'bbox': full_coords,
                'corners': [
                    (x1 + x_off, y1 + y_off),
                    (x1 + x_off, y2 + y_off),
                    (x2 + x_off, y2 + y_off),
                    (x2 + x_off, y1 + y_off),
                ]
            })

    unique_detections = merge_detections(detections)

    if draw and detect_dir:
        try:
            exif_data = Image.open(img_path).info.get("exif")
        except:
            exif_data = None
        img_draw = cv2.imread(img_path)
        if unique_detections:
            img_draw = draw_detections(img_draw, unique_detections)
        save_image(Path(detect_dir) / Path(img_name).stem, img_draw, exif_data)

    for det in unique_detections:
        row = [det['image'], latitude, longitude, det['confidence']] + [f"({x},{y})" for (x, y) in det['corners']]
        detections_csv.append(row)

    shutil.rmtree(tile_dir)
    return detections_csv

def process_images(img_dir, model_dir, conf_threshold, draw=True, output_dir=None):
    start_time = time.time()
    Image.MAX_IMAGE_PIXELS = None

    run_dir, detect_dir, _ = create_output_folders(img_dir, output_dir, draw, conf_threshold)

    image_paths = [
        os.path.join(img_dir, fname)
        for fname in os.listdir(img_dir)
        if fname.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print("Running in single-threaded mode...")

    detections_csv = []
    for img_path in tqdm(image_paths, desc="Detection Workflow"):
        results = process_single_image(
            img_path,
            model_path=model_dir,
            conf_threshold=conf_threshold,
            draw=draw,
            run_dir=run_dir,
            detect_dir=detect_dir
        )
        detections_csv.extend(results)

    csv_path = os.path.join(run_dir, "detections.csv")
    df = pd.DataFrame(detections_csv, columns=[
        "Image", "Latitude", "Longitude", "Confidence",
        "Top Left", "Bottom Left", "Bottom Right", "Top Right"
    ])
    df.to_csv(csv_path, index=False)

    print(f"Seal detections saved to: {csv_path}")

    tiles_root_dir = os.path.join(run_dir, "TILES")
    if os.path.exists(tiles_root_dir):
        shutil.rmtree(tiles_root_dir)

    elapsed = time.time() - start_time
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    print(f"Completed Detection Workflow in {int(h)}h {int(m)}m {int(s)}s.")

    return csv_path

if __name__ == "__main__":
    image_dir = "Sample_Images/"
    model_path = "Models/seal-segmentation-v2-1/weights/best.pt"
    output_dir = None
    conf_threshold = 0.65
    draw = True

    csv_file = process_images(image_dir, model_path, conf_threshold, draw, output_dir)
    print(f"Detections CSV saved to: {csv_file}")
