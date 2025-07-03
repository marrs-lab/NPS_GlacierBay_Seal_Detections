import os
import shutil
import time
import csv
from itertools import product
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm
import argparse
from ultralytics import YOLO
from roboflow import Roboflow
import supervision as sv

# -------------------------------------------- #
IMAGE_DIR = r"C:\Users\sa553\Desktop\NPS\NPS_GlacierBay_Seal_Detections\EVAL_JHI_FullSurvey_75m_Survey1 Flight 01\Batch 4"
CROPS_DIR = os.path.join(IMAGE_DIR, "Crops")
LOGGER_DIR = os.path.join(IMAGE_DIR, "Logger")
MODEL_PATH = r"C:\Users\sa553\Desktop\NPS\NPS_GlacierBay_Seal_Detections\Models\seal-segmentation-v2-1\weights\best.pt"
CLASSES = ["seal"]
TILE_SIZE = 640
CONF_MIN = 0.2
CONF_MAX = 0.85
CSV_LOG_PATH = os.path.join(LOGGER_DIR, "detections_log.csv")
# -------------------------------------------- #

def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)

def create_directory(directory, clear=False):
    if clear:
        clear_directory(directory)
    os.makedirs(directory, exist_ok=True)

def tile_image_to_dir(filename, input_dir, output_dir, tile_size=640):
    name, ext = os.path.splitext(filename)
    img_path = os.path.join(input_dir, filename)
    img = Image.open(img_path)

    try:
        exif = img._getexif()
        orientation = exif.get(274) if exif else None
        if orientation == 3:
            img = img.rotate(180, expand=True)
        elif orientation == 6:
            img = img.rotate(270, expand=True)
        elif orientation == 8:
            img = img.rotate(90, expand=True)
    except:
        pass

    w, h = img.size
    tile_paths = []
    for y, x in product(range(0, h, tile_size), range(0, w, tile_size)):
        box = (x, y, x + tile_size, y + tile_size)
        crop = img.crop(box)
        tile_filename = f"{name}_{y}_{x}{ext}"
        tile_path = os.path.join(output_dir, tile_filename)
        crop.save(tile_path)
        tile_paths.append(tile_path)

    return tile_paths

def main(dry_run=False):
    start_time = time.time()

    create_directory(CROPS_DIR, clear=True)
    create_directory(LOGGER_DIR, clear=True)

    # Initialize CSV log
    with open(CSV_LOG_PATH, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["image", "x1", "y1", "x2", "y2", "confidence"])

    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    project = rf.workspace("waffles").project("glacier-bay-harbor-seals")
    model = YOLO(MODEL_PATH)

    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print(f"[INFO] No image files found in {IMAGE_DIR}. Exiting.")
        return

    for filename in tqdm(image_files, desc="Tiling and Inferring"):
        tile_paths = tile_image_to_dir(filename, IMAGE_DIR, CROPS_DIR, TILE_SIZE)

        for tile_path in tile_paths:
            image = cv2.imread(tile_path)
            if image is None:
                print(f"[WARNING] Failed to load tile {tile_path}, skipping.")
                continue

            results = model(image, conf=0.1)[0]
            detections = sv.Detections.from_ultralytics(results)

            filtered = [
                (conf, box) for conf, box in zip(detections.confidence, detections.xyxy)
                if CONF_MIN <= conf <= CONF_MAX
            ]

            if not filtered:
                os.remove(tile_path)
                continue

            annotated = image.copy()
            for conf, box in filtered:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"seal {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Log to CSV
                with open(CSV_LOG_PATH, mode='a', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow([os.path.basename(tile_path), x1, y1, x2, y2, round(conf, 3)])

            out_img_path = os.path.join(LOGGER_DIR, os.path.basename(tile_path))
            cv2.imwrite(out_img_path, annotated)

            if not dry_run:
                project.upload(
                    image_path=tile_path,
                    batch_name="Batch 4 (0.2-0.85) - EVAL_JHI_FullSurvey_75m_Survey1 Flight 01",
                    is_prediction=True
                )

            os.remove(tile_path)

    clear_directory(CROPS_DIR)

    elapsed = time.time() - start_time
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    print(f"\n[INFO] Processing complete. Elapsed time: {int(h)}h {int(m)}m {int(s)}s")
    print(f"[INFO] Results saved to LOGGER_DIR and detection log at {CSV_LOG_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tile images, run YOLO inference, and upload predictions.")
    parser.add_argument("--dry-run", action="store_true", help="Do not upload predictions to Roboflow.")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
