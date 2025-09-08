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
MODEL_PATH = r"C:\Users\sa553\Desktop\NPS\NPS_GlacierBay_Seal_Detections\Models\seal-segmentation-v2-1\weights\best.pt"
CLASSES = ["seal"]
TILE_SIZE = 640
CONF_MIN = 0.2
CONF_MAX = 1.0
# -------------------------------------------- #

def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)

def create_directory(directory, clear=False):
    if clear:
        clear_directory(directory)
    os.makedirs(directory, exist_ok=True)

def is_valid_image(file_path):
    try:
        img = Image.open(file_path)
        img.verify()
        return img.size[0] > 0 and img.size[1] > 0
    except Exception as e:
        print(f"[WARNING] Invalid image skipped: {file_path} ({e})")
        return False

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

        if ext.lower() in [".jpg", ".jpeg"]:
            crop.save(tile_path, quality=100, subsampling=0)
        else:
            crop.save(tile_path)

        tile_paths.append(tile_path)

    return tile_paths

def process_dataset(image_dir, batch_name, dry_run=False):
    start_time = time.time()

    crops_dir = os.path.join(image_dir, "Crops")
    logger_dir = os.path.join(image_dir, "Logger")
    csv_log_path = os.path.join(logger_dir, "detections_log.csv")

    create_directory(crops_dir, clear=True)
    create_directory(logger_dir, clear=True)

    with open(csv_log_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["image", "x1", "y1", "x2", "y2", "confidence"])

    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    project = rf.workspace("waffles").project("glacier-bay-harbor-seals")
    model = YOLO(MODEL_PATH)

    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith("._")
    ]

    if not image_files:
        print(f"[INFO] No image files found in {image_dir}. Exiting.")
        return

    for filename in tqdm(image_files, desc=f"Tiling + Inferring ({batch_name})"):
        img_path = os.path.join(image_dir, filename)
        if not is_valid_image(img_path):
            continue

        tile_paths = tile_image_to_dir(filename, image_dir, crops_dir, TILE_SIZE)

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

                with open(csv_log_path, mode='a', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow([os.path.basename(tile_path), x1, y1, x2, y2, round(conf, 3)])

            out_img_path = os.path.join(logger_dir, os.path.basename(tile_path))
            if out_img_path.lower().endswith((".jpg", ".jpeg")):
                cv2.imwrite(out_img_path, annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            else:
                cv2.imwrite(out_img_path, annotated)

            if not dry_run:
                project.upload(
                    image_path=tile_path,
                    batch_name=batch_name,
                    is_prediction=True
                )

            os.remove(tile_path)

    clear_directory(crops_dir)

    elapsed = time.time() - start_time
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    print(f"\n[INFO] {batch_name} complete. Time: {int(h)}h {int(m)}m {int(s)}s")
    print(f"[INFO] Results saved to {logger_dir} and log at {csv_log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tile images, run YOLO inference, and upload predictions.")
    parser.add_argument("--dry-run", action="store_true", help="Do not upload predictions to Roboflow.")
    args = parser.parse_args()

    IMAGE_DIR_1 = r"Z:\Projects\Clients\NPS_GlacierBay\2023\WingtraPilotProjects\230621_Data\JHI_Maiden1 Flight 01\OUTPUT"
    BATCH_NAME_1 = "230621_Data_JHI_Maiden1 Flight 01"

    IMAGE_DIR_2 = r"Z:\Projects\Clients\NPS_GlacierBay\2023\WingtraPilotProjects\230623_Data\JHI_FullSurvey_75m_survey5 Flight 01\OUTPUT"
    BATCH_NAME_2 = "230623_Data_JHI_FullSurvey_75m_survey5 Flight 01"

    # process_dataset(IMAGE_DIR_1, BATCH_NAME_1, dry_run=args.dry_run) #449
    process_dataset(IMAGE_DIR_2, BATCH_NAME_2, dry_run=args.dry_run) 
