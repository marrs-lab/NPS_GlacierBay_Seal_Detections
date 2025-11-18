# Glacier Bay Seal Ice Preference Analysis

This repository provides a full workflow for detecting seals in drone imagery from Glacier Bay National Park, drawing detection results, and analyzing seal-ice relationships. The system uses YOLOv8 models for object detection.

## Repository Structure

### Core Files
- **`Full_System_Run.py`** — Main orchestration script that runs the complete detection and analysis pipeline with multiprocessing support
- **`requirements.txt`** — Python package dependencies
- **`LICENSE-CC-BY-SA.md`** — Creative Commons Attribution-ShareAlike license

### Directories

#### `Models/`
Contains trained YOLO models:
- **`seal-box-v3-1/`** — Most recently trained bounding box detection model
  - `weights/best.pt` — Best performing model checkpoint
  - `weights/last.pt` — Latest training checkpoint
  - `args.yaml` — Training configuration
  - `results.csv` — Training metrics

#### `Roboflow_V3/`
Bounding box training dataset from Roboflow:
- **`data.yaml`** — Dataset configuration file
- **`train/`**, **`valid/`**, **`test/`** — Split datasets with images and YOLO format labels
- `README.dataset.txt`, `README.roboflow.txt` — Dataset documentation

#### `Sample_Images/`
Sample drone imagery for testing:
- Contains example images
- **`REPROC/`** — Output folder created by processing runs
  - Each run creates a timestamped subfolder (e.g., `2025_09_10_09_42_CONF_80/`)
  - Contains detection CSVs, analysis results, and annotated images

#### `Scripts/`
Core processing modules:
- **`yolov8m_seg_detection_workflow.py`** — Image tiling, YOLO detection, duplicate merging, and CSV export
- **`draw_detections.py`** — Visualizes bounding boxes on images from detection CSVs
- **`seal_ice_analysis.py`** — Analyzes seal-ice spatial relationships, generates ice segmentation
- **`Training_Tools/`** — Model training and evaluation scripts
  - `yolov8m-seg_train.py` — Training script
  - `yolov8m-seg_tune.py` — Hyperparameter tuning
  - `Roboflow_eval.py` — Model evaluation on Roboflow dataset

## Workflow Overview

The system follows a three-stage pipeline:

### 1. Detection Workflow (`yolov8m_seg_detection_workflow.py`)
- **Input**: Directory of high-resolution drone images
- **Process**:
  - Tiles large images with overlap (default: 640×640 tiles, 320×320 overlap)
  - Runs YOLO detection on each tile
  - Merges duplicate detections across tile boundaries using IOU and containment metrics
  - Extracts GPS coordinates from EXIF data
- **Output**: `detections.csv` with seal locations, confidence scores, and bounding box coordinates

### 2. Draw Detections (`draw_detections.py`) — Optional
- **Input**: `detections.csv` + original images
- **Process**: Draws red bounding boxes on images at detected seal locations
- **Output**: Annotated images in `Detect_Images/` folder

### 3. Seal-Ice Analysis (`seal_ice_analysis.py`)
- **Input**: `detections.csv` + original images
- **Process**:
  - Masks out seal bounding boxes to prevent false ice detection
  - Segments ice using RGB thresholding and morphological operations
  - Traces ice contours and calculates areas
  - Matches each seal to its underlying ice chunk via point-in-polygon test
  - Optionally samples random images for validation
- **Output**:
  - `seal_ice_chunk_analysis.csv` — Per-seal data with ice chunk area
  - `ice_chunk_summary.csv` — All ice chunks detected per image
  - `Seal_Ice_Analysis/` folder — Images with seals and ice contours overlaid
  - `Data_Sampling/` folder — Random sample of annotated images (if enabled)

## How to Run

### Quick Start

1. **Configure parameters** in `Full_System_Run.py`:

```python
IMAGE_DIRECTORY = "Sample_Images/"          # Path to image folder(s)
MODEL_PATH = "Models/seal-box-v3-1/weights/best.pt"  # YOLO model
CONFIDENCE = 0.8                            # Detection confidence threshold (0.0-1.0)
DRAW_SEALS = False                          # Draw bounding boxes only
DRAW_SEALS_ON_ICE = True                    # Draw seals + ice contours
DATA_SAMPLING = True                        # Enable random image sampling
SAMPLING_SIZE = 20                          # Number of images to sample
```

2. **Add image directories** to process (supports multiprocessing):

```python
output_paths = ["Sample_Images/", "Path/To/Other/Images/"]
```

3. **Run the pipeline**:

```bash
python Full_System_Run.py
```

The script automatically allocates worker threads (default: `cpu_count // 2`) and processes all paths in parallel.

## Script Parameters Reference

### `Full_System_Run.py`
**Main orchestration script with multiprocessing**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `IMAGE_DIRECTORY` | str | - | Path to folder containing images to process |
| `MODEL_PATH` | str | `"Models/seal-box-v3-1/weights/best.pt"` | Path to trained YOLO model |
| `CONFIDENCE` | float | `0.8` | Confidence threshold (0.0-1.0) for detections |
| `DRAW_SEALS` | bool | `False` | If True, draws only bounding boxes on images |
| `DRAW_SEALS_ON_ICE` | bool | `True` | If True, draws seals + traces ice contours |
| `DATA_SAMPLING` | bool | `True` | Enable random sampling of output images |
| `SAMPLING_SIZE` | int | `20` | Number of random images to sample |
| `output_paths` | list | `["Sample_Images/"]` | List of image directories to process in parallel |

### `yolov8m_seg_detection_workflow.py`
**Detection workflow with tiling and merging**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `img_dir` | str | - | Input image directory |
| `model_dir` | str | - | Path to YOLO model (.pt file) |
| `output_dir` | str | `None` | Output directory (defaults to `img_dir/REPROC/`) |
| `conf_threshold` | float | `0.8` | Minimum confidence for detections |
| `draw` | bool | `False` | Draw bounding boxes on images |
| `tile_size` | tuple | `(640, 640)` | Tile dimensions in pixels |
| `overlap` | tuple | `(320, 320)` | Overlap between tiles in pixels |
| `iou_thresh` | float | `0.6` | IOU threshold for merging duplicate detections |
| `containment_thresh` | float | `0.9` | Containment threshold for removing nested boxes |

**Process:**
1. Tiles images with overlap to handle high-resolution inputs
2. Runs YOLO detection on each tile
3. Merges overlapping detections across tiles
4. Extracts GPS coordinates from EXIF data
5. Outputs consolidated CSV with all detections

### `draw_detections.py`
**Visualizes bounding boxes from CSV**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_dir` | str | - | Directory containing original images |
| `csv_path` | str | - | Path to `detections.csv` |
| `output_dir` | str | `None` | Output directory (defaults to CSV location) |

**Process:**
- Copies original images (preserving EXIF metadata)
- Draws red bounding boxes at detection coordinates
- Saves annotated images to `Detect_Images/` folder

### `seal_ice_analysis.py`
**Analyzes seal-ice spatial relationships**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csv_path` | str | - | Path to `detections.csv` |
| `image_dir` | str | - | Directory containing original images |
| `output_dir` | str | `None` | Output directory (defaults to CSV location) |
| `save_images` | bool | `True` | Save annotated images with ice contours |
| `lower_thresh` | tuple | `(150, 150, 150)` | Lower RGB threshold for ice detection |
| `upper_thresh` | tuple | `(245, 245, 245)` | Upper RGB threshold for ice detection |
| `kernel_size` | int | `2` | Morphological operation kernel size |
| `enable_sampling` | bool | `False` | Enable random image sampling |
| `sample_size` | int | `10` | Number of images to randomly sample |

**Process:**
1. Masks seal bounding boxes to exclude from ice detection
2. Segments ice using RGB thresholding
3. Applies morphological operations (closing → opening)
4. Finds ice chunk contours and calculates areas
5. Matches each seal to its ice chunk via centroid point-in-polygon test
6. Optionally samples random images weighted by seal count

## Output Files

All outputs are saved in a timestamped folder: `<IMAGE_DIR>/REPROC/YYYYMMDD_HHMM_C##/`

### CSV Files

**`detections.csv`**
- Image filename
- GPS latitude and longitude (from EXIF)
- Detection confidence score
- Bounding box corners (Top Left, Bottom Left, Bottom Right, Top Right)

**`seal_ice_chunk_analysis.csv`**
- All fields from `detections.csv`
- `Ice Chunk Area` — Area in pixels of the ice chunk the seal is on (0 if not on ice)

**`ice_chunk_summary.csv`**
- Image filename
- Columns for each ice chunk: center coordinates (x, y) and area
- Variable number of columns based on max ice chunks in any image

### Image Folders

**`Detect_Images/`** (if `DRAW_SEALS=True`)
- Original images with red bounding boxes drawn at seal locations

**`Seal_Ice_Analysis/`** (if `DRAW_SEALS_ON_ICE=True`)
- Images with red bounding boxes (seals) and blue contours (ice chunks)

**`Data_Sampling/`** (if `DATA_SAMPLING=True`)
- Random sample of images from `Seal_Ice_Analysis/`
- `sampled_data.csv` — Image filenames and seal counts

**`TILES/`** (temporary, auto-deleted)
- Intermediate tile images created during detection

## Requirements

- Python 3.8+
- Required packages: `opencv-python`, `torch`, `ultralytics`, `pandas`, `numpy`, `Pillow`, `tqdm`

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Key Features

- **Multiprocessing**: Processes multiple image directories in parallel
- **Image Tiling**: Handles high-resolution drone imagery with configurable tile size and overlap
- **Smart Detection Merging**: Uses IOU and containment metrics to remove duplicates across tile boundaries
- **GPS Integration**: Extracts and preserves GPS EXIF data from source images
- **Ice Segmentation**: RGB-based ice detection with morphological filtering
- **Seal-Ice Matching**: Spatial analysis to determine which ice chunk each seal occupies
- **Data Sampling**: Random sampling for validation and quality checking
- **Flexible Output**: Configurable visualization and analysis options

## Acknowledgements

- Glacier Bay National Park & Preserve
- Marine Robotics and Remote Sensing Lab @ Duke University

## License

See `LICENSE-CC-BY-SA.md` for details.
