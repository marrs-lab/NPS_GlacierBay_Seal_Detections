# Glacier Bay Seal Ice Preference Analysis

This repository provides a full workflow for detecting seals in drone imagery from Glacier Bay National Park, drawing detection results, and analyzing seal-ice relationships. The system uses YOLOv8 segmentation models.

## Repository Structure

- `Full_System_Run.py` — Main script to run the full detection and analysis pipeline.
- `Sample_Images/` — Example drone images for testing the workflow.
- `Models/` — Pretrained YOLOv8 segmentation models and tuning attempts.
- `Roboflow_V2/` — Dataset and configuration files for model training.
- `Scripts/` — Core processing scripts:
  - `yolov8m_seg_detection_workflow.py` — Image tiling, detection, and CSV output.
  - `draw_detections.py` — Draws bounding boxes on images from detection CSVs.
  - `seal_ice_analysis.py` — Analyzes seal detections in relation to ice, outputs summary CSVs and annotated images.

## Workflow Overview

1. **Detection**: Images are tiled and processed in parallel using a YOLOv8 segmentation model. Detections are merged and saved to a CSV file. Optionally, bounding boxes are drawn on images based on detection results.
2. **Drawing**: Optionally, bounding boxes are drawn on images based on the csv of detection results.
3. **Seal-Ice Analysis**: For each detected seal, the area of the ice patch it is located on is calculated and included in the per-seal CSV output. An additional CSV summarizes all ice regions detected in each image. Annotated images are also optionally generated.

## License
See `LICENSE-CC-BY-SA.md` for details.
