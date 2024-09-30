**Glacier Bay Seal Ice Preference Analysis**
**Overview**
This project provides a workflow for processing drone imagery from Glacier Bay National Park to analyze seal ice preference. The workflow includes scripts that detect seal locations, draw bounding boxes (optional), and analyze ice formations in relation to those locations. The final output reports both seal locations and the ice characteristics, including ice area of seals located on ice.

**Workflow:**
**Seal Detection**: A script processes the images to infer seal locations using the combined results of three YOLOV8 models.

**Optional Bounding Box Visualization**: Another script can be run to draw bounding boxes around detected seals.

**Ice Formation Analysis**: A final script identifies ice formations, compares them with seal locations, and returns ice characteristics (e.g., area of ice if a seal is on it).

**Directory Structure**
**CropsAll Folder**: Contains image crops and corresponding JSON files used for model training. Crops were generated from larger images containing seals, and the seals were traced using LabelMe.

**Models Folder**: Contains three models whose outputs are combined to produce the final seal detection result:

A combined model trained on both inverse and color crops.
A color model trained only on color crops.
An inverse model trained only on inverse crops.
These models are merged to increase confidence in seal detection during inference.

**Sample_Images Folder**: Includes sample images of seals for testing and demonstration purposes.

**Scripts Folder**: Contains scripts for detailed processing, including crop image augmentation for model creation and smaller step scripts for ice thresholding and tracing.

**Usage Instructions**
Running the main script Full_System_Run.py with the sample images will:
**Generate a log file** of seal locations.
**Optionally overlay bounding boxes** on the original images (using log file data).
**Profile ice formations** and compare them with seal locations to report ice characteristics and area, if applicable.
