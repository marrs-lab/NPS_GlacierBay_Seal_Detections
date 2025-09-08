
Glacier Bay - Harbor Seals -- Paper Data Set - v1 Roboflow_V3
==============================

This dataset was exported via roboflow.com on September 8, 2025 at 4:28 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 3520 images.
Objects-L0k1-NXtZ are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 4 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Randomly crop between 0 and 10 percent of the image
* Random rotation of between -15 and +15 degrees
* Random brigthness adjustment of between -15 and +15 percent
* Random exposure adjustment of between -10 and +10 percent
* Salt and pepper noise was applied to 1 percent of pixels

The following transformations were applied to the bounding boxes of each image:
* Random shear of between -4° to +4° horizontally and -4° to +4° vertically


