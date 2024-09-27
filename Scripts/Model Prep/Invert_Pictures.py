# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 00:38:22 2023

@author: Saagar

Invert Pictures. Leaves the inverted pictures in the specified directory.
Does Not copy JSONs. Only makes inverted photos. 

"""

import os
from PIL import Image, ImageOps

# Directory of images
images_directory = r"C:\Users\sa553\Desktop\NPS\Testing\TestSetColorAndInverse"

# Specify the directory to store inverted photos
output_directory = r"C:\Users\sa553\Desktop\NPS\Testing\TestSetColorAndInverse"

def invert_and_save(image_path, output_directory):
    """Invert the colors of an image and save it."""
    # Load the original image
    original_image = Image.open(image_path)

    # Invert the colors of the image
    inverted_image = ImageOps.invert(original_image)

    # Save the inverted image to the output directory
    inverted_filename = os.path.basename(image_path).split('.')[0] + "_inv.jpg"
    inverted_image.save(os.path.join(
        output_directory, inverted_filename), format='JPEG')
    print(f'{os.path.basename(image_path)} processed and saved in {output_directory}')

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Iterate over images in the "images" directory
for filename in os.listdir(images_directory):
    # Check if the file is an image (e.g., JPG or JPEG)
    if filename.lower().endswith(('.jpg', '.jpeg')):
        # Invert the image and save to the specified directory
        invert_and_save(os.path.join(
            images_directory, filename), output_directory)
