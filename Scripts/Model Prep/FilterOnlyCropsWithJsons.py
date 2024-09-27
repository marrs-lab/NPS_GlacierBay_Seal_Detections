# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 22:58:15 2023

@author: sa553
Filters crops with only associated JSONS from subdirectories to a new directory
"""

import os
import shutil

dir_in = r"C:\Users\sa553\Desktop\NPS\TrainingSets\CropsAllSource\David_Crops"
dir_out = r"C:\Users\sa553\Desktop\NPS\TrainingSets\CropsAll"

def copy_images_with_json(dir_in, dir_out):
    # Ensure the output directory exists
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    # Iterate through files in the input directory
    for filename in os.listdir(dir_in):
        # Check if the file is an image (you may want to add more file format checks)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image_path = os.path.join(dir_in, filename)
            json_path = os.path.join(dir_in, f"{os.path.splitext(filename)[0]}.json")

            # Check if the associated JSON file exists
            if os.path.exists(json_path):
                # Copy the image and its associated JSON file to the output directory
                shutil.copy(image_path, dir_out)
                shutil.copy(json_path, dir_out)
                print(f"Copied {filename} and its associated JSON.")
                
copy_images_with_json(dir_in, dir_out)


