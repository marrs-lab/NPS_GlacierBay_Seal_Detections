# -*- coding: utf-8 -*-
"""
   @author: David
   
   Meant for creating a set of crops based off of images and
   naming the crops with the grid dimensions of each crops
   """
import os
import shutil

# Path to the directory containing the images
directory_in = r''
directory_out = r''
# Iterate through the files in the directory
for filename in os.listdir(directory_in):
    if filename.endswith('.JPG') or filename.endswith('.jpg'):  # Adjust the file extension based on your actual files
        # Extract relevant information from the filename
        parts = filename.split('_')

        # Print debug information
        print(f"Filename: {filename}")
        print(f"Parts: {parts}")

        image_number = parts[5]
        image_name = '_'.join(parts[:7])+'.jpg'
        number_of_seals_list = parts[8].split('.')
        number_of_seals = number_of_seals_list[0] 

        if (int(number_of_seals)>0):
            shutil.copy(os.path.join(directory_in,filename), os.path.join(directory_out,image_name))
