# -*- coding: utf-8 -*-
"""
   @author: Saagar
   
   Creates a CSV file that that splits the name of analyzed photos 
   into parts that can be mapped in a spreadsheet containing name 
   and number of seals that can be used to validate results and accuracy

   """
import os
import csv

# Path to the directory containing the images
directory_path = r"C:\Users\sa553\Desktop\NPS\JHI_FullSurvey_75m_Survey1 Flight 01\IMAGES\Recomb\Inverse (1) (0.85) (1639)"
csv_file_path = directory_path+".csv"

# Create a list to store the data
data_list = [["Image Number", "Image Name", "Number of Seals"]]

# Iterate through the files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.JPG') or filename.endswith('.jpg'):  # Adjust the file extension based on your actual files
        # Extract relevant information from the filename
        parts = filename.split('_')

        # Print debug information
        print(f"Filename: {filename}")
        print(f"Parts: {parts}")

        image_number = parts[5]
        image_name = '_'.join(parts[:6])
        number_of_seals_list = parts[7].split('.')
        number_of_seals = number_of_seals_list[0]        

        # Append the data to the list
        data_list.append([image_number, image_name, number_of_seals])

# Sort the data list by "Image Name"
data_list.sort(key=lambda x: x[1])

# Print debug information
print(f"Data List: {data_list}")

# Export the data to a CSV file
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(data_list)

print(f"CSV file exported successfully to {csv_file_path}")
