"""
Created on Wed Nov 21 02:04:18 2023

@author: Saagar

Takes a directory of crops. Copies them to a new directory and augments them 90, 180, 270, mirrored x, and mirrored y.
"""
import os
from PIL import Image
import numpy as np
import json
import shutil
import base64

# directory of crops and polygons
crops = r"C:\Users\sa553\Desktop\NPS\TrainingSets\CropsAll"
# directory of augmented crops and polygon files
# outdirectory
augmentedCrops = r"C:\Users\sa553\Desktop\NPS\TrainingSets\ColorCropsAll"
d = 640  # crop width/height (always square)


def rotate_point(point, angle, center):
    """Rotate a point around a center."""
    x, y = point
    cx, cy = center
    angle_rad = np.radians(angle)
    new_x = (x - cx) * np.cos(angle_rad) + (y - cy) * np.sin(angle_rad) + cx
    new_y = (x - cx) * -np.sin(angle_rad) + (y - cy) * np.cos(angle_rad) + cy
    return [new_x, new_y]


def rotate_polygon(polygon, angle, center):
    """Rotate a polygon around a center."""
    return [rotate_point(point, angle, center) for point in polygon]


def mirror_polygon_x(polygon, image_width):
    """Mirror a polygon over the x-axis."""
    return [[image_width - x, y] for x, y in polygon]


def mirror_polygon_y(polygon, image_height):
    """Mirror a polygon over the y-axis."""
    return [[x, image_height - y] for x, y in polygon]


def image_to_base64(img_path):
    """Convert an image to base64."""
    with open(img_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read())
    return encoded.decode('utf-8')


for filename in os.listdir(crops):
    # Check if the file is an image (e.g., JPG or JPEG)
    if filename.lower().endswith(('.jpg', '.jpeg')):
        # Load the original image
        original_image = Image.open(os.path.join(crops, filename))

        # Load the associated JSON file
        json_filename = os.path.splitext(filename)[0] + '.json'
        json_filepath = os.path.join(crops, json_filename)
        if os.path.isfile(json_filepath):
            with open(json_filepath, 'r') as json_file:
                json_data = json.load(json_file)

            # Copy the original image and JSON file to the augmented directory
            shutil.copy(os.path.join(crops, filename),
                        os.path.join(augmentedCrops, filename))
            shutil.copy(json_filepath, os.path.join(
                augmentedCrops, json_filename))

            # Rotate the image and save with the new filenames
            for angleO in [90, 180, 270]:  # Change the order to process 90, 180, and then 270
                rotated_image = original_image.rotate(angleO)
                rotated_filename = os.path.splitext(
                    filename)[0] + f"_{angleO}.jpg"
                rotated_image.save(os.path.join(
                    augmentedCrops, rotated_filename), format='JPEG')

                # Rotate all polygons in the JSON data
                # Rotate all polygons in the JSON data
                if angleO == 90:
                    angle = 270
                elif angleO == 180:
                    angle = 270
                elif angleO == 270:
                    angle = 270
                for shape in json_data["shapes"]:
                    rotated_polygon = rotate_polygon(
                        shape["points"], -angle, center=(d/2, d/2))  # Note the negative angle
                    shape["points"] = rotated_polygon

                # Update JSON data with rotated polygons and base64-encoded image data
                rotated_json_filename = os.path.splitext(rotated_filename)[
                    0] + ".json"
                json_data["imagePath"] = rotated_filename
                json_data["imageData"] = image_to_base64(
                    os.path.join(augmentedCrops, rotated_filename))
                with open(os.path.join(augmentedCrops, rotated_json_filename), 'w') as rotated_json_file:
                    json.dump(json_data, rotated_json_file)

            # Mirror over the x-axis
            angle = angleO
            mirrored_x_image = original_image.transpose(Image.FLIP_TOP_BOTTOM)
            mirrored_x_filename = os.path.splitext(filename)[0] + "_mx.jpg"
            mirrored_x_image.save(os.path.join(
                augmentedCrops, mirrored_x_filename), format='JPEG')

            # Mirror all polygons over the x-axis
            for shape in json_data["shapes"]:
                rotated_polygon = rotate_polygon(
                    shape["points"], -90, center=(d/2, d/2))  # Note the negative angle
                shape["points"] = rotated_polygon
                mirrored_x_polygon = mirror_polygon_x(shape["points"], d)
                shape["points"] = mirrored_x_polygon

            # Update JSON data with mirrored polygons and base64-encoded image data
            mirrored_x_json_filename = os.path.splitext(
                mirrored_x_filename)[0] + ".json"
            json_data["imagePath"] = mirrored_x_filename
            json_data["imageData"] = image_to_base64(
                os.path.join(augmentedCrops, mirrored_x_filename))
            with open(os.path.join(augmentedCrops, mirrored_x_json_filename), 'w') as mirrored_x_json_file:
                json.dump(json_data, mirrored_x_json_file)

            # Mirror over the y-axis
            mirrored_y_image = original_image.transpose(Image.FLIP_LEFT_RIGHT)
            mirrored_y_filename = os.path.splitext(filename)[0] + "_my.jpg"
            mirrored_y_image.save(os.path.join(
                augmentedCrops, mirrored_y_filename), format='JPEG')

            # Mirror all polygons over the y-axis
            for shape in json_data["shapes"]:
                mirrored_y_polygon = mirror_polygon_y(shape["points"], d)
                mirrored_y_polygon = mirror_polygon_x(mirrored_y_polygon, d)
                shape["points"] = mirrored_y_polygon

            # Update JSON data with mirrored polygons and base64-encoded image data
            mirrored_y_json_filename = os.path.splitext(
                mirrored_y_filename)[0] + ".json"
            json_data["imagePath"] = mirrored_y_filename
            json_data["imageData"] = image_to_base64(
                os.path.join(augmentedCrops, mirrored_y_filename))
            with open(os.path.join(augmentedCrops, mirrored_y_json_filename), 'w') as mirrored_y_json_file:
                json.dump(json_data, mirrored_y_json_file)

            print(f'{filename} processed and saved in {augmentedCrops}')
        else:
            print(f'Skipping {filename} as no associated JSON file found.')
