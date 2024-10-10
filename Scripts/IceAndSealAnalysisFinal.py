import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import shutil
from PIL import Image, ExifTags
import time

# Function to convert GPS coordinates to decimal format
def convert_to_degrees(value):
    d = float(value[0])
    m = float(value[1])
    s = float(value[2])
    return d + (m / 60.0) + (s / 3600.0)

# Function to extract latitude and longitude from image EXIF
def get_lat_lon(image_path):
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        if exif_data is not None:
            gps_info = None
            for tag, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                if tag_name == "GPSInfo":
                    gps_info = value
                    break
            if gps_info:
                lat_data = gps_info[2]
                lon_data = gps_info[4]
                lat_ref = gps_info[1]
                lon_ref = gps_info[3]

                latitude = convert_to_degrees(lat_data)
                if lat_ref != 'N':
                    latitude = -latitude

                longitude = convert_to_degrees(lon_data)
                if lon_ref != 'E':
                    longitude = -longitude

                return latitude, longitude
        return None, None
    except Exception as e:
        print(f"Error extracting GPS data from {image_path}: {e}")
        return None, None

# Function to correct image orientation
def correct_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()
        if exif is not None:
            orientation = exif.get(orientation)
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass  # Image doesn't have EXIF orientation info
    return image

# Function to copy image to output folder before processing
def copy_image(image_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    destination_path = os.path.join(output_folder, os.path.basename(image_path))
    shutil.copyfile(image_path, destination_path)
    return destination_path

# Function to process and save thresholded images and trace ice contours
def process_image_with_seals(image_path, lower_threshold, upper_threshold, smoothing_kernel_size, seal_coordinates, outlined_folder):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image '{image_path}'. Skipping.")
        return None

    # Convert image to RGB for thresholding
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply lower and upper threshold to exclude overly bright spots
    lower_mask = cv2.inRange(image_rgb, lower_threshold, np.array([255, 255, 255]))
    upper_mask = cv2.inRange(image_rgb, np.array([0, 0, 0]), upper_threshold)

    # Combine the masks to include valid ice but exclude bright reflections
    mask = cv2.bitwise_and(lower_mask, upper_mask)

    # Apply morphological operations to smooth the mask
    kernel = np.ones((smoothing_kernel_size, smoothing_kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours of the ice
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the ice contours on the original image
    outlined_image = cv2.drawContours(image_rgb, contours, -1, (255, 0, 0), 2)
    outlined_image_path = os.path.join(outlined_folder, os.path.basename(image_path))
    cv2.imwrite(outlined_image_path, cv2.cvtColor(outlined_image, cv2.COLOR_RGB2BGR))

    # Calculate ice chunk sizes
    areas = [cv2.contourArea(contour) for contour in contours]

    coordinates = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:  # Avoid division by zero
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            coordinates.append((cX, cY))
        else:
            coordinates.append(None)  # No center if the contour area is zero

    # Calculate which ice chunk each seal is on
    seal_ice_areas = []
    for bbox in seal_coordinates:
        x1, y1, x2, y2 = bbox
        seal_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        bbox_area = (x2 - x1) * (y2 - y1)
        found_ice_area = 0

        for contour in contours:
            if cv2.pointPolygonTest(contour, seal_center, False) >= 0:
                found_ice_area = cv2.contourArea(contour)
                break

        # If the found ice area matches the bounding box area, it's water (0 ice)
        if found_ice_area == bbox_area:
            found_ice_area = 0

        seal_ice_areas.append((bbox, found_ice_area))

    return areas, seal_ice_areas, coordinates

# Function to generate histogram and save CSV with ice data
def save_ice_distribution(image_name, areas, coordinates, histogram_folder, csv_data):
    if areas:
        avg_area = np.mean(areas)
        largest_area = max(areas)
    else:
        avg_area = 0
        largest_area = 0

    # Save histogram
    plt.figure(figsize=(10, 6))
    plt.hist(areas, bins=np.linspace(0, largest_area + 10, 30), edgecolor='black')
    plt.title(f'Ice Area Distribution for {image_name}')
    plt.xlabel('Ice Area')
    plt.ylabel('Frequency')
    plt.grid(True)
    histogram_path = os.path.join(histogram_folder, f"{image_name}.png")
    plt.savefig(histogram_path)
    plt.close()
    data_front_append = [image_name, avg_area, largest_area]
    data_to_append = [item for pair in zip(coordinates, areas) for item in pair]
    csv_data.append(data_front_append + data_to_append)
    
# Function to apply mask color to seal bounding boxes (included in ice mask)
def apply_mask_color(image, seal_coordinates, mask_color=(200, 200, 200)):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply mask color to seal bounding boxes
    for bbox in seal_coordinates:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), mask_color, thickness=cv2.FILLED)

    return image_rgb

# Function to mask image seals and save to IceThresholdOutlined
def mask_and_save_seal_areas(image_path, seal_coordinates, outlined_folder, mask_color=(200, 200, 200)):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image '{image_path}'. Skipping.")
        return None
    
    # Apply the mask color to seals
    masked_image = apply_mask_color(image, seal_coordinates, mask_color)
    
    # Save the masked image
    outlined_image_path = os.path.join(outlined_folder, os.path.basename(image_path))
    cv2.imwrite(outlined_image_path, cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
    return outlined_image_path

# Function to process log and images
def process_log_and_images(log_file_path, image_folder, lower_threshold=np.array([150, 150, 150]), upper_threshold=np.array([240, 240, 240]), smoothing_kernel_size=5, output_folder="ProcessedImages"):
    if not os.path.isdir(image_folder):
        raise ValueError(f"Image folder '{image_folder}' does not exist.")
    
    os.makedirs(output_folder, exist_ok=True)

    # Create additional output folders
    outlined_folder = os.path.join(output_folder, "IceThresholdOutlined")
    histogram_folder = os.path.join(output_folder, "IceDistributions")
    os.makedirs(outlined_folder, exist_ok=True)
    os.makedirs(histogram_folder, exist_ok=True)

    seal_ice_data = []
    csv_data = []

    with open(log_file_path, 'r') as log_file:
        lines = log_file.readlines()

    for line in lines:
        
        if not line.strip():
            continue
        
        print(line)
        parts = line.strip().split(', ')
        if not parts:
            continue

        image_name = parts[0]
        image_path = os.path.join(image_folder, image_name)

        # Get seal bounding boxes
        seal_coordinates = []
        for part in parts[1:]:
            coords = list(map(float, part.split()))
            if len(coords) == 4:
                seal_coordinates.append([int(c) for c in coords])

        # Extract GPS data
        latitude, longitude = get_lat_lon(image_path)

        # Step 1: Mask and save the image to IceThresholdOutlined
        masked_image_path = mask_and_save_seal_areas(image_path, seal_coordinates, outlined_folder)

        # Step 2: Process the masked image, trace ice, and save thresholded image
        areas, seal_ice_chunks, coordinates = process_image_with_seals(masked_image_path, lower_threshold, upper_threshold, smoothing_kernel_size, seal_coordinates, outlined_folder)
        if areas:
            if coordinates:
                save_ice_distribution(image_name, areas, coordinates, histogram_folder, csv_data)

        # Save seal ice chunk analysis data
        for bbox, ice_area in seal_ice_chunks:
            seal_ice_data.append([image_name, latitude, longitude, bbox, ice_area])

    # Save CSV file with ice data
    csv_file_path = os.path.join(histogram_folder, "ice_area_analysis.csv")
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Image Filename', 'Average Ice Area', 'Largest Ice Area', 'Coordinate', 'Area'])
        csv_writer.writerows(csv_data)

    # Save CSV file with seal ice chunk data
    seal_csv_file_path = os.path.join(output_folder, "seal_ice_chunk_analysis.csv")
    with open(seal_csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Image Filename','Latitude', 'Longitude','Seal Bounding Box', 'Ice Chunk Area'])
        csv_writer.writerows(seal_ice_data)

    print(f"Analysis complete. Ice data saved to {csv_file_path}")
    print(f"Seal ice chunk analysis saved to {seal_csv_file_path}")

# Callable function for external use
def run_ice_and_seal_analysis(log_file_path, image_folder, lower_threshold=np.array([150, 150, 150]), upper_threshold=np.array([240, 240, 240]), smoothing_kernel_size=5, output_folder=None):
    if output_folder is None:
        output_folder = os.path.join(image_folder, "Seal_Ice_Overlap")
    
    start_time = time.time()
    process_log_and_images(log_file_path, image_folder, lower_threshold, upper_threshold, smoothing_kernel_size, output_folder)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Processing time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

if __name__ == "__main__":
    log_file_path = r''
    image_directory = r''
    run_ice_and_seal_analysis(log_file_path, image_directory)
