import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_image(image_path, threshold, smoothing_kernel_size, output_folder):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Warning: Could not load image '{image_path}'. Skipping.")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.inRange(image_rgb, threshold, np.array([255, 255, 255]))

    kernel = np.ones((smoothing_kernel_size, smoothing_kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = np.zeros_like(image_rgb)

    cv2.drawContours(output, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    filename = os.path.basename(image_path)
    output_image_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_image_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    print(f"Processed and saved: {output_image_path}")

def process_images(folder_path, threshold=np.array([150, 150, 150]), smoothing_kernel_size=5, max_workers=4):
    # Check if the folder exists
    if not os.path.isdir(folder_path):
        raise ValueError(f"Folder path '{folder_path}' does not exist.")

    output_folder = os.path.join(folder_path, "IceThreshold")
    os.makedirs(output_folder, exist_ok=True)

    image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Use ThreadPoolExecutor to process images in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_image, image_path, threshold, smoothing_kernel_size, output_folder) for image_path in image_paths]

        for future in as_completed(futures):
            future.result()  # Wait for each future to complete

folder_path = r"C:\Users\sa553\Desktop\SummerHelp\JHI_FullSurvey_75m_Survey1 Flight 01"
threshold = np.array([150, 150, 150])
smoothing_kernel_size = 1
max_workers = 16  # Number of parallel workers

process_images(folder_path, threshold, smoothing_kernel_size, max_workers)
print("Processing complete.")
