import os
import shutil
from PIL import Image, ImageDraw, ExifTags

# Function to create the LogVerification folder if it doesn't exist
def setup_log_verification_folder(log_verification_directory):
    if not os.path.exists(log_verification_directory):
        os.makedirs(log_verification_directory)

# Function to copy a single image from source to the LogVerification folder
def copy_image(image_name, source_image_directory, log_verification_directory):
    src_file = os.path.join(source_image_directory, image_name)
    dst_file = os.path.join(log_verification_directory, image_name)
    if os.path.exists(src_file):
        shutil.copyfile(src_file, dst_file)
        return dst_file  # Return the new image path in the LogVerification folder
    else:
        print(f"Image {image_name} not found in the source directory.")
        return None

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

# Function to draw rectangles on an image
def draw_rectangles(image_path, coordinates):
    with Image.open(image_path) as img:
        img = correct_orientation(img)
        draw = ImageDraw.Draw(img)
        for coord in coordinates:
            x1, y1, x2, y2 = coord
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        img.save(image_path, format='JPEG', subsampling=0, quality=100)

# Function to read the log file and process each image
def process_log(log_file_path, source_image_directory, log_verification_directory):
    with open(log_file_path, 'r') as log_file:
        lines = log_file.readlines()

    for line in lines:
        if not line.strip():
            continue
        
        parts = line.strip().split(', ')
        if not parts:
            continue

        image_name = parts[0]
        new_image_path = copy_image(image_name, source_image_directory, log_verification_directory)
        if new_image_path is None:
            continue  # Skip if image not found

        # Collect coordinates from the log
        coordinates = []
        for part in parts[1:]:
            coords = list(map(float, part.split()))
            if len(coords) == 4:
                coordinates.append([int(c) for c in coords])

        # Draw rectangles on the copied image
        if os.path.exists(new_image_path):
            draw_rectangles(new_image_path, coordinates)
        else:
            print(f"Image {new_image_path} not found in LogVerification directory.")

# Main function
def run_draw_log(log_file_path, source_image_directory):
    log_verification_directory = os.path.join(source_image_directory, 'Log')
    setup_log_verification_folder(log_verification_directory)
    process_log(log_file_path, source_image_directory, log_verification_directory)

# If called directly, execute the main function
if __name__ == '__main__':
    log_file_path = r'C:\Users\sa553\Desktop\NPS\JHI_FullSurvey_75m_Survey1 Flight 01\IMAGES\Recomb\FinalFlightAnalysis (3) (0.85) (1875)\LOG.txt'
    source_image_directory = r'C:\Users\sa553\Desktop\NPS\JHI_FullSurvey_75m_Survey1 Flight 01\IMAGES'        
    run_draw_log(log_file_path, source_image_directory)
