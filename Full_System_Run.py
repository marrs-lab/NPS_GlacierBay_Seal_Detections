from Scripts import yolov8_NPS_workflow_v4_FINAL as Workflow
from Scripts import Draw_LOG as LOG
from Scripts import IceAndSealAnalysisFinal as Analysis
from Scripts import Log_CSV as LogToCsv
import time


def run_scripts():
    start_time = time.time()

    combined_model_path = "Models/CombinedV9.pt"
    color_model_path = "Models/ColorV9.pt"
    inverse_model_path = "Models/InverseV9.pt"
    
    image_directory = "Sample_Images/"
    
    # Run seal detections
    confidence = .85
    log_file_path = Workflow.run_workflow(combined_model_path, color_model_path, inverse_model_path, image_directory,confidence)

    # Convert Log file to a more friendly CSV file
    LogToCsv.log_to_csv(log_file_path)

    # Draw seals on photos
    LOG.run_draw_log(log_file_path, image_directory)

    # Analyze seal locations & profile ice
    Analysis.run_ice_and_seal_analysis(log_file_path, image_directory)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Time script took to run = {int(hours)} hours, {int(minutes)} minutes, and {int(seconds)} seconds.")
    
if __name__ == "__main__":
    run_scripts()
