from Scripts import yolov8_NPS_workflow_v4_FINAL as Workflow
from Scripts import Draw_LOG as LOG
from Scripts import IceAndSealAnalysisFinal as Analysis



def run_scripts():

    combined_model_path = "Models/CombinedV9.pt"
    color_model_path = "Models/ColorV9.pt"
    inverse_model_path = "Models/InverseV9.pt"
    
    image_directory = "Sample_Images/"
    
    confidence = .85
    log_file_path = Workflow.run_workflow(combined_model_path, color_model_path, inverse_model_path, image_directory,confidence)

    LOG.run_draw_log(log_file_path, image_directory)
    Analysis.run_ice_and_seal_analysis(log_file_path, image_directory)
    
if __name__ == "__main__":
    run_scripts()
