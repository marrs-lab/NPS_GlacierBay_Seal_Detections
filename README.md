CropsAll folder contains all crops with JSONS used for model creation

Models folder contains the three models whose output is merged together for the final seal location output

Sample_Images folder contains sample seal images

Scripts folder contains more detailed processes
  
Scripts folder also contains crop data modulation scripts and some scripts that break 
ice thesholding and tracing into smaller steps

Running the Full_System_Run.py defaults to using the sample images
  Creates a log file of seal locations
  Uses the log file to draw overtop the original images
  Profiles Ice pieces and uses log file data to compare seal and ice locations
