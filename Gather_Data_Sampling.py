import os
import shutil
import pandas as pd
from pathlib import Path

def gather_data_sampling_folders(root_dir):
    root_dir = Path(root_dir)
    master_dir = root_dir / "Data_Sampling_Master"
    master_dir.mkdir(exist_ok=True)

    combined_rows = []
    all_found = list(root_dir.rglob("Data_Sampling"))

    if not all_found:
        print("No 'Data_Sampling' folders found.")
        return

    for ds_path in all_found:
        parent_name = ds_path.parent.name
        target_dir = master_dir / parent_name
        target_dir.mkdir(exist_ok=True)

        # Copy all image files
        for file in ds_path.glob("*"):
            if file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                shutil.copy(file, target_dir)

        # Read CSV if exists
        csv_path = ds_path / "sampled_data.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df.insert(0, "SourceFolder", parent_name)
            combined_rows.append(df)
        else:
            print(f"No CSV found in: {ds_path}")

    # Combine all CSVs
    if combined_rows:
        combined_df = pd.concat(combined_rows, ignore_index=True)
        combined_csv_path = master_dir / "combined_sampled_data.csv"
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"Combined CSV saved to: {combined_csv_path}")
    else:
        print("No CSV files were found to combine.")

# Example usage:
gather_data_sampling_folders(r"Z:\Projects\Clients\NPS_GlacierBay\2023\WingtraPilotProjects")