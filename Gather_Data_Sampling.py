import shutil
import pandas as pd
from pathlib import Path
import uuid

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
        # Build folder name: three levels up + one level up
        try:
            parent_level = ds_path.parents[3].name   # e.g. "JHI_Maiden1 Flight 01"
            run_id = ds_path.parents[0].name         # e.g. "20250911_1209_C80"
            target_dir_name = f"{parent_level}_{run_id}"
        except IndexError:
            print(f"Path too shallow to extract names properly: {ds_path}")
            continue

        # Ensure uniqueness (add UUID if collision)
        target_dir = master_dir / target_dir_name
        if target_dir.exists():
            unique_suffix = str(uuid.uuid4())[:8]
            target_dir_name = f"{target_dir_name}_{unique_suffix}"
            target_dir = master_dir / target_dir_name

        target_dir.mkdir(exist_ok=True)

        # Copy image files
        for file in ds_path.glob("*"):
            if file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                shutil.copy(file, target_dir)

        # Read and process CSV
        csv_path = ds_path / "sampled_data.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)

            if not df.empty:
                # Sort by the first column
                first_col = df.columns[0]
                df = df.sort_values(by=first_col)

            # Add SourceFolder column using new folder name
            df.insert(0, "SourceFolder", target_dir_name)

            combined_rows.append(df)
            # Add an empty row (all NaN values) for separation
            combined_rows.append(pd.DataFrame([{}]))
        else:
            print(f"No CSV found in: {ds_path}")

    # Combine all CSVs
    if combined_rows:
        combined_df = pd.concat(combined_rows, ignore_index=True)
        combined_csv_path = master_dir / "data_sampling_master.csv"
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"Combined CSV saved to: {combined_csv_path}")
    else:
        print("No CSV files were found to combine.")

# Example usage:
gather_data_sampling_folders(r"Z:\Projects\Clients\NPS_GlacierBay\2023\WingtraPilotProjects")
