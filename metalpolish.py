import pandas as pd
import glob
import os

# -------- USER SETTINGS -------- #
input_folder = "e:\Data\Raw\ICP-MS\MasterMetal"
output_file = "e:\Data\Processed\ICPMS\combined_heavy_metals.csv"

sample_id_col = "Label"  # change if needed

# -------- PROCESS FILES -------- #
all_data = [] 

csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

for file in csv_files:
    df = pd.read_csv(file)

    # --- Check Sample ID column --- #
    if sample_id_col not in df.columns:
        raise ValueError(f"{sample_id_col} not found in {file}")

    # --- Keep Sample ID + EVERYTHING ELSE --- #
    new_df = df.copy()

    # --- Add required columns --- #
    new_df["Date"] = ""
    new_df["Time"] = ""
    new_df["Notes"] = os.path.basename(file)  # track source file
    new_df["Type"] = ""

    # --- Reorder columns --- #
    cols = new_df.columns.tolist()

   

    all_data.append(new_df)

# -------- COMBINE -------- #
final_df = pd.concat(all_data, ignore_index=True)

# -------- SAVE -------- #
final_df.to_csv(output_file, index=False)

print(f"Combined file saved to: {output_file}")