# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 12:00:46 2025

@author: Tobie
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# USER SETTINGS
# =========================
FOLDER  = r"e:\Data\Raw\ICP-MS\MasterMetal"   # <-- folder with your ICP-MS csvs
PATTERN = "*.csv"                # or "ECAL_Bloom(Prather)_*.csv"

# =========================
# LOAD ALL FILES IN FOLDER
# =========================
file_list = sorted(glob.glob(os.path.join(FOLDER, PATTERN)))

if len(file_list) == 0:
    raise FileNotFoundError(f"No files found in {FOLDER} matching {PATTERN}")

print(f"Found {len(file_list)} files")

df_list = []
for fp in file_list:
    print("Reading:", os.path.basename(fp))
    try:
        tmp = pd.read_csv(fp, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        tmp = pd.read_csv(fp, encoding="latin1", low_memory=False)

    tmp = tmp.dropna(how="all").reset_index(drop=True)
    tmp["Source_File"] = os.path.basename(fp)
    df_list.append(tmp)

df = pd.concat(df_list, ignore_index=True)
df = df.dropna(how="all").reset_index(drop=True)

# =========================
# HELPERS
# =========================
def list_elements():
    """Return a list of element columns (with units like [ppb])."""
    return [col for col in df.columns if "(" in col or "[" in col]

def list_samples():
    """Return a list of available sample labels (across all files)."""
    if "Label" not in df.columns:
        return []
    return df["Label"].dropna().astype(str).unique().tolist()

def list_source_files():
    """Return list of loaded filenames."""
    return df["Source_File"].dropna().astype(str).unique().tolist()

# =========================
# PLOTTING
# =========================
def plot_elements(element_cols, sample_filter=None, source_files=None, title="ICP-MS Field Samples"):
    """
    Grouped bar chart with x-axis = Label + Source_File (no averaging).
    element_cols: list (or str) of element column names
    sample_filter: list of sample names (or None for all)
    source_files: list of filenames to include (or None for all)
    """
    if isinstance(element_cols, str):
        element_cols = [element_cols]

    # Required columns
    if "Label" not in df.columns:
        raise KeyError("Column 'Label' not found in dataframe. Check your CSV headers.")
    if "Source_File" not in df.columns:
        raise KeyError("Column 'Source_File' not found. Something went wrong during folder load.")

    keep_cols = ["Source_File", "Label"] + element_cols
    data = df[keep_cols].copy()

    # Optional: filter by source file(s)
    if source_files is not None:
        source_files = [str(x) for x in source_files]
        data["Source_File"] = data["Source_File"].astype(str)
        data = data[data["Source_File"].isin(source_files)]

    # Convert to numeric
    for col in element_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Drop rows where all chosen elements are NaN
    data = data.dropna(subset=element_cols, how="all")

    # Optional: filter by sample label(s)
    if sample_filter:
        sample_filter = [str(x) for x in sample_filter]
        data["Label"] = data["Label"].astype(str)
        data = data[data["Label"].isin(sample_filter)]

    # Build combined x labels
    data["XLabel"] = data["Label"].astype(str) 

    # Keep deterministic order (Label then file), or preserve file order
    data["Source_File"] = pd.Categorical(data["Source_File"], categories=list_source_files(), ordered=True)
    data = data.sort_values(["Label", "Source_File"]).reset_index(drop=True)

    labels = data["XLabel"].values
    x = np.arange(len(labels))
    width = 0.8 / max(len(element_cols), 1)

    plt.figure(figsize=(max(12, 0.35 * len(labels)), 6))
    for i, col in enumerate(element_cols):
        plt.bar(x + i * width, data[col].values, width, label=col)

    plt.xticks(x + width * (len(element_cols) - 1) / 2, labels, rotation=90)
    plt.title(title)
    plt.xlabel("Sample Label | Source File", fontsize=10)
    plt.ylabel("Concentration [ppb]", fontsize=10)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

# =========================
# EXAMPLE USAGE
# =========================
print("Available elements (first 20):", list_elements()[:20])
print("Available samples:", list_samples())
print("Loaded source files:", list_source_files())

plot_elements(
    #['55Mn (KED) [ppb]', '60Ni (KED) [ppb]', '63Cu (KED) [ppb]', '66Zn (KED) [ppb]'],
    ['51V (KED) [ppb]', '52Cr (KED) [ppb]', '59Co (KED) [ppb]', '75As (KED) [ppb]', '77Se (KED) [ppb]', '111Cd (KED) [ppb]', '208Pb (KED) [ppb]'],
    sample_filter=['917amSTR', '917pmSTR', '917AMS_air', '917PMS_air'],
    #title="ICP-MS Ulva Samples (Label + Source File)"
)

# You can also filter to specific files like:
# plot_elements(
#     ['55Mn (KED) [ppb]'],
#     source_files=["ECAL_Bloom(Prather)_20260224.csv"],
#     title="Mn from one run"
# )