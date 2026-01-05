# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 12:00:46 2025

@author: Tobie
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Load data, skip header junk ---
file_path = r'E:/Data/Raw/ICP-MS/ECAL_Bloom(Prather)_20251217.csv'   # safer path with raw string
df = pd.read_csv(file_path)

# --- Clean data ---
df = df.dropna(how="all").reset_index(drop=True)

# --- Helpers to explore ---
def list_elements():
    """Return a list of element columns (with units like [ppb])."""
    return [col for col in df.columns if "(" in col or "[" in col]

def list_samples():
    """Return a list of available sample labels."""
    return df["Label"].dropna().astype(str).unique().tolist()

# --- Plotting function ---
def plot_elements(element_cols, sample_filter=None):
    """
    Draw grouped bar chart for chosen elements and samples.
    element_cols: list of element column names
    sample_filter: list of sample names (or None for all)
    """
    if isinstance(element_cols, str):
        element_cols = [element_cols]
    
    # Subset
    data = df[["Label"] + element_cols].copy()
    
    # Convert to numeric
    for col in element_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    
    # Drop empty rows
    data = data.dropna(subset=element_cols, how="all")
    
    # Apply sample filter (and enforce order)
    if sample_filter:
        data = data[data["Label"].astype(str).isin(sample_filter)]
        # Reindex to follow sample_filter order exactly
        data["Label"] = data["Label"].astype(str)
        data = data.set_index("Label").reindex(sample_filter).reset_index()
    
    # Grouped bar chart
    labels = data["Label"].astype(str).values
    x = np.arange(len(labels))
    width = 0.8 / len(element_cols)
    
    plt.figure(figsize=(12,6))
    for i, col in enumerate(element_cols):
        plt.bar(x + i*width, data[col].values, width, label=col)
    
    plt.xticks(x + width*(len(element_cols)-1)/2, labels, rotation=90)
    plt.title("ICP-MS Field Samples")
    plt.xlabel("Date", fontsize=10)
    plt.ylabel("Concentration [ppb]", fontsize=10)
    plt.legend( fontsize=18)
    plt.tight_layout()
    plt.show()

# --- Example usage ---
print("Available elements:", list_elements())  # show first 10
print("Available samples:", list_samples())

# Plot one element for all samples
# Custom sample order
plot_elements(
   # [ '55Mn (KED) [ppb]', '60Ni (KED) [ppb]', '63Cu (KED) [ppb]', '66Zn (KED) [ppb]'],
  ['51V (KED) [ppb]', '52Cr (KED) [ppb]', '59Co (KED) [ppb]', '75As (KED) [ppb]', '77Se (KED) [ppb]', '111Cd (KED) [ppb]', '208Pb (KED) [ppb]'],
    sample_filter=['IBA_05032025', 'IBA_05042025', 'IBA_05052025', 'IBA_05082025', 'IBA_05132025', 'IBA_05142025', 'IBA_05152025', 'IBA_05162025', 'IBA_05172025', 'IBA_05182025', 'IBA_05192025', 'IBA_05202025', 'IBA_05212025', 'IBA_05222025', 'IBA_05232025', 'IBA_05242025', 'IBA_05252025', 'IBA_05262025', 'IBA_05272025', 'FB_05252025', 'MB_12162025']
)
#trace 
#'51V (KED) [ppb]', '52Cr (KED) [ppb]', '59Co (KED) [ppb]', '75As (KED) [ppb]', '77Se (KED) [ppb]', '111Cd (KED) [ppb]', '208Pb (KED) [ppb]'
#Large 
#  '55Mn (KED) [ppb]', '60Ni (KED) [ppb]', '63Cu (KED) [ppb]', '66Zn (KED) [ppb]'


# Plot multiple elements for selected samples
# plot_elements(["23Na (KED) [ppb]", "24Mg (KED) [ppb]"], 
#               sample_filter=["Sample_01", "Sample_02"])
