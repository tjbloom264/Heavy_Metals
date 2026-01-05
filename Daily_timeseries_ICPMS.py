# -*- coding: utf-8 -*-

import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
# Your original path:
file_path = r"/Users/auroraczajkowski/Desktop/COAST Metal Data/ECAL_Bloom(Prather)_20251217.csv"

# If you ever run this in the ChatGPT notebook environment, use:
# file_path = "/mnt/data/ECAL_Bloom(Prather)_20251217.csv"

df = pd.read_csv(file_path)

# ------------------------------------------------------------
# Clean data
# ------------------------------------------------------------
df = df.dropna(how="all").reset_index(drop=True)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def list_elements():
    """Return a list of element columns (columns containing '(' or '[' in the name)."""
    return [col for col in df.columns if ("(" in col) or ("[" in col)]

def list_samples():
    """Return a list of available sample labels from the Label column."""
    if "Label" not in df.columns:
        raise KeyError("Could not find a 'Label' column in the CSV.")
    return df["Label"].dropna().astype(str).unique().tolist()

def label_to_date(label: str) -> pd.Timestamp:
    """
    Extract MMDDYYYY from labels like IBA_05032025 (or FB_05252025, etc.)
    Returns pandas Timestamp or NaT if it can't parse.
    """
    if pd.isna(label):
        return pd.NaT
    s = str(label).strip()

    # Match trailing _MMDDYYYY
    m = re.search(r"_(\d{8})$", s)
    if not m:
        return pd.NaT

    return pd.to_datetime(m.group(1), format="%m%d%Y", errors="coerce")

# ------------------------------------------------------------
# Plotting function
# ------------------------------------------------------------
def plot_elements(element_cols, sample_filter=None, date_fmt="%Y-%m-%d"):
    """
    Draw grouped bar chart for chosen elements and samples.

    element_cols: list[str] or str
    sample_filter: list[str] of labels to include (and order to enforce), or None for all
    date_fmt: strftime format for x-axis labels (default 'YYYY-MM-DD')
    """
    if isinstance(element_cols, str):
        element_cols = [element_cols]

    if "Label" not in df.columns:
        raise KeyError("Could not find a 'Label' column in the CSV.")

    # Subset to only the columns we need
    keep_cols = ["Label"] + element_cols
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing column(s) in CSV: {missing}")

    data = df[keep_cols].copy()

    # Convert requested element columns to numeric
    for col in element_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Drop rows where all selected elements are NaN
    data = data.dropna(subset=element_cols, how="all")

    # Apply sample filter (and enforce order exactly)
    if sample_filter:
        sample_filter = [str(s) for s in sample_filter]
        data["Label"] = data["Label"].astype(str)

        data = data[data["Label"].isin(sample_filter)]
        data = data.set_index("Label").reindex(sample_filter).reset_index()

    # Build date tick labels from Label
    data["Date"] = data["Label"].apply(label_to_date)
    tick_labels = np.where(
        data["Date"].notna(),
        data["Date"].dt.strftime(date_fmt),
        data["Label"].astype(str).values,  # fallback if parsing fails
    )

    # Grouped bar chart geometry
    n = len(data)
    if n == 0:
        raise ValueError("No rows to plot after filtering. Check your sample_filter and element columns.")

    x = np.arange(n)
    width = 0.8 / len(element_cols)

    plt.figure(figsize=(12, 6))
    for i, col in enumerate(element_cols):
        plt.bar(x + i * width, data[col].values, width, label=col)

    plt.xticks(x + width * (len(element_cols) - 1) / 2, tick_labels, rotation=90)
    plt.title("ICP-MS Field Samples")
    plt.xlabel("Date", fontsize=10)
    plt.ylabel("Concentration [ppb]", fontsize=10)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------
print("Available elements:", list_elements())
print("Available samples:", list_samples())

plot_elements(
    ['55Mn (KED) [ppb]', '60Ni (KED) [ppb]', '63Cu (KED) [ppb]', '66Zn (KED) [ppb]'],
    sample_filter=[
        'IBA_05032025', 'IBA_05042025', 'IBA_05052025', 'IBA_05082025',
        'IBA_05132025', 'IBA_05142025', 'IBA_05152025', 'IBA_05162025',
        'IBA_05172025', 'IBA_05182025', 'IBA_05192025', 'IBA_05202025',
        'IBA_05212025', 'IBA_05222025', 'IBA_05232025', 'IBA_05242025',
        'IBA_05252025', 'IBA_05262025', 'IBA_05272025', 'FB_05252025', 'MB_12162025'
    ],
    date_fmt="%Y-%m-%d"  # change to "%m/%d" or "%b %d" if you prefer
)

# Trace metals example:
# plot_elements(
#     ['51V (KED) [ppb]', '52Cr (KED) [ppb]', '59Co (KED) [ppb]',
#      '75As (KED) [ppb]', '77Se (KED) [ppb]', '111Cd (KED) [ppb]', '208Pb (KED) [ppb]'],
#     sample_filter=[...],
#     date_fmt="%m/%d"
# )
