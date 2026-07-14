# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 12:52:27 2026

@author: Tobie
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# USER SETTINGS
# ============================================================
ICP_FILE  = r'd:\Data\Raw\ICP-MS\MasterMetal\ECAL_Bloom(Prather)_20260224.csv' 

# River flow file (CSV) and its columns
FLOW_FILE      = r'd:\Data\Raw\Flow\Flow_0914-0924.csv'  # <-- change to your real path
FLOW_DATE_COL  = "DateTime"                                # <-- change if needed
FLOW_VALUE_COL = "Flow"                                    # <-- change if needed
FLOW_UNITS     = "MG/DAY"                                     # e.g., cfs or m3/s

# If your flow data are higher frequency, resample for stable matching (daily/hourly/etc.)
FLOW_RESAMPLE_FREQ = "h"   # "D" daily, "H" hourly, None for no resample

# ============================================================
# MANUAL SAMPLE -> DATETIME MAP (EDIT IF NEEDED)
# Assumption: September 15–23, 2025; AM=09:15, PM=21:15
# ============================================================
sample_date_map = {
    
    '915PMS_air': "2025-09-15 21:15",

    "916amSTR": "2025-09-16 09:15",
    "916PMS_air": "2025-09-16 21:15",

    "917AMS_air": "2025-09-17 09:15",
    "917PMS_air": "2025-09-17 21:15",

    "918AMS_air": "2025-09-18 09:15",
    "918PMS_air": "2025-09-18 21:15",

}

# Choose which elements to plot (edit these to match your column names)
ELEMENTS_TO_PLOT = [
    '55Mn (KED) [ppb]', '60Ni (KED) [ppb]', '63Cu (KED) [ppb]', '66Zn (KED) [ppb]'
]

# Order of samples on the x-axis (edit as needed)
SAMPLE_ORDER = [
     '915PMS_air', '916PMS_air',
    '917AMS_air', '917PMS_air',
    '918AMS_air', '918PMS_air',
]

# ============================================================
# LOAD ICP-MS DATA
# ============================================================
df = pd.read_csv(ICP_FILE)
df = df.dropna(how="all").reset_index(drop=True)

if "Label" not in df.columns:
    raise ValueError("Expected a 'Label' column in the ICP-MS file but didn't find it.")

# Add manual SampleDate
df["Label"] = df["Label"].astype(str)
df["SampleDate"] = df["Label"].map(sample_date_map)
df["SampleDate"] = pd.to_datetime(df["SampleDate"], errors="coerce")

# Optional: warn if labels exist without a mapped date
missing_dates = df.loc[df["Label"].isin(SAMPLE_ORDER) & df["SampleDate"].isna(), "Label"].unique().tolist()
if missing_dates:
    print("⚠️ These samples are in SAMPLE_ORDER but missing from sample_date_map:")
    print(missing_dates)

# ============================================================
# LOAD FLOW DATA
# ============================================================
flow_df = pd.read_csv(FLOW_FILE)

if FLOW_DATE_COL not in flow_df.columns:
    raise ValueError(f"Flow file missing datetime column '{FLOW_DATE_COL}'. Columns: {flow_df.columns.tolist()}")
if FLOW_VALUE_COL not in flow_df.columns:
    raise ValueError(f"Flow file missing value column '{FLOW_VALUE_COL}'. Columns: {flow_df.columns.tolist()}")

flow_df[FLOW_DATE_COL] = pd.to_datetime(flow_df[FLOW_DATE_COL], errors="coerce")
flow_df[FLOW_VALUE_COL] = pd.to_numeric(flow_df[FLOW_VALUE_COL], errors="coerce")
flow_df = flow_df.dropna(subset=[FLOW_DATE_COL]).sort_values(FLOW_DATE_COL).set_index(FLOW_DATE_COL)

if FLOW_RESAMPLE_FREQ:
    flow_df = flow_df.resample(FLOW_RESAMPLE_FREQ).mean(numeric_only=True)

# ============================================================
# PLOTTING FUNCTION
# ============================================================
def plot_elements_with_flow(element_cols, sample_filter_order):
    # Subset ICP data
    cols_needed = ["Label", "SampleDate"] + element_cols
    missing_cols = [c for c in cols_needed if c not in df.columns]
    if missing_cols:
        raise ValueError(f"ICP file missing columns: {missing_cols}")

    data = df[cols_needed].copy()

    # Convert metals to numeric
    for col in element_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Keep only samples in filter order, enforce order
    data = data[data["Label"].isin(sample_filter_order)]
    data = data.set_index("Label").reindex(sample_filter_order).reset_index()

    # Grouped bars setup
    labels = data["Label"].astype(str).values
    x = np.arange(len(labels))
    width = 0.8 / max(len(element_cols), 1)

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Metals on left axis
    for i, col in enumerate(element_cols):
        ax1.bar(x + i * width, data[col].values, width, label=col)

    ax1.set_xticks(x + width * (len(element_cols) - 1) / 2)
    ax1.set_xticklabels(labels, rotation=90)
    ax1.set_title("ICP-MS Field Samples + River Flow")
    ax1.set_xlabel("Sample Label", fontsize=10)
    ax1.set_ylabel("Concentration [ppb]", fontsize=10)
    ax1.grid(True, axis="y", alpha=0.3)

    # Flow alignment:
    # If flow_df is daily ("D"), match using date floor("D").
    # If flow_df is hourly ("H"), match using floor("H").
    if FLOW_RESAMPLE_FREQ == "D":
        sample_keys = pd.to_datetime(data["SampleDate"]).dt.floor("D")
    elif FLOW_RESAMPLE_FREQ == "H":
        sample_keys = pd.to_datetime(data["SampleDate"]).dt.floor("H")
    elif FLOW_RESAMPLE_FREQ is None:
        # Nearest lookup (simple approach): take flow at nearest timestamp
        # We'll do nearest by reindexing with method='nearest' on the flow index.
        sample_keys = pd.to_datetime(data["SampleDate"])
    else:
        # Generic: floor to the resample freq if it's a pandas offset alias
        # (If this fails, just set FLOW_RESAMPLE_FREQ to "D" or "H".)
        sample_keys = pd.to_datetime(data["SampleDate"]).dt.floor(FLOW_RESAMPLE_FREQ)

    if FLOW_RESAMPLE_FREQ is None:
        flow_on_samples = flow_df[FLOW_VALUE_COL].reindex(sample_keys.values, method="nearest").values
    else:
        flow_on_samples = flow_df.reindex(sample_keys.values)[FLOW_VALUE_COL].values

    # Plot flow on secondary axis
    ax2 = ax1.twinx()
    x_center = x + width * (len(element_cols) - 1) / 2
    ax2.plot(x_center, flow_on_samples, marker="o", linewidth=2, alpha=0.8, label="River Flow")
    ax2.set_ylabel(f"River Flow ({FLOW_UNITS})", fontsize=10)

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=11, loc="upper right")

    plt.tight_layout()
    plt.show()


# ============================================================
# RUN
# ============================================================
plot_elements_with_flow(ELEMENTS_TO_PLOT, SAMPLE_ORDER)
