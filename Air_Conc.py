# -*- coding: utf-8 -*-
"""
Calculate air-sampled volume from explicit start/end times,
then calculate metal air concentrations from ICP-MS extraction data.

Designed for ICP-MS columns like:
    23Na (KED) [ppb]
    66Zn (KED) [ppb]
    111Cd (KED) [ppb]

Assumptions:
- ICP-MS solution concentration [ppb] is treated as ug/L
- dissolved/extraction volume is in mL
- air flow is in L/min or m3/min
- air concentration output is ug/m3 or ng/m3
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# USER SETTINGS
# =========================================================

# ---------- FILE PATHS ----------
sampling_file = r"e:\Data\Raw\ICP-MS\Filter Times  .csv"
metals_file   = r"e:\Data\Raw\ICP-MS\Air_samples_Raw.csv"
output_file_full  = r"E:\Data\Processed\filter_metals_air_concentrations_FULL.csv"
output_file_table = r"E:\Data\Processed\filter_metals_air_concentrations_TABLE.csv"

# ---------- JOIN COLUMN ----------
sample_id_col = "SampleID"

# ---------- SAMPLING CSV COLUMNS ----------
start_date_col = "StartDate"
start_time_col = "StartTime"
end_date_col   = "EndDate"
end_time_col   = "EndTime"
flow_col       = "FlowRate"

# ---------- FLOW UNITS ----------
flow_units = "L/min"   # "L/min" or "m3/min"

# ---------- DISSOLVED / EXTRACTION VOLUME ----------
dissolved_vol_col = "DissolvedVolume_mL"
use_constant_dissolved_volume = True
constant_dissolved_volume_mL = 20.0

# ---------- DATETIME FORMAT ----------
datetime_format = None

# ---------- AIR CONCENTRATION OUTPUT ----------
air_conc_output_units = "ng/m3"   # "ug/m3" or "ng/m3"

# ---------- PLOTTING ----------
make_grouped_plot = True
make_individual_sample_plots = True
make_table_figure = True

rotate_xticks = 90
grouped_figsize = (16, 8)
individual_figsize = (12, 6)

# ---------- OPTIONAL SAMPLE ORDER ----------
sample_order = ["917amSTR","917pmSTR"] 

# ---------- METALS TO PLOT ----------
# Use clean element symbols, not full ICP column names.
# Examples:
# selected_metals_to_plot = ["Na", "Zn", "Cd", "Pb"]
# selected_metals_to_plot = None   # plots all detected metals
selected_metals_to_plot =[ "Zn", "Mn", "Ni","Cu","Pb","Cd","As","Cr","Co","V","Se","Sb","Sn","Ba"]

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def build_datetime(df, date_col, time_col, out_col, dt_format=None):
    dt_str = (
        df[date_col].astype(str).str.strip() + " " +
        df[time_col].astype(str).str.strip()
    )
    df[out_col] = pd.to_datetime(dt_str, format=dt_format, errors="coerce")
    return df

def convert_flow_to_m3_per_min(series, units):
    if units == "L/min":
        return series / 1000.0
    elif units == "m3/min":
        return series
    else:
        raise ValueError("flow_units must be 'L/min' or 'm3/min'")

def extract_element_label(col_name):
    match = re.search(r'^\s*\d*\s*([A-Z][a-z]?)', str(col_name))
    if match:
        return match.group(1)
    return str(col_name)

def auto_detect_icp_columns(df, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = set()

    detected = []
    for col in df.columns:
        if col in exclude_cols:
            continue

        col_str = str(col)

        looks_like_icp = (
            ("[ppb]" in col_str.lower()) or
            ("[ppm]" in col_str.lower()) or
            ("(" in col_str and ")" in col_str)
        )

        if not looks_like_icp:
            continue

        numeric_test = pd.to_numeric(df[col], errors="coerce")
        if numeric_test.notna().any():
            detected.append(col)

    return detected

def mass_ug_from_ppb_ugL(conc_ugL, vol_mL):
    vol_L = vol_mL / 1000.0
    return conc_ugL * vol_L

def make_unique_labels(labels):
    counts = {}
    out = []
    for label in labels:
        if label not in counts:
            counts[label] = 1
            out.append(label)
        else:
            counts[label] += 1
            out.append(f"{label}_{counts[label]}")
    return out

# =========================================================
# LOAD SAMPLING DATA
# =========================================================

sampling_df = pd.read_csv(sampling_file)
sampling_df.columns = sampling_df.columns.str.strip()
sampling_df = sampling_df.dropna(how="all").reset_index(drop=True)

sampling_df[sample_id_col] = sampling_df[sample_id_col].astype(str).str.strip()
sampling_df[flow_col] = pd.to_numeric(sampling_df[flow_col].astype(str).str.strip(), errors="coerce")

sampling_df = build_datetime(sampling_df, start_date_col, start_time_col, "StartDateTime", datetime_format)
sampling_df = build_datetime(sampling_df, end_date_col, end_time_col, "EndDateTime", datetime_format)

sampling_df["SampleTime_min"] = (
    sampling_df["EndDateTime"] - sampling_df["StartDateTime"]
).dt.total_seconds() / 60.0

sampling_df.loc[sampling_df["SampleTime_min"] <= 0, "SampleTime_min"] = np.nan

sampling_df["Flow_m3_per_min"] = convert_flow_to_m3_per_min(sampling_df[flow_col], flow_units)
sampling_df["AirVolume_m3"] = sampling_df["SampleTime_min"] * sampling_df["Flow_m3_per_min"]
sampling_df["AirVolume_L"] = sampling_df["AirVolume_m3"] * 1000.0

sampling_keep = [
    sample_id_col,
    "StartDateTime",
    "EndDateTime",
    "SampleTime_min",
    flow_col,
    "Flow_m3_per_min",
    "AirVolume_m3",
    "AirVolume_L"
]
sampling_df = sampling_df[sampling_keep].copy()

# =========================================================
# LOAD METALS DATA
# =========================================================

metals_df = pd.read_csv(metals_file)
metals_df.columns = metals_df.columns.str.strip()
metals_df = metals_df.dropna(how="all").reset_index(drop=True)

metals_df[sample_id_col] = metals_df[sample_id_col].astype(str).str.strip()

if use_constant_dissolved_volume:
    metals_df["DissolvedVolume_mL_used"] = constant_dissolved_volume_mL
else:
    metals_df[dissolved_vol_col] = pd.to_numeric(
        metals_df[dissolved_vol_col].astype(str).str.strip(),
        errors="coerce"
    )
    metals_df["DissolvedVolume_mL_used"] = metals_df[dissolved_vol_col]

exclude_cols = {
    sample_id_col,
    dissolved_vol_col,
    "DissolvedVolume_mL_used"
}

metal_columns = auto_detect_icp_columns(metals_df, exclude_cols=exclude_cols)

if len(metal_columns) == 0:
    raise ValueError("No ICP-MS metal columns were detected.")

for col in metal_columns:
    metals_df[col] = pd.to_numeric(
        metals_df[col].astype(str).str.strip(),
        errors="coerce"
    )

# =========================================================
# MERGE
# =========================================================

df = pd.merge(
    metals_df,
    sampling_df,
    on=sample_id_col,
    how="inner"
)

if sample_order is not None:
    df["_sample_order"] = pd.Categorical(df[sample_id_col], categories=sample_order, ordered=True)
    df = df.sort_values("_sample_order").drop(columns="_sample_order").reset_index(drop=True)

# =========================================================
# CALCULATE METAL MASS + AIR CONCENTRATIONS
# =========================================================

metal_short_labels = [extract_element_label(col) for col in metal_columns]
metal_short_labels = make_unique_labels(metal_short_labels)

final_air_cols = []
mass_cols = []

for metal_col, short_label in zip(metal_columns, metal_short_labels):
    mass_col = f"{short_label}_mass_ug"
    ugm3_col = f"{short_label}_air_ug_m3"

    df[mass_col] = mass_ug_from_ppb_ugL(
        conc_ugL=df[metal_col],
        vol_mL=df["DissolvedVolume_mL_used"]
    )

    df[ugm3_col] = df[mass_col] / df["AirVolume_m3"]

    if air_conc_output_units == "ug/m3":
        final_col = ugm3_col
    elif air_conc_output_units == "ng/m3":
        final_col = f"{short_label}_air_ng_m3"
        df[final_col] = df[ugm3_col] * 1000.0
    else:
        raise ValueError("air_conc_output_units must be 'ug/m3' or 'ng/m3'")

    mass_cols.append(mass_col)
    final_air_cols.append(final_col)

# =========================================================
# BUILD SUMMARY TABLE
# =========================================================

summary_cols = [
    sample_id_col,
    "StartDateTime",
    "EndDateTime",
    "SampleTime_min",
    "AirVolume_m3"
] + final_air_cols

table_df = df[summary_cols].copy()

for col in table_df.columns:
    if pd.api.types.is_numeric_dtype(table_df[col]):
        table_df[col] = table_df[col].round(4)

# =========================================================
# SAVE OUTPUTS
# =========================================================

df.to_csv(output_file_full, index=False)
table_df.to_csv(output_file_table, index=False)

print("\nDone.\n")
print("Detected metals:")
for orig, short in zip(metal_columns, metal_short_labels):
    print(f"  {orig}  -->  {short}")

print(f"\nSaved full output to:\n{output_file_full}")
print(f"Saved summary table to:\n{output_file_table}")

print("\nSummary table:")
print(table_df.to_string(index=False))

# =========================================================
# PREP FOR PLOTTING
# =========================================================

plot_df = table_df.set_index(sample_id_col)[final_air_cols].copy()

rename_map = {}
for final_col, short_label in zip(final_air_cols, metal_short_labels):
    rename_map[final_col] = short_label

plot_df = plot_df.rename(columns=rename_map)

# =========================================================
# FILTER METALS TO PLOT
# =========================================================

if selected_metals_to_plot is not None:
    missing_metals = [m for m in selected_metals_to_plot if m not in plot_df.columns]
    if missing_metals:
        print("\nWarning: these selected metals were not found and will be skipped:")
        print(missing_metals)

    metals_found = [m for m in selected_metals_to_plot if m in plot_df.columns]

    if len(metals_found) == 0:
        raise ValueError("None of the selected_metals_to_plot were found in detected metals.")

    plot_df = plot_df[metals_found]

# =========================================================
# GROUPED BAR PLOT
# =========================================================

if make_grouped_plot:
    ax = plot_df.plot(kind="bar", figsize=grouped_figsize)
    ax.set_xlabel("Sample")
    ax.set_ylabel(f"Air Concentration ({air_conc_output_units})")
    ax.set_title("Metal Air Concentrations by Sample")
    plt.xticks(rotation=rotate_xticks)
    plt.tight_layout()
    plt.show()

# =========================================================
# INDIVIDUAL SAMPLE BAR PLOTS
# =========================================================

if make_individual_sample_plots:
    for sample_name, row in plot_df.iterrows():
        plt.figure(figsize=individual_figsize)
        plt.bar(plot_df.columns, row.values)
        plt.xlabel("Metal")
        plt.ylabel(f"Air Concentration ({air_conc_output_units})")
        plt.title(f"Metal Air Concentrations - {sample_name}")
        plt.xticks(rotation=rotate_xticks)
        plt.tight_layout()
        plt.show()