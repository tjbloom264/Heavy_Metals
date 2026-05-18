# -*- coding: utf-8 -*-
"""
Single sample bar plot for selected elements

- Does NOT filter out any sample types
- Selects one sample from the full file
- Lists available samples
- Lists available ICP-MS element columns
- Lets you choose which elements to plot
- Applies a dilution factor to concentrations
- Writes corrected concentration value above each bar
- Uses different colors for each bar
- Uses log-scale y-axis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# FILE PATH
# =========================================================

FILE = r"d:\Data\Processed\ICPMS\combined_heavy_metals.csv"

# =========================================================
# USER SETTINGS
# =========================================================

# ---------------------------------------------------------
# Choose one sample
# ---------------------------------------------------------

# Option 1: choose sample by exact Label
SAMPLE_LABEL = "FoamD"
# Example:
# SAMPLE_LABEL = "IBW_09152025_AM"

# Option 2: choose sample by row number after loading the file
# Used only if SAMPLE_LABEL = None
SAMPLE_INDEX = None
# 0 = first sample
# 1 = second sample
# 2 = third sample, etc.

# ---------------------------------------------------------
# Dilution correction
# ---------------------------------------------------------

# Multiplies each plotted concentration by this value.
# Example:
# DILUTION_FACTOR = 2 means corrected concentration = measured concentration * 2
# DILUTION_FACTOR = 10 means corrected concentration = measured concentration * 10
DILUTION_FACTOR = 160.0

# ---------------------------------------------------------
# Optional filters
# ---------------------------------------------------------

# Leave as None to keep everything.
TYPE_FILTER = None
# Example:
# TYPE_FILTER = "Water"
# TYPE_FILTER = "Air"
# TYPE_FILTER = "Foam"

LOCATION_FILTER = None
# Example:
# LOCATION_FILTER = "STR"

START_DATE = None
END_DATE = None
# Example:
# START_DATE = "2025-09-15"
# END_DATE   = "2025-09-24"

# ---------------------------------------------------------
# Choose elements to plot
# ---------------------------------------------------------

# If True, plot every numeric ICP-MS element column
PLOT_ALL_ELEMENTS = False

# If PLOT_ALL_ELEMENTS = False, choose by printed element numbers
ELEMENT_NUMBERS_TO_PLOT = [
    0,
    1,
    2,
]
# Example:
# ELEMENT_NUMBERS_TO_PLOT = [8, 10, 15, 17]

# Or choose by exact column names.
# If this list has anything in it, it overrides ELEMENT_NUMBERS_TO_PLOT.
ELEMENT_COLUMNS_TO_PLOT = [
    "27Al (KED) [ppb]",
    "55Mn (KED) [ppb]",
    "57Fe (KED) [ppb]",
    "60Ni (KED) [ppb]",
    "63Cu (KED) [ppb]",
    "66Zn (KED) [ppb]",
    "75As (KED) [ppb]",
    "111Cd (KED) [ppb]",
    "208Pb (KED) [ppb]"
]

# ---------------------------------------------------------
# Plot settings
# ---------------------------------------------------------

SAVE_FIG = False
OUTPUT_FIG = r"E:\Data\Processed\ICPMS\single_sample_selected_elements_log_bar.png"

REMOVE_ZERO_AND_NEGATIVE = True
# Log scale cannot plot zero or negative values.
# If True, zero/negative element values are removed from the plot.

SORT_BY_CONCENTRATION = False
# True = highest concentration on left
# False = keeps file column order

FIGSIZE = (15, 7)

# Value labels above bars
VALUE_LABELS = True
VALUE_LABEL_SIZE = 9
VALUE_LABEL_ROTATION = 45

# Use ".2f" to avoid scientific notation
# ".1f" = 1 decimal
# ".2f" = 2 decimals
# ".3f" = 3 decimals
VALUE_LABEL_FORMAT = ".2f"

# Different colors for bars
BAR_COLORMAP = "tab10"
# Other options:
# "tab20", "Set2", "Dark2", "viridis", "plasma"

# Extra space above tallest bar for labels on log scale
LOG_Y_TOP_MULTIPLIER = 4

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(FILE)
df.columns = df.columns.str.strip()

# Clean common text columns if present
for col in ["Date", "Time", "Type", "Location", "Label"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# =========================================================
# OPTIONAL FILTERS
# =========================================================

if TYPE_FILTER is not None:
    if "Type" not in df.columns:
        raise ValueError("TYPE_FILTER was set, but the file has no 'Type' column.")

    df = df[df["Type"].str.lower() == TYPE_FILTER.lower()].copy()

if LOCATION_FILTER is not None:
    if "Location" not in df.columns:
        raise ValueError("LOCATION_FILTER was set, but the file has no 'Location' column.")

    df = df[df["Location"].str.upper() == LOCATION_FILTER.upper()].copy()

# =========================================================
# MAKE DATETIME FOR OPTIONAL DATE FILTERING
# =========================================================

if "Date" in df.columns and "Time" in df.columns:
    TIME_MAP = {
        "AM": "09:00:00",
        "PM": "21:00:00",
        "24HR": "12:00:00",
        "24H": "12:00:00",
        "24": "12:00:00",
    }

    def clean_time(value):
        value = str(value).strip().upper()

        if value in TIME_MAP:
            return TIME_MAP[value]

        if value in ["", "NAN", "NONE"]:
            return "12:00:00"

        return value

    df["Time_clean"] = df["Time"].apply(clean_time)

    df["DateTime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time_clean"],
        errors="coerce"
    )

    df = df.sort_values("DateTime", na_position="last").reset_index(drop=True)

    if START_DATE is not None:
        df = df[df["DateTime"] >= pd.to_datetime(START_DATE)].copy()

    if END_DATE is not None:
        df = df[df["DateTime"] <= pd.to_datetime(END_DATE)].copy()

    df = df.reset_index(drop=True)

else:
    print("WARNING: Date and/or Time columns not found. Date filtering will be skipped.")

if df.empty:
    raise ValueError("No samples left after optional filtering.")

# =========================================================
# LIST ALL SAMPLES
# =========================================================

print("\nAvailable samples:")
print("-" * 110)

for i, row in df.iterrows():
    label = row["Label"] if "Label" in df.columns else "No Label"
    sample_type = row["Type"] if "Type" in df.columns else "No Type"
    location = row["Location"] if "Location" in df.columns else "No Location"
    date = row["Date"] if "Date" in df.columns else "No Date"
    time = row["Time"] if "Time" in df.columns else "No Time"

    print(
        f"{i:02d}: "
        f"Label={label} | "
        f"Type={sample_type} | "
        f"Location={location} | "
        f"Date={date} | "
        f"Time={time}"
    )

print("-" * 110)

# =========================================================
# SELECT ONE SAMPLE
# =========================================================

if SAMPLE_LABEL is not None:
    if "Label" not in df.columns:
        raise ValueError("SAMPLE_LABEL was set, but the file has no 'Label' column.")

    sample_df = df[df["Label"] == SAMPLE_LABEL].copy()

    if sample_df.empty:
        raise ValueError(f"No sample found with Label = {SAMPLE_LABEL}")

    sample = sample_df.iloc[0]

else:
    if SAMPLE_INDEX is None:
        raise ValueError("Set either SAMPLE_LABEL or SAMPLE_INDEX.")

    if SAMPLE_INDEX >= len(df):
        raise ValueError(
            f"SAMPLE_INDEX {SAMPLE_INDEX} is too large. "
            f"There are only {len(df)} samples after optional filtering."
        )

    sample = df.iloc[SAMPLE_INDEX]

# =========================================================
# FIND AVAILABLE ELEMENT COLUMNS
# =========================================================

element_cols = []

for col in df.columns:
    col_clean = col.strip()

    if "(KED)" in col_clean:
        test_vals = (
            df[col_clean]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.strip()
        )

        test_vals = pd.to_numeric(test_vals, errors="coerce")

        if test_vals.notna().any():
            element_cols.append(col_clean)

element_cols = list(dict.fromkeys(element_cols))

if len(element_cols) == 0:
    raise ValueError("No numeric ICP-MS element columns found.")

print("\nAvailable element columns:")
print("-" * 110)

for i, col in enumerate(element_cols):
    clean_name = (
        col.replace(" (KED) [ppb]", "")
           .replace(" (KED) [ppm]", "")
           .replace(" (KED)", "")
    )

    print(f"{i:02d}: {clean_name:<10} --> {col}")

print("-" * 110)

# =========================================================
# CHOOSE ELEMENTS TO PLOT
# =========================================================

if PLOT_ALL_ELEMENTS:
    selected_element_cols = element_cols

elif len(ELEMENT_COLUMNS_TO_PLOT) > 0:
    missing_elements = [
        col for col in ELEMENT_COLUMNS_TO_PLOT
        if col not in element_cols
    ]

    if len(missing_elements) > 0:
        print("\nWARNING: These requested columns were not found and will be skipped:")
        for col in missing_elements:
            print(f"  {col}")

    selected_element_cols = [
        col for col in ELEMENT_COLUMNS_TO_PLOT
        if col in element_cols
    ]

else:
    selected_element_cols = []

    for i in ELEMENT_NUMBERS_TO_PLOT:
        if i < len(element_cols):
            selected_element_cols.append(element_cols[i])
        else:
            print(f"WARNING: Element number {i} is out of range and skipped.")

if len(selected_element_cols) == 0:
    raise ValueError("No elements selected for plotting.")

print("\nElements selected for plot:")
print("-" * 110)

for col in selected_element_cols:
    print(f"  {col}")

print("-" * 110)

# =========================================================
# EXTRACT VALUES FOR THE SELECTED SAMPLE
# =========================================================

values = []

for col in selected_element_cols:
    raw_value = sample[col]

    value = (
        str(raw_value)
        .replace(",", "")
        .replace("%", "")
        .strip()
    )

    value = pd.to_numeric(value, errors="coerce")

    if pd.isna(value):
        continue

    # Apply dilution correction
    value_corrected = value * DILUTION_FACTOR

    if REMOVE_ZERO_AND_NEGATIVE and value_corrected <= 0:
        continue

    clean_label = (
        col.replace(" (KED) [ppb]", "")
           .replace(" (KED) [ppm]", "")
           .replace(" (KED)", "")
    )

    if "[ppm]" in col:
        unit = "ppm"
    elif "[ppb]" in col:
        unit = "ppb"
    else:
        unit = "unknown"

    values.append({
        "Element": clean_label,
        "Column": col,
        "Measured_Concentration": value,
        "Dilution_Factor": DILUTION_FACTOR,
        "Corrected_Concentration": value_corrected,
        "Unit": unit
    })

if len(values) == 0:
    raise ValueError(
        "No positive numeric element values found for this selected sample. "
        "Log scale cannot plot zero or negative values."
    )

plot_df = pd.DataFrame(values)

if SORT_BY_CONCENTRATION:
    plot_df = plot_df.sort_values("Corrected_Concentration", ascending=False)

# =========================================================
# SAMPLE TITLE INFO
# =========================================================

sample_label = sample["Label"] if "Label" in df.columns else "Unknown sample"
sample_type = sample["Type"] if "Type" in df.columns else "Unknown type"
sample_date = sample["Date"] if "Date" in df.columns else ""
sample_time = sample["Time"] if "Time" in df.columns else ""
sample_location = sample["Location"] if "Location" in df.columns else ""

#title = (
#    f"Selected Elements in One Sample\n"
 #   f"{sample_label} | {sample_type} | {sample_location} | {sample_date} {sample_time}\n"
 #   f"Dilution factor = {DILUTION_FACTOR:g}"
#)

# =========================================================
# BAR PLOT
# =========================================================

fig, ax = plt.subplots(figsize=FIGSIZE)

# Different colors for each bar
cmap = plt.get_cmap(BAR_COLORMAP)
bar_colors = [cmap(i % cmap.N) for i in range(len(plot_df))]

bars = ax.bar(
    plot_df["Element"],
    plot_df["Corrected_Concentration"],
    color=bar_colors
)

ax.set_yscale("log")

# Add room above the tallest bar so value labels fit
ymax = plot_df["Corrected_Concentration"].max()
ax.set_ylim(top=ymax * LOG_Y_TOP_MULTIPLIER)

# Write value above each bar, not in scientific notation
if VALUE_LABELS:
    for bar, value in zip(
        bars,
        plot_df["Corrected_Concentration"]
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:{VALUE_LABEL_FORMAT}}",
            ha="center",
            va="bottom",
            fontsize=VALUE_LABEL_SIZE,
            rotation=VALUE_LABEL_ROTATION
        )

#ax.set_title(title, fontsize=15)
ax.set_xlabel("Element", fontsize=13)
ax.set_ylabel("Dilution-Corrected Concentration, log scale", fontsize=13)

ax.grid(axis="y", alpha=0.3, which="both")

plt.xticks(rotation=60, ha="right")
plt.tight_layout()

# =========================================================
# SAVE / SHOW
# =========================================================

if SAVE_FIG:
    output_path = Path(OUTPUT_FIG)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to: {output_path}")

plt.show()

# =========================================================
# PRINT SAMPLE AND VALUES
# =========================================================

print("\nSample plotted:")
print("-" * 110)
print(f"Label:           {sample_label}")
print(f"Type:            {sample_type}")
print(f"Location:        {sample_location}")
print(f"Date:            {sample_date}")
print(f"Time:            {sample_time}")
print(f"Dilution factor: {DILUTION_FACTOR:g}")
print("-" * 110)

print("\nValues plotted:")
print(plot_df.to_string(index=False))