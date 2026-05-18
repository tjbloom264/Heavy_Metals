# -*- coding: utf-8 -*-
"""
Water heavy metals monthly average bar plot

- Filters Type == Water
- Optional Location filter
- Uses raw ppb concentrations
- Averages selected metals by month
- Only shows months that contain samples
- Removes empty date gaps between sparse samples
- Adds n=x above each bar
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

METALS_TO_PLOT = [
    "66Zn (KED) [ppb]",
    "63Cu (KED) [ppb]",
    "60Ni (KED) [ppb]",
]

# Optional location filter
# Set to None to include all locations
LOCATION_FILTER = "STR"
# Example:
# LOCATION_FILTER = "STR"

START_DATE = None
END_DATE = None
# Example:
# START_DATE = "2025-09-15"
# END_DATE   = "2025-09-24"

REMOVE_NEGATIVES = True

SAVE_FIG = True
OUTPUT_FIG = r"d:\Plots\Metals\monthly_averagesbss.png"

USE_LOG_Y = False

FIGSIZE = (14, 7)

TITLE = "Monthly Average Water Heavy Metals"
TITLE_SIZE = 16
LABEL_SIZE = 13
TICK_SIZE = 11

GRID_ALPHA = 0.3

SHOW_N_LABELS = True
N_LABEL_SIZE = 9
N_LABEL_ROTATION = 0

# Adds a little space above the tallest bar so n labels fit
Y_TOP_PAD_FRACTION = 0.15

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(FILE)
df.columns = df.columns.str.strip()

for col in ["Date", "Time", "Type", "Location", "Label"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# =========================================================
# CHECK REQUIRED COLUMNS
# =========================================================

required_cols = ["Date", "Time", "Type"]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# =========================================================
# FILTER WATER ONLY
# =========================================================

df = df[df["Type"].str.lower() == "water"].copy()

if df.empty:
    raise ValueError("No rows found where Type == Water.")

# =========================================================
# OPTIONAL LOCATION FILTER
# =========================================================

if LOCATION_FILTER is not None:
    if "Location" not in df.columns:
        raise ValueError("LOCATION_FILTER was set, but no 'Location' column was found.")

    df = df[df["Location"].str.upper() == LOCATION_FILTER.upper()].copy()

    if df.empty:
        raise ValueError(f"No water samples found for LOCATION_FILTER = {LOCATION_FILTER}")

# =========================================================
# MAKE DATETIME
# =========================================================

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

df = df.dropna(subset=["DateTime"]).copy()
df = df.sort_values("DateTime")

if df.empty:
    raise ValueError("No valid DateTime values after parsing Date and Time columns.")

# =========================================================
# DATE RANGE FILTER
# =========================================================

if START_DATE is not None:
    df = df[df["DateTime"] >= pd.to_datetime(START_DATE)]

if END_DATE is not None:
    df = df[df["DateTime"] <= pd.to_datetime(END_DATE)]

if df.empty:
    raise ValueError("No data left after START_DATE / END_DATE filtering.")

# =========================================================
# CLEAN METAL COLUMNS
# =========================================================

available_metals = []

for metal in METALS_TO_PLOT:
    if metal not in df.columns:
        print(f"WARNING: Column not found and skipped: {metal}")
        continue

    df[metal] = (
        df[metal]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )

    df[metal] = pd.to_numeric(df[metal], errors="coerce")

    if REMOVE_NEGATIVES:
        df.loc[df[metal] < 0, metal] = np.nan

    available_metals.append(metal)

if len(available_metals) == 0:
    raise ValueError("None of the selected metal columns were found.")

# =========================================================
# MONTHLY AVERAGES + SAMPLE COUNTS
# =========================================================

df["Month"] = df["DateTime"].dt.to_period("M")

# Monthly averages
monthly = (
    df.groupby("Month")[available_metals]
    .mean()
    .reset_index()
)

# Monthly sample counts for each metal
# This counts only valid, non-NaN values
monthly_n = (
    df.groupby("Month")[available_metals]
    .count()
    .reset_index()
)

# Drop months where all selected metals are NaN
monthly = monthly.dropna(subset=available_metals, how="all").copy()

if monthly.empty:
    raise ValueError("No monthly averages available after cleaning/filtering.")

# Keep sample count rows matching plotted months
monthly_n = monthly_n[monthly_n["Month"].isin(monthly["Month"])].copy()

monthly["Month_Start"] = monthly["Month"].dt.to_timestamp()
monthly["Month_Label"] = monthly["Month_Start"].dt.strftime("%b %Y")

# =========================================================
# PRINT MONTHLY AVERAGES AND COUNTS
# =========================================================

print("\nMonthly average concentrations:")
print_cols = ["Month_Label"] + available_metals
print(monthly[print_cols].to_string(index=False))

print("\nMonthly sample counts:")
monthly_n_print = monthly_n.copy()
monthly_n_print["Month_Label"] = monthly_n_print["Month"].dt.to_timestamp().dt.strftime("%b %Y")
print_cols_n = ["Month_Label"] + available_metals
print(monthly_n_print[print_cols_n].to_string(index=False))

# =========================================================
# MONTHLY AVERAGE BAR PLOT
# =========================================================

fig, ax = plt.subplots(figsize=FIGSIZE)

n_metals = len(available_metals)

# Use category positions instead of real dates.
# This removes large blank spaces between sparse sample months.
x = np.arange(len(monthly))

bar_width = 0.8 / n_metals

all_bar_heights = []

for i, metal in enumerate(available_metals):
    offset = (i - n_metals / 2) * bar_width + bar_width / 2
    clean_label = metal.replace(" (KED) [ppb]", "")

    bars = ax.bar(
        x + offset,
        monthly[metal],
        width=bar_width,
        label=clean_label,
        align="center"
    )

    # Add n=x above each bar
    if SHOW_N_LABELS:
        for j, bar in enumerate(bars):
            height = bar.get_height()

            if np.isnan(height):
                continue

            all_bar_heights.append(height)

            month_value = monthly.loc[j, "Month"]

            n_value = monthly_n.loc[
                monthly_n["Month"] == month_value,
                metal
            ].values[0]

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"n={int(n_value)}",
                ha="center",
                va="bottom",
                fontsize=N_LABEL_SIZE,
                rotation=N_LABEL_ROTATION
            )

# =========================================================
# FORMATTING
# =========================================================

ax.set_title(TITLE, fontsize=TITLE_SIZE)
ax.set_xlabel("Month", fontsize=LABEL_SIZE)
ax.set_ylabel("Monthly Average Concentration (ppb)", fontsize=LABEL_SIZE)

if USE_LOG_Y:
    ax.set_yscale("log")
    ax.set_ylabel("Monthly Average Concentration (ppb, log scale)", fontsize=LABEL_SIZE)

# Only ticks for months that have samples
ax.set_xticks(x)
ax.set_xticklabels(
    monthly["Month_Label"],
    rotation=45,
    ha="right",
    fontsize=TICK_SIZE
)

ax.tick_params(axis="y", labelsize=TICK_SIZE)

# Add room above labels
if len(all_bar_heights) > 0 and not USE_LOG_Y:
    ymax = np.nanmax(all_bar_heights)
    ax.set_ylim(top=ymax * (1 + Y_TOP_PAD_FRACTION))

ax.grid(axis="y", alpha=GRID_ALPHA)

ax.legend(title="Metal", fontsize=10, ncol=2)

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