# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =========================================================
# FILES
# =========================================================

DATA_FILE = r"E:\Data\Processed\ICPMS\combined_heavy_metals.csv"
FLOW_FILE = r"E:\Data\Raw\Flow\DataSetExport-Discharge.Instantaneous Flow.MGD@11013300-Aggregate-M US Gal d-20260430074734.csv"

# =========================================================
# USER CONTROLS
# =========================================================

SAMPLE_TYPE = "Air"     # "Air", "Water", "Foam"

LOCATION_FILTER = "STR"  
# Example:
# "River"
# "Beach"
# "Outfall"
# None  → keep all locations

ELEMENT_TO_PLOT = "66Zn (KED) [ppb]"

START_DATE = "2025-09-14"
END_DATE   = "2025-09-24"

PLOT_FLOW = True

# =========================================================
# COLUMN NAMES
# =========================================================

DATE_COL = "Date"
TIME_COL = "Time"
TYPE_COL = "Type"
LOCATION_COL = "Location"

SAMPLE_VOL_COL = "Sample Volume (L)"
FINAL_VOL_COL = "Final Volume (L)"

FLOW_DATETIME_COL = "DateTime"
FLOW_VALUE_COL = "Flow"

# =========================================================
# HELPERS
# =========================================================

def clean_number(series):
    return pd.to_numeric(
        series.astype(str)
              .str.replace(",", "", regex=False)
              .str.strip(),
        errors="coerce"
    )

def make_datetime(df):

    time_map = {
        "AM": "09:00",
        "PM": "21:00",
        "24hr": "12:00",
        "24HR": "12:00",
        "24 Hr": "12:00",
        "24 hr": "12:00"
    }

    df["ClockTime"] = (
        df[TIME_COL]
        .astype(str)
        .str.strip()
        .map(time_map)
    )

    df["DateTime"] = pd.to_datetime(
        df[DATE_COL].astype(str).str.strip() + " " +
        df["ClockTime"],
        errors="coerce"
    )

    return df

def apply_date_range(df):

    if START_DATE is not None:
        df = df[df["DateTime"] >= pd.to_datetime(START_DATE)]

    if END_DATE is not None:
        df = df[df["DateTime"] <= pd.to_datetime(END_DATE)]

    return df

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()

# Filter sample type
df[TYPE_COL] = df[TYPE_COL].astype(str).str.strip()

df = df[
    df[TYPE_COL].str.lower()
    == SAMPLE_TYPE.lower()
].copy()

# =========================================================
# LOCATION FILTER
# =========================================================

if LOCATION_FILTER is not None:

    df[LOCATION_COL] = df[LOCATION_COL].astype(str).str.strip()

    df = df[
        df[LOCATION_COL].str.lower()
        == LOCATION_FILTER.lower()
    ].copy()

# =========================================================
# DATETIME
# =========================================================

df = make_datetime(df)

# Clean element values
df[ELEMENT_TO_PLOT] = clean_number(df[ELEMENT_TO_PLOT])

# =========================================================
# CALCULATIONS
# =========================================================

if SAMPLE_TYPE.lower() == "air":

    df[SAMPLE_VOL_COL] = clean_number(df[SAMPLE_VOL_COL])
    df[FINAL_VOL_COL] = clean_number(df[FINAL_VOL_COL])

    df["AirVolume_m3"] = df[SAMPLE_VOL_COL] / 1000.0

    df["Mass_ng"] = (
        df[ELEMENT_TO_PLOT]
        * df[FINAL_VOL_COL]
        * 1000.0
    )

    df["ValueToPlot"] = (
        df["Mass_ng"]
        / df["AirVolume_m3"]
    )

    y_label = "Air concentration (ng/m³)"

else:

    df["ValueToPlot"] = df[ELEMENT_TO_PLOT]

    y_label = "Concentration (ppb)"

# Remove bad rows
df = df.dropna(subset=["DateTime", "ValueToPlot"])

# Apply date range
df = apply_date_range(df)

df = df.sort_values("DateTime")

print("Sample Type:", SAMPLE_TYPE)
print("Location:", LOCATION_FILTER)
print("Rows plotted:", len(df))

# =========================================================
# LOAD FLOW
# =========================================================

flow = None

if PLOT_FLOW:

    flow = pd.read_csv(FLOW_FILE)
    flow.columns = flow.columns.str.strip()

    flow["DateTime"] = pd.to_datetime(
        flow[FLOW_DATETIME_COL],
        errors="coerce"
    )

    flow[FLOW_VALUE_COL] = clean_number(
        flow[FLOW_VALUE_COL]
    )

    flow = flow.dropna(
        subset=["DateTime", FLOW_VALUE_COL]
    )

    flow = apply_date_range(flow)

    flow = flow.sort_values("DateTime")

    # Smooth flow
    flow = (
        flow.set_index("DateTime")
            .resample("1h")
            .mean(numeric_only=True)
            .reset_index()
    )

# =========================================================
# PLOT
# =========================================================

fig, ax1 = plt.subplots(figsize=(13,6))

ax1.plot(
    df["DateTime"],
    df["ValueToPlot"],
    marker="o",
    linewidth=2.5,
    color="darkorange",
    label=ELEMENT_TO_PLOT.replace(" (KED) [ppb]", "")
)

ax1.set_xlabel("Date")
ax1.set_ylabel(y_label)

title = f"{ELEMENT_TO_PLOT} — {SAMPLE_TYPE}"

if LOCATION_FILTER is not None:
    title += f" ({LOCATION_FILTER})"

ax1.set_title(title)

ax1.xaxis.set_major_formatter(
    mdates.DateFormatter("%Y-%m-%d")
)

plt.setp(
    ax1.get_xticklabels(),
    rotation=45,
    ha="right"
)

ax1.grid(True, alpha=0.3)

# Flow overlay
if PLOT_FLOW and flow is not None:

    ax2 = ax1.twinx()

    ax2.plot(
        flow["DateTime"],
        flow[FLOW_VALUE_COL],
        color="black",
        linewidth=2,
        alpha=0.75,
        label="Flow"
    )

    ax2.set_ylabel("Flow")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="best"
    )

else:

    ax1.legend(loc="best")

plt.tight_layout()
plt.show()