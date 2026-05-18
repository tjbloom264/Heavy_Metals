# -*- coding: utf-8 -*-
"""
Air metals statistics table from combined ICP-MS file

For air samples:
    extract concentration: ppb = ug/L
    final volume: L
    sampled air volume: L

Calculation:
    mass_ug = extract_ppb * final_volume_L
    mass_ng = mass_ug * 1000
    air_conc_ng_m3 = mass_ng / sampled_air_volume_m3
"""

import pandas as pd
import numpy as np
import re

# =========================================================
# FILE PATHS
# =========================================================

INPUT_FILE = r"d:\Data\Processed\ICPMS\combined_heavy_metals.csv"

OUTPUT_STATS_FILE = r"d:\Data\Processed\ICPMS\air_statistics_ng_m3.csv"
OUTPUT_AIR_CONC_FILE = r"d:\Data\Processed\ICPMS\air_concentrations_ng_m3.csv"

# =========================================================
# FILTER OPTIONS
# =========================================================

SAMPLE_TYPE = "Air"        # usually "Air"
LOCATION_FILTER = "STR"    # example: "SIO", "TJRE", or None

START_DATE = None          # example: "2025-09-01" or None
END_DATE = None            # example: "2025-09-30" or None

# =========================================================
# VOLUME COLUMNS
# =========================================================

SAMPLE_VOL_COL = "Sample Volume (L)"
FINAL_VOL_COL = "Final Volume (L)"

# Your current file appears to use:
# Sample Volume = 28,800 L
# Final Volume = 0.02 L
SAMPLE_VOLUME_UNITS = "L"
FINAL_VOLUME_UNITS = "L"

# Extra dilution factor, if any.
# Example: use 10 if sample was diluted 10x before ICP-MS.
DILUTION_FACTOR = 1

# =========================================================
# OPTIONS
# =========================================================

REMOVE_NEGATIVES = True
ROUND_DECIMALS = 3

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(INPUT_FILE)
df.columns = df.columns.astype(str).str.strip()

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def clean_numeric(series):
    """
    Converts messy numeric text to numbers.
    Handles commas like 28,800.
    """
    return pd.to_numeric(
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False),
        errors="coerce"
    )


def volume_to_liters(series, units):
    values = clean_numeric(series)

    if units.lower() == "ml":
        values = values / 1000
    elif units.lower() == "l":
        values = values
    else:
        raise ValueError("Volume units must be 'L' or 'mL'.")

    return values


def extract_element_name(col):
    """
    Examples:
        '55Mn (KED) [ppb]' -> 'Mn'
        '208Pb (KED) [ppb]' -> 'Pb'
        '232Th (KED) [ppm]' -> 'Th'
    """
    match = re.search(r"\d+([A-Z][a-z]?)", col)
    if match:
        return match.group(1)
    return col


def make_unique_names(names):
    """
    Prevents duplicate names, for example if Th appears in ppm and ppb columns.
    """
    seen = {}
    output = []

    for name in names:
        if name not in seen:
            seen[name] = 1
            output.append(name)
        else:
            seen[name] += 1
            output.append(f"{name}_{seen[name]}")

    return output


def convert_extract_to_ppb(series, colname):
    values = clean_numeric(series)

    if "[ppm]" in colname:
        values = values * 1000

    return values


# =========================================================
# DATE FILTERING
# =========================================================

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    if START_DATE is not None:
        df = df[df["Date"] >= pd.to_datetime(START_DATE)]

    if END_DATE is not None:
        df = df[df["Date"] <= pd.to_datetime(END_DATE)]

# =========================================================
# SAMPLE TYPE AND LOCATION FILTERING
# =========================================================

if SAMPLE_TYPE is not None:
    if "Type" not in df.columns:
        raise ValueError("No 'Type' column found.")

    df = df[
        df["Type"].astype(str).str.strip().str.lower()
        == SAMPLE_TYPE.lower()
    ].copy()

if LOCATION_FILTER is not None:
    if "Location" not in df.columns:
        raise ValueError("No 'Location' column found.")

    df = df[
        df["Location"].astype(str).str.strip().str.lower()
        == LOCATION_FILTER.lower()
    ].copy()

if df.empty:
    raise ValueError("No rows left after filtering.")

print(f"Rows after filtering: {len(df)}")

# =========================================================
# FIND ELEMENT COLUMNS
# =========================================================

element_cols = [
    col for col in df.columns
    if ("[ppb]" in col or "[ppm]" in col)
]

if len(element_cols) == 0:
    raise ValueError("No element columns with [ppb] or [ppm] found.")

raw_element_names = [extract_element_name(c) for c in element_cols]
element_names = make_unique_names(raw_element_names)

print("\nElement columns found:")
for col, name in zip(element_cols, element_names):
    print(f"  {col}  ->  {name}")

# =========================================================
# CHECK VOLUMES
# =========================================================

if SAMPLE_VOL_COL not in df.columns:
    raise ValueError(f"Missing column: {SAMPLE_VOL_COL}")

if FINAL_VOL_COL not in df.columns:
    raise ValueError(f"Missing column: {FINAL_VOL_COL}")

sample_volume_L = volume_to_liters(df[SAMPLE_VOL_COL], SAMPLE_VOLUME_UNITS)
final_volume_L = volume_to_liters(df[FINAL_VOL_COL], FINAL_VOLUME_UNITS)

sample_volume_m3 = sample_volume_L / 1000

print("\nVolume check:")
print(f"Sample volume valid values: {sample_volume_L.notna().sum()} / {len(sample_volume_L)}")
print(f"Final volume valid values:  {final_volume_L.notna().sum()} / {len(final_volume_L)}")

if sample_volume_L.notna().sum() == 0:
    raise ValueError("All sample volumes became NaN. Check Sample Volume column formatting.")

if final_volume_L.notna().sum() == 0:
    raise ValueError("All final volumes became NaN. Check Final Volume column formatting.")

# =========================================================
# CALCULATE AIR CONCENTRATIONS
# =========================================================

air_df = df.copy()
calculated_cols = []

for col, element in zip(element_cols, element_names):

    extract_ppb = convert_extract_to_ppb(air_df[col], col)

    # ppb = ug/L
    mass_ug = extract_ppb * final_volume_L * DILUTION_FACTOR
    mass_ng = mass_ug * 1000

    air_conc_ng_m3 = mass_ng / sample_volume_m3

    if REMOVE_NEGATIVES:
        air_conc_ng_m3 = air_conc_ng_m3.where(air_conc_ng_m3 >= 0, np.nan)

    out_col = f"{element}_ng_m3"
    air_df[out_col] = air_conc_ng_m3
    calculated_cols.append(out_col)

# =========================================================
# MAKE STATISTICS TABLE
# Elements as rows, stats as columns
# =========================================================

stats_rows = []

for col in calculated_cols:
    element = col.replace("_ng_m3", "")

    values = pd.to_numeric(air_df[col], errors="coerce").dropna()

    stats_rows.append({
        "Element": element,
        "n": values.count(),
        "mean": values.mean(),
        "median": values.median(),
        "std": values.std(),
        "min": values.min(),
        "q25": values.quantile(0.25),
        "q75": values.quantile(0.75),
        "max": values.max()
    })

stats_table = pd.DataFrame(stats_rows)

stats_table = stats_table.round(ROUND_DECIMALS)
# =========================================================
# SAVE OUTPUTS
# =========================================================

air_df.to_csv(OUTPUT_AIR_CONC_FILE, index=False)
stats_table.to_csv(OUTPUT_STATS_FILE)

# =========================================================
# PRINT RESULTS
# =========================================================

print("\nAir concentration file saved to:")
print(OUTPUT_AIR_CONC_FILE)

print("\nStatistics table saved to:")
print(OUTPUT_STATS_FILE)

print("\nStatistics table, units = ng/m3:")
print(stats_table.to_string())