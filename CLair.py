"""
air_metals_timeseries.py
========================================================================
Parse ICP-MS heavy-metal data from air-filter samples and plot a time
series of selected metals, corrected for dilution and per volume of air.

Units of the final plot: ng metal / m^3 air.

The workbook has an unusual, block-based layout: each sample is a block
that starts with a metadata row (Date / AM_PM / Location / Volume Air),
followed by 5 replicate ICP-MS reads, then "Mean:", "RSD [%]:" and "SD:"
rows. This script reads the "Mean:" row of each block as that sample's
concentration.

--------------------------------------------------------------------
THE CONVERSION
--------------------------------------------------------------------
Aqueous concentration in ppb == ug/L == ng/mL. So for each metal:

    ng/m^3 = C_measured[ppb]  x  DILUTION_FACTOR  x  EXTRACT_VOLUME_mL
             --------------------------------------------------------
                              AIR_VOLUME_m3

  * C_measured        : ICP-MS output, ppb (= ng/mL)
  * DILUTION_FACTOR   : per-sample, read from a "Dilution" column in the
                        sheet if present; otherwise DEFAULT_DILUTION_FACTOR
                        (2, for a 1:1 dilution with internal standard)
  * EXTRACT_VOLUME_mL : filter extracted into 20 mL           -> 20
  * AIR_VOLUME_m3     : volume of air pulled through the filter (YOU supply)

--------------------------------------------------------------------
AIR VOLUME (IMPORTANT)
--------------------------------------------------------------------
The "Volume Air" column in the supplied workbook is EMPTY. Air volume is
resolved per sample in this priority order:

  1. The "Volume Air" column in the Excel file (m^3), if filled, else
  2. The AIR_VOLUMES dict below, keyed by sample label, if provided, else
  3. DEFAULT_AIR_VOLUME_m3 (currently 288 m^3).

So everything plots out of the box on the 288 m^3 default; fill in real
per-sample volumes as they become available.
========================================================================
"""

import re
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ======================================================================
# USER SETTINGS  --  edit these
# ======================================================================

INPUT_FILE = "d:\Data\Raw\ICP-MS\Air_Metal\Metal_air_labeled.xlsx"     # path to the workbook

DEFAULT_DILUTION_FACTOR = 2.0   # used when a sample has no value in the
                                # dilution column (1:1 with IS = 2x)
EXTRACT_VOLUME_mL = 20.0        # filter extracted into 20 mL

# Default air volume (m^3) used for any sample that has no volume in the
# Excel sheet or the AIR_VOLUMES dict below. Set per-sample values later
# to override this.
DEFAULT_AIR_VOLUME_m3 = 288.0

# Which metals to plot. Use element symbols (case-insensitive), with or
# without mass number: "Pb", "208Pb", "Zn", "Cu" all work.
METALS_TO_PLOT = [ "Zn"]

# Optional: restrict to one or more sampling locations (e.g. ["SIO"]).
# Leave as None to include every location.
LOCATIONS = None

# Air sample volumes in m^3, keyed by sample Label.
# Fill these in (or fill the "Volume Air" column in the Excel instead).
# The dict is pre-seeded with every sample label found in the file so you
# only have to type the numbers.  A value of None = "not yet provided".
AIR_VOLUMES = {
    # "1A": 12.5,
    # "2A": 12.5,
    # ...
}

# AM/PM -> hour of day, used only to place points on the time axis.
AM_PM_HOUR = {"AM": 7, "PM": 19}

# River flow overlay. Set FLOW_FILE = None to disable the flow trace.
FLOW_FILE  = "d:\Data\Raw\Flow\Flow_0914-0924.csv"
FLOW_UNITS = "cfs"   # label only -- change to match your data (e.g. "m3/s")

# ======================================================================
# PARSING
# ======================================================================

STAT_LABELS = {"Mean:", "RSD [%]:", "SD:"}


def _clean(x):
    """Trim stray whitespace from string cells (file has 'SIO ', 'PM ', ...)."""
    return x.strip() if isinstance(x, str) else x


def parse_workbook(path):
    """
    Read the block-structured workbook into a tidy DataFrame:
    one row per sample, columns = metadata + one column per metal (ppb).

    Returns
    -------
    df        : tidy DataFrame
    metal_cols: list of metal column names (the [ppb] analytes)
    vol_from_sheet : dict label -> air volume read from the sheet (if any)
    """
    raw = pd.read_excel(path, header=None)
    header = [_clean(c) for c in raw.iloc[0].tolist()]

    # Column roles from the header.
    # Metals are the "[ppb]" columns; the "%" columns are internal standards.
    metal_cols = [h for h in header if isinstance(h, str) and "[ppb]" in h]

    col_idx = {h: i for i, h in enumerate(header)}
    i_date, i_ampm, i_loc = col_idx["Date"], col_idx["AM_PM"], col_idx["Location"]
    i_vol, i_label = col_idx["Volume Air"], col_idx["Label"]

    # Auto-detect a per-sample dilution column (any header containing
    # "dilution" or a standalone "DF"). None if the workbook has no such column.
    i_dil = None
    for h, i in col_idx.items():
        if isinstance(h, str) and re.search(r"dilut|(^|\s)df($|\s|\b)", h, re.I):
            i_dil = i
            break
    if i_dil is not None:
        print(f"[info] Using per-sample dilution column: {header[i_dil]!r} "
              f"(missing cells default to {DEFAULT_DILUTION_FACTOR})")
    else:
        print(f"[info] No dilution column found; using "
              f"{DEFAULT_DILUTION_FACTOR} for all samples.")

    records = []
    vol_from_sheet = {}
    current = None      # metadata of the block we are inside
    pending_rec = None  # record awaiting its SD: row

    for _, row in raw.iloc[1:].iterrows():
        date_cell = row[i_date]
        label = _clean(row[i_label])

        # A new block begins on any row carrying a real date.
        if isinstance(date_cell, (datetime.datetime, pd.Timestamp)):
            dil_raw = row[i_dil] if i_dil is not None else None
            dil_val = pd.to_numeric(dil_raw, errors="coerce")
            current = {
                "date": pd.Timestamp(date_cell).normalize(),
                "am_pm": _clean(row[i_ampm]),
                "location": _clean(row[i_loc]),
                "volume_air_sheet": row[i_vol],   # usually NaN in this file
                "dilution": (float(dil_val) if pd.notna(dil_val)
                             else DEFAULT_DILUTION_FACTOR),
                "label": None,
            }
            pending_rec = None

        if current is None:
            continue

        # Capture the sample label (first non-stat, non-'Label' id in the block).
        if (isinstance(label, str) and label not in STAT_LABELS
                and label != "Label" and current["label"] is None):
            current["label"] = label
            if pd.notna(current["volume_air_sheet"]):
                vol_from_sheet[label] = float(current["volume_air_sheet"])

        # The "Mean:" row holds the sample's mean concentrations.
        if label == "Mean:":
            rec = {
                "label": current["label"],
                "date": current["date"],
                "am_pm": current["am_pm"],
                "location": current["location"],
                "dilution": current["dilution"],
            }
            for m in metal_cols:
                rec[m] = pd.to_numeric(row[col_idx[m]], errors="coerce")
            records.append(rec)
            pending_rec = rec

        # The "SD:" row holds the per-metal standard deviation (same units, ppb).
        if label == "SD:" and pending_rec is not None:
            for m in metal_cols:
                pending_rec[m + "__sd_ppb"] = pd.to_numeric(
                    row[col_idx[m]], errors="coerce")
            pending_rec = None

    df = pd.DataFrame(records)

    # Build a plottable timestamp from date + AM/PM.
    df["timestamp"] = df.apply(
        lambda r: r["date"] + pd.Timedelta(hours=AM_PM_HOUR.get(r["am_pm"], 12)),
        axis=1,
    )
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df, metal_cols, vol_from_sheet


# ======================================================================
# METAL SELECTION HELPERS
# ======================================================================

def match_metal_columns(requested, metal_cols):
    """
    Map user-friendly names ('Pb', '208Pb', 'zinc-ish') to real column
    names. Matching is on the element symbol, case-insensitive.
    """
    def element_of(colname):
        # "208Pb (KED) [ppb]" -> "Pb"
        m = re.match(r"\s*\d*([A-Za-z]+)", colname)
        return m.group(1).lower() if m else colname.lower()

    by_element = {element_of(c): c for c in metal_cols}
    resolved, missing = [], []
    for req in requested:
        key = re.sub(r"^\d+", "", req).strip().lower()  # drop mass number
        if key in by_element:
            resolved.append(by_element[key])
        else:
            missing.append(req)
    if missing:
        print(f"[warn] Not found in file, skipped: {missing}")
        print(f"       Available metals: "
              f"{[element_of(c).capitalize() for c in metal_cols]}")
    return resolved


# ======================================================================
# CONVERSION TO ng / m^3
# ======================================================================

def resolve_air_volume(label, vol_from_sheet):
    """Sheet value wins, then AIR_VOLUMES dict, then DEFAULT_AIR_VOLUME_m3."""
    if label in vol_from_sheet:
        return vol_from_sheet[label]
    v = AIR_VOLUMES.get(label)
    if v is not None:
        return float(v)
    return DEFAULT_AIR_VOLUME_m3


def add_ng_per_m3(df, metal_cols, vol_from_sheet):
    """Add air volume plus '<metal>__ng_m3' and '<metal>__ng_m3_sd' columns."""
    df = df.copy()
    df["air_volume_m3"] = df["label"].apply(
        lambda lbl: resolve_air_volume(lbl, vol_from_sheet)
    )
    # ppb (ng/mL) * dilution * extract mL / air m^3  ->  ng/m^3
    # dilution is per-sample (from the sheet, else the default).
    dil = df["dilution"] if "dilution" in df.columns else DEFAULT_DILUTION_FACTOR
    factor = dil * EXTRACT_VOLUME_mL / df["air_volume_m3"]
    for m in metal_cols:
        df[m + "__ng_m3"] = df[m] * factor
        # SD scales by the same linear factor.
        sd_col = m + "__sd_ppb"
        if sd_col in df.columns:
            df[m + "__ng_m3_sd"] = df[sd_col] * factor
        else:
            df[m + "__ng_m3_sd"] = pd.NA
    return df


# ======================================================================
# RIVER FLOW
# ======================================================================

def load_flow(path):
    """Read the flow CSV -> DataFrame with 'timestamp' and 'flow' columns."""
    fdf = pd.read_csv(path)
    # Find the datetime column and the flow column flexibly.
    dt_col = next((c for c in fdf.columns if "date" in c.lower()), fdf.columns[0])
    flow_col = next((c for c in fdf.columns if "flow" in c.lower()), fdf.columns[-1])
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(fdf[dt_col], errors="coerce"),
        "flow": pd.to_numeric(fdf[flow_col], errors="coerce"),
    }).dropna().sort_values("timestamp").reset_index(drop=True)
    return out


# ======================================================================
# PLOT
# ======================================================================

def plot_timeseries(df, selected_cols, locations=None, save_path=None,
                    flow_df=None):
    plot_df = df.copy()
    if locations:
        plot_df = plot_df[plot_df["location"].isin(locations)]

    have_vol = plot_df.dropna(subset=["air_volume_m3"])
    missing = plot_df[plot_df["air_volume_m3"].isna()]["label"].tolist()
    if missing:
        print(f"[warn] No air volume for {len(missing)} sample(s); "
              f"excluded from plot: {missing}")
    if have_vol.empty:
        raise SystemExit(
            "\nNo samples have an air volume yet, so nothing can be plotted "
            "in ng/m^3.\nFill the 'Volume Air' column in the Excel file, or "
            "the AIR_VOLUMES dict at the top of this script, then re-run."
        )

    have_vol = have_vol.sort_values("timestamp")
    multi_loc = have_vol["location"].nunique() > 1

    # Build one bar series per (metal[, location]).
    series = []
    for col in selected_cols:
        elem = re.match(r"\s*\d*([A-Za-z]+)", col).group(1)
        for loc, sub in have_vol.groupby("location"):
            lbl = elem if not multi_loc else f"{elem} - {loc}"
            series.append((lbl, col, sub.sort_values("timestamp")))
    n_series = len(series)

    # Bar geometry on a real datetime axis (positions in matplotlib date units).
    # Group width is a fraction of the smallest gap between sample times so
    # neighbouring AM/PM groups don't collide.
    times = mdates.date2num(have_vol["timestamp"].sort_values().unique())
    if len(times) > 1:
        min_gap = min(t2 - t1 for t1, t2 in zip(times, times[1:]) if t2 > t1)
    else:
        min_gap = 1.0
    group_w = 0.7 * min_gap
    bar_w = group_w / max(n_series, 1)

    fig, ax = plt.subplots(figsize=(13, 6.5))
    for k, (lbl, col, sub) in enumerate(series):
        x = mdates.date2num(sub["timestamp"]) + (k - (n_series - 1) / 2) * bar_w
        y = sub[col + "__ng_m3"].values
        yerr_col = col + "__ng_m3_sd"
        yerr = sub[yerr_col].values if yerr_col in sub.columns else None
        ax.bar(x, y, width=bar_w, yerr=yerr, capsize=2,
               error_kw={"elinewidth": 0.8}, label=lbl, align="center")

    ax.set_xlabel("Sample date")
    ax.set_ylabel("Concentration (ng metal / m$^3$ air)")
    title = "Air-filter metal concentrations"
    if locations:
        title += f"  [{', '.join(locations)}]"
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(True, axis="y", alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()

    # River-flow overlay on a secondary y-axis.
    if flow_df is not None and not flow_df.empty:
        ax2 = ax.twinx()
        (lh,) = ax2.plot(flow_df["timestamp"], flow_df["flow"],
                         color="black", lw=2, marker="s", ms=4,
                         zorder=5, label="River flow")
        ax2.set_ylabel(f"River flow ({FLOW_UNITS})" if FLOW_UNITS else "River flow")
        ax2.set_ylim(bottom=0)
        handles.append(lh)
        labels.append("River flow")

    ax.set_title(title)
    ax.legend(handles, labels, title="Series", ncol=2, fontsize=9,
              loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[ok] Plot saved to {save_path}")
    plt.show()
    return fig


# ======================================================================
# MAIN
# ======================================================================

def main():
    df, metal_cols, vol_from_sheet = parse_workbook(INPUT_FILE)
    print(f"Parsed {len(df)} samples across "
          f"{df['location'].nunique()} location(s): "
          f"{sorted(df['location'].dropna().unique())}")

    # Handy: print every sample label so you can fill AIR_VOLUMES.
    print("\nSample labels found (use these keys in AIR_VOLUMES):")
    print(", ".join(str(l) for l in df["label"].tolist()))

    selected = match_metal_columns(METALS_TO_PLOT, metal_cols)
    if not selected:
        raise SystemExit("No valid metals selected — edit METALS_TO_PLOT.")

    df = add_ng_per_m3(df, metal_cols, vol_from_sheet)

    flow_df = None
    if FLOW_FILE:
        try:
            flow_df = load_flow(FLOW_FILE)
            print(f"[ok] Loaded {len(flow_df)} river-flow points from {FLOW_FILE}")
        except FileNotFoundError:
            print(f"[warn] Flow file not found: {FLOW_FILE} — plotting without it.")

    plot_timeseries(df, selected, locations=LOCATIONS,
                    save_path="metal_timeseries.png", flow_df=flow_df)

    # Optional: also save the tidy, converted table for your records.
    out_cols = ["label", "timestamp", "location", "air_volume_m3"]
    for c in selected:
        out_cols += [c + "__ng_m3", c + "__ng_m3_sd"]
    df[out_cols].to_csv("metal_ng_per_m3.csv", index=False)
    print("[ok] Converted data written to metal_ng_per_m3.csv")


if __name__ == "__main__":
    main()