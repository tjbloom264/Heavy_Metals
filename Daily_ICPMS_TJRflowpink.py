# -*- coding: utf-8 -*-
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# FILE PATHS
# ============================================================

metals_file_path = r"/Users/auroraczajkowski/Desktop/COAST Metal Data/ECAL_Bloom(Prather)_20251217.csv"
tjr_flow_file_path = r"/Users/auroraczajkowski/Desktop/TJ river flow/April12025-December12025 TJR flow.csv"

# ============================================================
# LOAD METALS DATA
# ============================================================

df = pd.read_csv(metals_file_path)
df = df.dropna(how="all").reset_index(drop=True)

# ============================================================
# HELPERS
# ============================================================

def list_elements():
    return [col for col in df.columns if ("(" in col) or ("[" in col)]

def list_samples():
    if "Label" not in df.columns:
        raise KeyError("Could not find a 'Label' column in the metals CSV.")
    return df["Label"].dropna().astype(str).unique().tolist()

def label_to_date(label: str) -> pd.Timestamp:
    if pd.isna(label):
        return pd.NaT
    m = re.search(r"_(\d{8})$", str(label))
    if not m:
        return pd.NaT
    return pd.to_datetime(m.group(1), format="%m%d%Y", errors="coerce")

# ============================================================
# TJR FLOW HELPERS
# ============================================================

def load_tjr_flow(flow_csv_path: str) -> pd.DataFrame:
    raw = pd.read_csv(flow_csv_path)

    first_row = raw.iloc[0].astype(str).tolist()
    if any("timestamp" in s.lower() for s in first_row) and any("value" in s.lower() for s in first_row):
        flow = raw.iloc[1:].copy()
        flow.columns = first_row
    else:
        flow = raw.copy()

    ts_col, val_col = None, None
    for c in flow.columns:
        lc = str(c).lower()
        if ts_col is None and ("timestamp" in lc or "time" in lc or "date" in lc):
            ts_col = c
        if val_col is None and ("value" in lc or "gal" in lc or "mgd" in lc):
            val_col = c

    ts_col = ts_col or flow.columns[0]
    val_col = val_col or flow.columns[1]

    flow = flow[[ts_col, val_col]].rename(columns={ts_col: "dt", val_col: "flow_mgd"})
    flow["dt"] = pd.to_datetime(flow["dt"], errors="coerce")
    flow["flow_mgd"] = pd.to_numeric(flow["flow_mgd"], errors="coerce")
    flow = flow.dropna().sort_values("dt").reset_index(drop=True)
    flow["date"] = flow["dt"].dt.normalize()
    return flow

def daily_mean_flow(flow_df: pd.DataFrame) -> pd.Series:
    return flow_df.groupby("date")["flow_mgd"].mean()

# ============================================================
# CUSTOM PINK COLOR PALETTE (ELEMENT → HEX)
# ============================================================

element_colors = {
    '55Mn (KED) [ppb]': '#F7A1C4',  # soft pink
    '60Ni (KED) [ppb]': '#F06292',  # rose
    '63Cu (KED) [ppb]': '#EC407A',  # deep pink
    '66Zn (KED) [ppb]': '#AD1457',  # dark magenta
}

# ============================================================
# PLOTTING FUNCTION
# ============================================================

def plot_elements(
    element_cols,
    sample_filter=None,
    date_fmt="%Y-%m-%d",
    overlay_tjr_flow=False,
    tjr_flow_csv_path=None,
    flow_agg="daily_mean",
    flow_label="TJR River Flow",
    element_colors=None,
):
    if isinstance(element_cols, str):
        element_cols = [element_cols]

    data = df[["Label"] + element_cols].copy()
    for col in element_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=element_cols, how="all")

    if sample_filter:
        data["Label"] = data["Label"].astype(str)
        data = data[data["Label"].isin(sample_filter)]
        data = data.set_index("Label").reindex(sample_filter).reset_index()

    data["Date"] = data["Label"].apply(label_to_date)

    tick_labels = np.where(
        data["Date"].notna(),
        data["Date"].dt.strftime(date_fmt),
        data["Label"].astype(str)
    )

    x = np.arange(len(data))
    width = 0.8 / len(element_cols)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # ---- METALS (PINK SHADES) ----
    for i, col in enumerate(element_cols):
        ax1.bar(
            x + i * width,
            data[col].values,
            width,
            label=col,
            color=element_colors.get(col) if element_colors else None
        )

    ax1.set_title("ICP-MS Daily Samples with TJR Flow")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Concentration [ppb]")

    x_center = x + width * (len(element_cols) - 1) / 2
    ax1.set_xticks(x_center)
    ax1.set_xticklabels(tick_labels, rotation=90)

    ax2 = None
    if overlay_tjr_flow:
        flow_df = load_tjr_flow(tjr_flow_csv_path)
        flow_daily = daily_mean_flow(flow_df)

        flow_vals = []
        for d in data["Date"].dt.normalize():
            flow_vals.append(flow_daily.get(d, np.nan))

        ax2 = ax1.twinx()
        ax2.plot(
            x_center,
            flow_vals,
            marker="o",
            linewidth=2,
            color="#4A4A4A",
            label=f"{flow_label} (daily mean)"
        )
        ax2.set_ylabel("TJR Discharge [MGD]")

    # ---- ONE COMBINED LEGEND ----
    h1, l1 = ax1.get_legend_handles_labels()
    if ax2:
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    else:
        ax1.legend(h1, l1, loc="upper left")

    fig.tight_layout()
    plt.show()

# ============================================================
# RUN
# ============================================================

plot_elements(
    ['208Pb (KED) [ppb]'],
     #51V (KED) [ppb],'52Cr (KED) [ppb]', '59Co (KED) [ppb]','75As (KED) [ppb]', '77Se (KED) [ppb]', '111Cd (KED) [ppb]', '208Pb (KED) [ppb]'],
    sample_filter=[
        'IBA_05032025', 'IBA_05042025', 'IBA_05052025', 'IBA_05082025',
        'IBA_05132025', 'IBA_05142025', 'IBA_05152025', 'IBA_05162025',
        'IBA_05172025', 'IBA_05182025', 'IBA_05192025', 'IBA_05202025',
        'IBA_05212025', 'IBA_05222025', 'IBA_05232025', 'IBA_05242025',
        'IBA_05252025', 'IBA_05262025', 'IBA_05272025'
    ],
    date_fmt="%Y-%m-%d",
    overlay_tjr_flow=True,
    tjr_flow_csv_path=tjr_flow_file_path,
    flow_label="TJR River Flow",
    element_colors=element_colors
)
