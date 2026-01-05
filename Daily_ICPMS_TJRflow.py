# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# define file paths
# Metals / ICP-MS CSV (your original path)
metals_file_path = r"/Users/auroraczajkowski/Desktop/COAST Metal Data/ECAL_Bloom(Prather)_20251217.csv"
# If running in the ChatGPT notebook environment, use:
# metals_file_path = "/mnt/data/ECAL_Bloom(Prather)_20251217.csv"

# TJR Flow CSV (your uploaded file)
tjr_flow_file_path = r"/Users/auroraczajkowski/Desktop/TJ river flow/April12025-December12025 TJR flow.csv"
# If running in the ChatGPT notebook environment, use:
# tjr_flow_file_path = "/mnt/data/April12025-December12025 TJR flow.csv"


# load data 
df = pd.read_csv(metals_file_path)
df = df.dropna(how="all").reset_index(drop=True)


# helpers 
def list_elements():
    """Return a list of element columns (columns containing '(' or '[' in the name)."""
    return [col for col in df.columns if ("(" in col) or ("[" in col)]

def list_samples():
    """Return a list of available sample labels from the Label column."""
    if "Label" not in df.columns:
        raise KeyError("Could not find a 'Label' column in the metals CSV.")
    return df["Label"].dropna().astype(str).unique().tolist()

def label_to_date(label: str) -> pd.Timestamp:
    """
    Extract MMDDYYYY from labels like IBA_05032025 (or FB_05252025, etc.)
    Returns pandas Timestamp or NaT if it can't parse.
    """
    if pd.isna(label):
        return pd.NaT
    s = str(label).strip()

    m = re.search(r"_(\d{8})$", s)  # trailing _MMDDYYYY
    if not m:
        return pd.NaT

    return pd.to_datetime(m.group(1), format="%m%d%Y", errors="coerce")

def load_tjr_flow(flow_csv_path: str) -> pd.DataFrame:
    """
    Load the 'April12025-December12025 TJR flow.csv' style export.

    That file often has the "real" headers in the first data row:
      Timestamp (UTC-08:00), Value (M US Gal/d)

    Returns a dataframe with columns:
      dt (datetime), flow_mgd (float), date (normalized day)
    """
    raw = pd.read_csv(flow_csv_path)

    # Some exports have the real headers stored in the first row
    # If the columns look generic (e.g., "Unnamed: 0"), rebuild headers from row 0.
    first_row = raw.iloc[0].astype(str).tolist()

    # Heuristic: if first row contains "Timestamp" and "Value", treat it as header-row
    if any("timestamp" in s.lower() for s in first_row) and any("value" in s.lower() for s in first_row):
        flow = raw.iloc[1:].copy()
        flow.columns = first_row
    else:
        flow = raw.copy()

    # Find timestamp and value columns robustly
    ts_col = None
    val_col = None
    for c in flow.columns:
        c_str = str(c).lower()
        if ts_col is None and ("timestamp" in c_str or "time" in c_str or "date" in c_str):
            ts_col = c
        if val_col is None and ("value" in c_str or "gal" in c_str or "mgd" in c_str or "discharge" in c_str):
            val_col = c

    # Fallback: assume first col = timestamp, second col = value
    if ts_col is None:
        ts_col = flow.columns[0]
    if val_col is None:
        val_col = flow.columns[1] if len(flow.columns) > 1 else flow.columns[0]

    flow = flow[[ts_col, val_col]].rename(columns={ts_col: "dt", val_col: "flow_mgd"})
    flow["dt"] = pd.to_datetime(flow["dt"], errors="coerce")
    flow["flow_mgd"] = pd.to_numeric(flow["flow_mgd"], errors="coerce")
    flow = flow.dropna(subset=["dt", "flow_mgd"]).sort_values("dt").reset_index(drop=True)

    flow["date"] = flow["dt"].dt.normalize()
    return flow

def daily_mean_flow(flow_df: pd.DataFrame) -> pd.Series:
    """Daily mean flow (MGD), indexed by date (Timestamp at midnight)."""
    return flow_df.groupby("date")["flow_mgd"].mean().sort_index()


# plotting setup 
def plot_elements(
    element_cols,
    sample_filter=None,
    date_fmt="%Y-%m-%d",
    overlay_tjr_flow=False,
    tjr_flow_csv_path=None,
    flow_agg="daily_mean",          # currently: "daily_mean"
    flow_label="TJR River Flow",    # legend label
):
    """
    Draw grouped bar chart for chosen elements and samples.
    If overlay_tjr_flow=True, overlay TJR flow on secondary y-axis.
    Combined legend includes metals + flow in ONE key.

    element_cols: list[str] or str
    sample_filter: list[str] of labels to include (and order to enforce), or None for all
    date_fmt: strftime format for x-axis labels (default 'YYYY-MM-DD')
    """
    if isinstance(element_cols, str):
        element_cols = [element_cols]

    if "Label" not in df.columns:
        raise KeyError("Could not find a 'Label' column in the metals CSV.")

    keep_cols = ["Label"] + element_cols
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing column(s) in metals CSV: {missing}")

    data = df[keep_cols].copy()

    # Convert metals to numeric
    for col in element_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Drop rows where all selected metals are NaN
    data = data.dropna(subset=element_cols, how="all")

    # Apply filter + enforce order
    if sample_filter:
        sample_filter = [str(s) for s in sample_filter]
        data["Label"] = data["Label"].astype(str)
        data = data[data["Label"].isin(sample_filter)]
        data = data.set_index("Label").reindex(sample_filter).reset_index()

    # Parse sample dates from labels
    data["Date"] = data["Label"].apply(label_to_date)

    # X tick labels: date only (fallback to label if parsing fails)
    tick_labels = np.where(
        data["Date"].notna(),
        data["Date"].dt.strftime(date_fmt),
        data["Label"].astype(str).values
    )

    n = len(data)
    if n == 0:
        raise ValueError("No rows to plot after filtering. Check sample_filter and element columns.")

    x = np.arange(n)
    width = 0.8 / len(element_cols)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Metals bars (primary axis) 
    for i, col in enumerate(element_cols):
        ax1.bar(x + i * width, data[col].values, width, label=col)

    ax1.set_title("ICP-MS Daily Samples with TJR Flow")
    ax1.set_xlabel("Date", fontsize=10)
    ax1.set_ylabel("Concentration [ppb]", fontsize=10)

    x_center = x + width * (len(element_cols) - 1) / 2
    ax1.set_xticks(x_center)
    ax1.set_xticklabels(tick_labels, rotation=90)

    # Optional: TJR flow overlay (secondary axis)
    ax2 = None
    if overlay_tjr_flow:
        if tjr_flow_csv_path is None:
            raise ValueError("overlay_tjr_flow=True but tjr_flow_csv_path is None.")

        flow_df = load_tjr_flow(tjr_flow_csv_path)

        if flow_agg == "daily_mean":
            flow_daily = daily_mean_flow(flow_df)
            flow_label_full = f"{flow_label} (daily mean)"
        else:
            raise ValueError(f"Unsupported flow_agg='{flow_agg}'. Use 'daily_mean'.")

        # Match each sample date to flow daily mean (exact match; fallback to nearest)
        sample_dates = data["Date"].dt.normalize()
        flow_vals = []
        for d in sample_dates:
            if pd.isna(d):
                flow_vals.append(np.nan)
                continue

            if d in flow_daily.index:
                flow_vals.append(flow_daily.loc[d])
            else:
                idx = flow_daily.index.get_indexer([d], method="nearest")
                flow_vals.append(flow_daily.iloc[idx[0]] if idx.size and idx[0] != -1 else np.nan)

        flow_vals = np.asarray(flow_vals, dtype=float)

        ax2 = ax1.twinx()
        ax2.plot(
            x_center, flow_vals,
            marker="o", linewidth=2,
            label=flow_label_full
        )
        ax2.set_ylabel("TJR Discharge [MGD]", fontsize=10)

    # ONE combined legend (metals + flow)
    handles1, labels1 = ax1.get_legend_handles_labels()
    if ax2 is not None:
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles = handles1 + handles2
        labels = labels1 + labels2
    else:
        handles = handles1
        labels = labels1

    ax1.legend(handles, labels, fontsize=12, loc="upper left")

    fig.tight_layout()
    plt.show()


# examples and definitions 
print("Available elements:", list_elements())
print("Available samples:", list_samples())

plot_elements(
    ['55Mn (KED) [ppb]', '60Ni (KED) [ppb]', '63Cu (KED) [ppb]'],
    sample_filter=[
        'IBA_05032025', 'IBA_05042025', 'IBA_05052025', 'IBA_05082025',
        'IBA_05132025', 'IBA_05142025', 'IBA_05152025', 'IBA_05162025',
        'IBA_05172025', 'IBA_05182025', 'IBA_05192025', 'IBA_05202025',
        'IBA_05212025', 'IBA_05222025', 'IBA_05232025', 'IBA_05242025',
        'IBA_05252025', 'IBA_05262025', 'IBA_05272025', 'FB_05252025', 'MB_12162025'
    ],
    date_fmt="%Y-%m-%d",           # change to "%m/%d" or "%b %d" if you prefer
    overlay_tjr_flow=True,
    tjr_flow_csv_path=tjr_flow_file_path,
    flow_agg="daily_mean",
    flow_label="TJR River Flow"
)

# Trace metals example:
# plot_elements(
#     ['51V (KED) [ppb]', '52Cr (KED) [ppb]', '59Co (KED) [ppb]',
#      '75As (KED) [ppb]', '77Se (KED) [ppb]', '111Cd (KED) [ppb]', '208Pb (KED) [ppb]'],
#     sample_filter=[...],
#     date_fmt="%m/%d",
#     overlay_tjr_flow=True,
#     tjr_flow_csv_path=tjr_flow_file_path
# )
