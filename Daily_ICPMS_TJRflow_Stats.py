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
def label_to_date(label: str) -> pd.Timestamp:
    """Parse '..._MMDDYYYY' at end of label into a Timestamp."""
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
    """
    Load TJR flow export. Handles files where:
    - first row is metadata and second row is header, OR
    - first row is header.
    Returns dataframe with columns: dt, flow_mgd, date
    """
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
        if val_col is None and ("value" in lc or "gal" in lc or "mgd" in lc or "mg/day" in lc):
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
    """Return daily mean flow indexed by normalized date."""
    s = flow_df.groupby("date")["flow_mgd"].mean()
    s.index = pd.to_datetime(s.index).normalize()
    return s

# ============================================================
# SPEARMAN RANK CORRELATION + LAG TEST
# ============================================================
def spearman_lag_test(
    metals_df: pd.DataFrame,
    element_cols: list[str],
    flow_daily: pd.Series,
    max_lag_days: int = 3,
    min_pairs: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute Spearman rank correlation between each metal and daily flow,
    testing lags from -max_lag_days ... +max_lag_days.

    Lag convention:
      lag > 0  => FLOW leads METAL by 'lag' days (metal responds after flow)
      lag < 0  => METAL leads FLOW by 'abs(lag)' days

    Returns:
      res  : tidy DataFrame (element, lag_days, n, spearman_rho)
      best : one row per element with best |rho|
    """
    out = []

    flow_daily = flow_daily.copy()
    flow_daily.index = pd.to_datetime(flow_daily.index).normalize()

    dates = pd.to_datetime(metals_df["Date"]).dt.normalize()

    for col in element_cols:
        metal = pd.to_numeric(metals_df[col], errors="coerce")

        for lag in range(-max_lag_days, max_lag_days + 1):
            # Compare metal(d) with flow(d - lag)
            flow_vals = dates.map(lambda d: flow_daily.get(d - pd.Timedelta(days=lag), np.nan))

            tmp = pd.DataFrame({"metal": metal.values, "flow": flow_vals.values}).dropna()
            n = len(tmp)

            if n < min_pairs:
                rho = np.nan
            else:
                rho = tmp["metal"].corr(tmp["flow"], method="spearman")

            out.append({"element": col, "lag_days": lag, "n": n, "spearman_rho": rho})

    res = pd.DataFrame(out)

    best = (
        res.dropna(subset=["spearman_rho"])
           .assign(abs_rho=lambda d: d["spearman_rho"].abs())
           .sort_values(["element", "abs_rho", "n"], ascending=[True, False, False])
           .groupby("element", as_index=False)
           .first()
           .rename(columns={"lag_days": "best_lag_days", "spearman_rho": "best_spearman_rho", "n": "best_n"})
           [["element", "best_lag_days", "best_spearman_rho", "best_n"]]
    )

    return res, best

# ============================================================
# CUSTOM COLOR PALETTE (ELEMENT → HEX)
# ============================================================
element_colors = {
    '55Mn (KED) [ppb]': '#F7A1C4',  # soft pink
    '60Ni (KED) [ppb]': '#F06292',  # rose
    '63Cu (KED) [ppb]': '#FF00FF',  # hot pink
    '66Zn (KED) [ppb]': '#AD1457',  # dark magenta
}

# ============================================================
# MAIN PLOT + ANALYSIS
# ============================================================
def plot_elements_with_flow_and_rankcorr(
    element_cols: list[str],
    sample_filter: list[str] | None,
    date_fmt: str,
    tjr_flow_csv_path: str,
    flow_label: str,
    element_colors: dict[str, str] | None,
    max_lag_days: int = 3,
    min_pairs: int = 6,
    save_tables: bool = True,
):
    # ---- prepare metals table
    if isinstance(element_cols, str):
        element_cols = [element_cols]

    if "Label" not in df.columns:
        raise KeyError("Could not find a 'Label' column in the metals CSV.")

    data = df[["Label"] + element_cols].copy()
    data["Label"] = data["Label"].astype(str)

    for col in element_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Keep rows that have at least one element value
    data = data.dropna(subset=element_cols, how="all")

    # Apply sample filter + preserve order
    if sample_filter:
        data = data[data["Label"].isin(sample_filter)]
        data = data.set_index("Label").reindex(sample_filter).reset_index()

    # Parse dates
    data["Date"] = data["Label"].apply(label_to_date)
    data = data.dropna(subset=["Date"]).copy()
    data["Date_norm"] = data["Date"].dt.normalize()

    # ---- load flow and compute daily mean
    flow_df = load_tjr_flow(tjr_flow_csv_path)
    flow_daily = daily_mean_flow(flow_df)

    # Flow values aligned to each sample date (no lag for the plot)
    flow_vals = data["Date_norm"].map(lambda d: flow_daily.get(d, np.nan)).to_numpy()

    # ============================================================
    # PRINT RANK CORRELATION TABLES (BEFORE PLOT SHOW)
    # ============================================================
    res, best = spearman_lag_test(
        metals_df=data,
        element_cols=element_cols,
        flow_daily=flow_daily,
        max_lag_days=max_lag_days,
        min_pairs=min_pairs,
    )

    print("\n=== Spearman rank correlation vs TJR flow (lag test) ===")
    print("Lag convention: lag > 0 means FLOW leads METAL by lag days.\n")
    print("Best lag per element (max |rho|):")
    if len(best) == 0:
        print("No results (not enough paired samples). Try lowering min_pairs or increasing sample count.")
    else:
        print(best.to_string(index=False))

    print("\nFull lag table:")
    print(res.sort_values(["element", "lag_days"]).to_string(index=False))

    if save_tables:
        res.to_csv("spearman_lag_table.csv", index=False)
        best.to_csv("spearman_best_lag.csv", index=False)
        print("\nSaved: spearman_lag_table.csv and spearman_best_lag.csv\n")

    # ============================================================
    # PLOT
    # ============================================================
    tick_labels = data["Date"].dt.strftime(date_fmt).to_numpy()

    x = np.arange(len(data))
    width = 0.8 / len(element_cols)
    x_center = x + width * (len(element_cols) - 1) / 2

    fig, ax1 = plt.subplots(figsize=(12, 6))

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
    ax1.set_xticks(x_center)
    ax1.set_xticklabels(tick_labels, rotation=90)

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

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    fig.tight_layout()
    plt.show()

# ============================================================
# RUN
# ============================================================
plot_elements_with_flow_and_rankcorr(
    element_cols=['55Mn (KED) [ppb]','60Ni (KED) [ppb]','63Cu (KED) [ppb]','66Zn (KED) [ppb]'],
    sample_filter=[
        'IBA_05032025', 'IBA_05042025', 'IBA_05052025', 'IBA_05082025',
        'IBA_05132025', 'IBA_05142025', 'IBA_05152025', 'IBA_05162025',
        'IBA_05172025', 'IBA_05182025', 'IBA_05192025', 'IBA_05202025',
        'IBA_05212025', 'IBA_05222025', 'IBA_05232025', 'IBA_05242025',
        'IBA_05252025', 'IBA_05262025', 'IBA_05272025'
    ],
    date_fmt="%Y-%m-%d",
    tjr_flow_csv_path=tjr_flow_file_path,
    flow_label="TJR River Flow",
    element_colors=element_colors,
    max_lag_days=3,     # tests lags -3..+3 days
    min_pairs=6,
    save_tables=True
)
