"""
Panel builder, refactored around a single canonical panel.

Architecture:

  1. build_canonical_panel(...)
       Builds one row per (country, month-end) using the original B-MIDAS
       logic: calendar-month grouping for both Google Trends and financial
       features, with normalize_to_4_weeks (GT) and normalize_values_to_4
       (financials) collapsing longer months by averaging the trailing
       values into _w4.  Months with fewer than 4 GT weeks are skipped.
       This is the single source of truth for B, C, and D.

  2. Panel B and Panel D
       Both written directly from the canonical panel + country dummies.
       They are byte-identical; the dual files exist only so existing LSTM
       scripts that read each path keep working without changes.

  3. Panel C
       Derived from the canonical panel: each row expands into four rows
       (one per week_position), with extrapolate_last_known applied to the
       _w1..._w4 values of every weekly variable.

  4. B_monthly_panel
       Kept on its original codepath (uses monthly Google Trends data, not
       the weekly canonical model).

No date shift is applied to the financial series -- financial dates are
used as they appear in financials_weekly_52w_logdiff.csv.
"""

import numpy as np
import pandas as pd

from config import DATA_TEMP, DATA_FINAL

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

INPUT_MONTHLY_GT = DATA_TEMP / "5_monthly_logdiff.csv"
INPUT_WEEKLY_GT = DATA_TEMP / "5_weekly_logdiff.csv"
INPUT_INFLATION = DATA_FINAL / "inflation.csv"
FINANCIALS_PATH = DATA_FINAL / "financials_weekly_52w_logdiff.csv"
OUT_DIR = DATA_FINAL

N_WEEKS = 4
N_MIDAS_LAGS = 4
WEEK_POSITIONS = [1, 2, 3, 4]
DROP_NA = False

DATE_CUTOFF = "2025-10-01"
START_DATE = "2005-01-01"

COUNTRIES = ["BR", "CA", "FR", "DE", "IN", "ID", "IT", "JP", "MX", "RU", "ZA", "KR", "TR", "GB", "US"]

CODE_MAP_3TO2 = {
    "ARG": "AR", "AUS": "AU", "BRA": "BR", "CAN": "CA", "CHN": "CN",
    "FRA": "FR", "DEU": "DE", "IND": "IN", "IDN": "ID", "ITA": "IT",
    "JPN": "JP", "MEX": "MX", "RUS": "RU", "SAU": "SA", "ZAF": "ZA",
    "KOR": "KR", "TUR": "TR", "GBR": "GB", "USA": "US",
}

GLOBAL_FIN_VARS = ["v_oil", "v_copper", "v_gold", "v_vix", "v_us10y"]
COUNTRY_FIN_VARS = ["v_fxrate", "v_stock"]
ALL_FIN_VARS = GLOBAL_FIN_VARS + COUNTRY_FIN_VARS


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_inflation(path):
    df = pd.read_csv(path)

    if "year_month" in df.columns:
        df["date"] = pd.to_datetime(df["year_month"]) + pd.offsets.MonthEnd(0)
    else:
        df["date"] = pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0)

    if "infl_yoy" not in df.columns and "value" in df.columns:
        df = df.rename(columns={"value": "infl_yoy"})

    if df["country"].str.len().max() == 3:
        df["country"] = df["country"].map(CODE_MAP_3TO2)

    df = df[df["country"].isin(COUNTRIES)]
    df = df[["date", "country", "infl_yoy"]].dropna()

    return df


def load_financials(path):
    if not path.exists():
        raise FileNotFoundError(f"Financial data not found: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    df["date"] = pd.to_datetime(df["date"])
    df["country"] = df["country"].map(CODE_MAP_3TO2)
    df = df[df["country"].isin(COUNTRIES)]

    return df


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def add_inflation_lag(df):
    df = df.sort_values(["country", "date"]).copy()

    if "week_position" in df.columns:
        monthly_infl = df[["date", "country", "infl_yoy"]].drop_duplicates(["date", "country"])
        monthly_infl = monthly_infl.sort_values(["country", "date"])
        monthly_infl["infl_yoy_lag1"] = monthly_infl.groupby("country")["infl_yoy"].shift(1)

        df = df.merge(
            monthly_infl[["date", "country", "infl_yoy_lag1"]],
            on=["date", "country"],
            how="left",
        )
    else:
        df["infl_yoy_lag1"] = df.groupby("country")["infl_yoy"].shift(1)

    return df


def normalize_to_4_weeks(df_weeks):
    if len(df_weeks) <= 4:
        return df_weeks

    first_3 = df_weeks.iloc[:3]
    remaining = df_weeks.iloc[3:]

    numeric_cols = remaining.select_dtypes(include=[np.number]).columns
    non_numeric_cols = [c for c in remaining.columns if c not in numeric_cols]

    avg_data = {}
    for col in numeric_cols:
        avg_data[col] = remaining[col].mean()
    for col in non_numeric_cols:
        avg_data[col] = remaining[col].iloc[-1]

    avg_row = pd.DataFrame([avg_data], index=[remaining.index[-1]])

    return pd.concat([first_3, avg_row[first_3.columns]])


def normalize_values_to_4(values):
    values = np.array(values, dtype=float)
    if len(values) <= 4:
        return values
    return np.concatenate([values[:3], [np.nanmean(values[3:])]])


def extrapolate_last_known(weekly_values, week_position):
    result = np.zeros(N_WEEKS)
    n_available = min(week_position, len(weekly_values))

    for i in range(n_available):
        result[i] = weekly_values[i]

    if n_available > 0:
        last_known = weekly_values[n_available - 1]
        for i in range(n_available, N_WEEKS):
            result[i] = last_known

    return result


def apply_date_filters(df):
    df = df[df["date"] >= pd.to_datetime(START_DATE)]
    df = df[df["date"] < pd.to_datetime(DATE_CUTOFF)]
    return df


def attach_dummies(df):
    dummies = pd.get_dummies(df["country"], prefix="C", drop_first=True).astype(int)
    return pd.concat([df, dummies], axis=1)


def get_country_gt_var_names(df_weekly_gt):
    """Return sorted list of base GT variable names (country prefix stripped)."""
    var_names = set()
    for col in df_weekly_gt.columns:
        parts = col.split("_", 1)
        if len(parts) == 2 and parts[0] in COUNTRIES:
            var_names.add(parts[1])
    return sorted(var_names)


# ---------------------------------------------------------------------------
# Canonical panel: original B-MIDAS logic
# ---------------------------------------------------------------------------

def build_canonical_panel(df_weekly_gt, df_infl, df_fin):
    """Original B-MIDAS canonical: calendar-month grouping for both GT
    (using the GT date index) and financials (using the financial date,
    no shift).  Months with < 4 GT weeks are skipped.
    """
    fin_lookup = df_fin.copy()
    fin_lookup["month_grp"] = fin_lookup["date"].dt.to_period("M")

    all_countries = []

    for country in tqdm(COUNTRIES, desc="Canonical panel", unit="country"):
        cols = [c for c in df_weekly_gt.columns if c.startswith(f"{country}_")]
        if not cols:
            continue

        df_c = df_weekly_gt[cols].copy()
        df_c["month_grp"] = df_c.index.to_period("M")
        fin_c = fin_lookup[fin_lookup["country"] == country]

        rows = []

        for mid, group in df_c.groupby("month_grp"):
            group_features = group[cols]
            group_norm = normalize_to_4_weeks(group_features)

            if len(group_norm) < N_MIDAS_LAGS:
                continue

            last_n = group_norm.tail(N_MIDAS_LAGS)
            row = {"date": mid.to_timestamp(freq="M") + pd.offsets.MonthEnd(0)}

            # Google Trends week features
            for col in cols:
                base = col.replace(f"{country}_", "")
                for i, val in enumerate(last_n[col].values):
                    row[f"{base}_w{i + 1}"] = val

            # Financial week features
            fin_month = fin_c[fin_c["month_grp"] == mid].sort_values("date")
            if len(fin_month) >= 2:
                for var in ALL_FIN_VARS:
                    values = normalize_values_to_4(fin_month[var].values)
                    values = values[-N_MIDAS_LAGS:]
                    if len(values) < N_MIDAS_LAGS:
                        padded = np.full(N_MIDAS_LAGS, np.nan)
                        padded[N_MIDAS_LAGS - len(values):] = values
                        values = padded
                    for i, val in enumerate(values):
                        row[f"{var}_w{i + 1}"] = val

            rows.append(row)

        if not rows:
            continue

        feat_df = pd.DataFrame(rows)
        feat_df["country"] = country

        infl_c = df_infl[df_infl["country"] == country]
        all_countries.append(pd.merge(feat_df, infl_c, on=["date", "country"], how="inner"))

    if not all_countries:
        return pd.DataFrame()

    df = pd.concat(all_countries, ignore_index=True)
    df = add_inflation_lag(df)
    return df


# ---------------------------------------------------------------------------
# Panel C: derive from canonical with per-week_position extrapolation
# ---------------------------------------------------------------------------

def derive_panel_C(canonical, var_names):
    """Expand canonical to one row per (country, month, wp).
    Apply extrapolate_last_known to _w1..._w4 of every weekly variable.
    GT vars are extrapolated directly; financial vars are ffilled within
    the sequence first to preserve the original C semantics.
    """
    rows = []
    canonical_records = canonical.to_dict("records")

    for cr in tqdm(canonical_records, desc="Panel C", unit="row"):
        for wp in WEEK_POSITIONS:
            new_row = dict(cr)
            new_row["week_position"] = wp

            # GT week features
            for var in var_names:
                cols = [f"{var}_w{w}" for w in range(1, N_WEEKS + 1)]
                if not all(c in new_row for c in cols):
                    continue
                values = np.array([new_row[c] for c in cols], dtype=float)
                extrap = extrapolate_last_known(values, wp)
                for i, c in enumerate(cols):
                    new_row[c] = extrap[i]

            # Financial week features
            for var in ALL_FIN_VARS:
                cols = [f"{var}_w{w}" for w in range(1, N_WEEKS + 1)]
                if not all(c in new_row for c in cols):
                    continue
                values = np.array([new_row[c] for c in cols], dtype=float)
                if not np.isnan(values).all():
                    values = pd.Series(values).ffill().values
                    extrap = extrapolate_last_known(values, wp)
                    for i, c in enumerate(cols):
                        new_row[c] = extrap[i]

            rows.append(new_row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# B_monthly_panel: separate codepath, uses monthly GT data
# ---------------------------------------------------------------------------

def create_panel_B_monthly(df_gt_monthly, df_infl, df_fin):
    df_long = df_gt_monthly.stack().reset_index()
    df_long.columns = ["date", "variable", "value"]
    df_long["country"] = df_long["variable"].apply(lambda x: x.split("_", 1)[0])
    df_long["category"] = df_long["variable"].apply(lambda x: x.split("_", 1)[1])

    panel = df_long.pivot_table(
        index=["date", "country"],
        columns="category",
        values="value",
    ).reset_index()

    panel["date"] = pd.to_datetime(panel["date"]) + pd.offsets.MonthEnd(0)
    merged = pd.merge(panel, df_infl, on=["date", "country"], how="inner")

    df_fm = df_fin.copy()
    df_fm["month_end"] = df_fm["date"] + pd.offsets.MonthEnd(0)
    df_fm = (
        df_fm.sort_values("date")
        .groupby(["month_end", "country"])[ALL_FIN_VARS]
        .last()
        .reset_index()
    )
    df_fm = df_fm.rename(columns={"month_end": "date"})

    merged = pd.merge(merged, df_fm, on=["date", "country"], how="left")
    merged = add_inflation_lag(merged)

    dummies = pd.get_dummies(merged["country"], prefix="C", drop_first=True).astype(int)

    return pd.concat([merged, dummies], axis=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_infl = load_inflation(INPUT_INFLATION)
    df_weekly_gt = pd.read_csv(INPUT_WEEKLY_GT, index_col=0, parse_dates=True)
    df_monthly_gt = pd.read_csv(INPUT_MONTHLY_GT, index_col=0, parse_dates=True)
    df_fin = load_financials(FINANCIALS_PATH)

    # Single source of truth (original B-MIDAS logic, no date shift).
    canonical = build_canonical_panel(df_weekly_gt, df_infl, df_fin)
    canonical = apply_date_filters(canonical)

    # Panels B and D: the canonical with country dummies attached.
    # Byte-identical files; both written so that B_LSTM.py and
    # D_weekspecific_LSTM.py keep working without changes.
    panel_full = attach_dummies(canonical)
    panel_full.to_csv(OUT_DIR / "B_panel.csv", index=False)
    panel_full.to_csv(OUT_DIR / "D_panel.csv", index=False)

    # Panel C: canonical expanded across week_positions with extrapolation.
    var_names = get_country_gt_var_names(df_weekly_gt)
    panel_c = derive_panel_C(canonical, var_names)
    panel_c = attach_dummies(panel_c)
    if DROP_NA:
        panel_c = panel_c.dropna()
    panel_c.to_csv(OUT_DIR / "C_panel.csv", index=False)

    # B_monthly_panel: separate codepath (monthly GT input).
    panel_b_monthly = create_panel_B_monthly(df_monthly_gt, df_infl, df_fin)
    panel_b_monthly = apply_date_filters(panel_b_monthly)
    panel_b_monthly.to_csv(OUT_DIR / "B_monthly_panel.csv", index=False)

    print(f"Done. Panels saved to: {OUT_DIR}")