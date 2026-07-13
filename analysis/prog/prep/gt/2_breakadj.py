import numpy as np
import pandas as pd

from config import DATA_TEMP

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

MONTHLY_RAW_PATH = DATA_TEMP / "1_monthly_raw.csv"
WEEKLY_SPLICED_PATH = DATA_TEMP / "1_weekly_spliced_raw.csv"
OUT_DIR = DATA_TEMP

BREAKS_CAT = [(2011, 2010), (2016, 2015), (2022, 2021)]
BREAKS_TOP = [(2011, 2010), (2016, 2015), (2017, 2016), (2022, 2021)]

WINDOW_MONTHS = 12


def is_topic(col):
    return "_t_" in col


def get_window_average(series, break_date, months, side="before"):
    freq = pd.infer_freq(series.index[:20])

    if side == "before":
        if freq and "W" in str(freq):
            n_periods = int(months * (52 / 12))
            window = series[
                (series.index < break_date)
                & (series.index >= break_date - pd.DateOffset(weeks=n_periods))
            ]
        else:
            window = series[
                (series.index < break_date)
                & (series.index >= break_date - pd.DateOffset(months=months))
            ]
    else:
        if freq and "W" in str(freq):
            n_periods = int(months * (52 / 12))
            window = series[
                (series.index >= break_date)
                & (series.index < break_date + pd.DateOffset(weeks=n_periods))
            ]
        else:
            window = series[
                (series.index >= break_date)
                & (series.index < break_date + pd.DateOffset(months=months))
            ]

    return window.mean() if len(window) > 0 else np.nan


def fix_breaks(series, breaks):
    adjusted = series.copy()

    for break_year, _ in sorted(breaks):
        break_date = pd.Timestamp(f"{break_year}-01-01")

        if break_date < adjusted.index.min() or break_date > adjusted.index.max():
            continue

        avg_before = get_window_average(adjusted, break_date, WINDOW_MONTHS, side="before")
        avg_after = get_window_average(adjusted, break_date, WINDOW_MONTHS, side="after")

        if np.isnan(avg_before) or np.isnan(avg_after):
            continue

        if avg_after < 1e-6:
            continue

        ratio = avg_before / avg_after

        if abs(ratio - 1.0) < 0.01:
            continue

        adjusted.loc[adjusted.index >= break_date] *= ratio

    return adjusted


def adjust_breaks(input_path, output_name):
    df_raw = pd.read_csv(input_path, index_col=0, parse_dates=True)

    adjusted = {}

    for col in tqdm(df_raw.columns, desc=f"Break-adjusting {output_name}", unit="series"):
        breaks = BREAKS_TOP if is_topic(col) else BREAKS_CAT
        adjusted[col] = fix_breaks(df_raw[col].dropna(), breaks)

    df_adjusted = pd.DataFrame(adjusted).sort_index()
    df_adjusted.to_csv(OUT_DIR / output_name)

    return df_adjusted


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    adjust_breaks(MONTHLY_RAW_PATH, "2_monthly_breakadj.csv")
    adjust_breaks(WEEKLY_SPLICED_PATH, "2_weekly_breakadj.csv")

    print(f"Done. Break-adjusted files saved to: {OUT_DIR}")