import sys
import warnings

import numpy as np
import pandas as pd

from config import DATA_TEMP

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

warnings.filterwarnings("ignore")

try:
    from scipy.interpolate import make_smoothing_spline
except ImportError:
    sys.exit(1)

INPUT_MONTHLY = DATA_TEMP / "2_monthly_breakadj.csv"
INPUT_WEEKLY = DATA_TEMP / "2_weekly_breakadj.csv"

OUT_MONTHLY = DATA_TEMP / "3_monthly_clean.csv"
OUT_WEEKLY = DATA_TEMP / "3_weekly_clean.csv"

WINDOW_MONTHLY = 5
WINDOW_WEEKLY = 20

LAMBDA_GRID = [0.1, 0.3, 0.5, 0.8, 1.0, 1.3, 1.5, 1.8, 2.0]
LAMBDA_TRAIN_END = "2018-01-01"
SELECTIVE_DENOISE = True


def shift_to_positive(df):
    df_shifted = df.copy()

    for col in df_shifted.columns:
        col_min = df_shifted[col].min()

        if col_min < 1:
            df_shifted[col] = df_shifted[col] + (1 - col_min)

    return df_shifted


def find_optimal_lambda(values, window_size, lambda_grid):
    n = len(values)

    if n <= window_size + 1:
        return lambda_grid[len(lambda_grid) // 2], np.inf

    best_lambda = lambda_grid[0]
    best_rmse = np.inf

    for lam in lambda_grid:
        preds = []
        actuals = []

        for t in range(window_size, n):
            window = values[t - window_size:t]
            x = np.arange(window_size, dtype=float)

            try:
                spline = make_smoothing_spline(x, window.astype(float), lam=lam)
                preds.append(float(spline(float(window_size))))
                actuals.append(values[t])
            except Exception:
                continue

        if preds:
            rmse = np.sqrt(np.mean((np.array(actuals) - np.array(preds)) ** 2))

            if rmse < best_rmse:
                best_rmse = rmse
                best_lambda = lam

    return best_lambda, best_rmse


def denoise_series(values, window_size, lam):
    denoised = values.copy().astype(float)
    n = len(values)

    for t in range(window_size - 1, n):
        start = t - window_size + 1
        window = values[start:t + 1].astype(float)
        x = np.arange(len(window), dtype=float)

        try:
            spline = make_smoothing_spline(x, window, lam=lam)
            denoised[t] = float(spline(x[-1]))
        except Exception:
            pass

    return denoised


def denoise_dataframe(df, window_size):
    train_end = pd.Timestamp(LAMBDA_TRAIN_END)
    df_denoised = df.copy()
    lambda_results = {}

    for col in tqdm(df.columns, desc="Tuning lambda", unit="series"):
        train_data = df[col][df.index < train_end].dropna().values

        if len(train_data) <= window_size + 5:
            lambda_results[col] = (1.0, 0.0)
            continue

        lambda_results[col] = find_optimal_lambda(train_data, window_size, LAMBDA_GRID)

    rmses = [rmse for _, rmse in lambda_results.values() if rmse < np.inf and rmse > 0]

    if SELECTIVE_DENOISE and rmses:
        median_rmse = np.median(rmses)
        to_denoise = {
            col for col, (_, rmse) in lambda_results.items()
            if rmse > median_rmse and rmse < np.inf
        }
    else:
        to_denoise = set(df.columns)

    for col in tqdm(to_denoise, desc="Denoising", unit="series"):
        values = df[col].values.copy()

        if np.any(np.isnan(values)):
            values = pd.Series(values).interpolate().bfill().ffill().values

        best_lambda = lambda_results[col][0]
        df_denoised[col] = denoise_series(values, window_size, best_lambda)

    return df_denoised


if __name__ == "__main__":
    DATA_TEMP.mkdir(parents=True, exist_ok=True)

    df_monthly = pd.read_csv(INPUT_MONTHLY, index_col=0, parse_dates=True)
    df_weekly = pd.read_csv(INPUT_WEEKLY, index_col=0, parse_dates=True)

    df_monthly_clean = denoise_dataframe(df_monthly, WINDOW_MONTHLY)
    df_weekly_clean = denoise_dataframe(df_weekly, WINDOW_WEEKLY)

    df_monthly_clean = shift_to_positive(df_monthly_clean)
    df_weekly_clean = shift_to_positive(df_weekly_clean)

    df_monthly_clean.to_csv(OUT_MONTHLY)
    df_weekly_clean.to_csv(OUT_WEEKLY)

    print(f"Done. Denoised and shifted files saved to: {DATA_TEMP}")