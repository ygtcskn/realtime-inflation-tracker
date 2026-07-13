import warnings

import numpy as np
import pandas as pd

from config import DATA_TEMP

warnings.filterwarnings("ignore")

INPUT_MONTHLY = DATA_TEMP / "4_monthly_detrended_final.csv"
INPUT_WEEKLY = DATA_TEMP / "4_weekly_detrended_final.csv"

OUT_MONTHLY = DATA_TEMP / "5_monthly_logdiff.csv"
OUT_WEEKLY = DATA_TEMP / "5_weekly_logdiff.csv"

YOY_LAG_MONTHLY = 12
YOY_LAG_WEEKLY = 52


def is_topic(col):
    parts = col.split("_", 1)
    return len(parts) == 2 and parts[1].startswith("t_")


def safe_log(df):
    return np.log(df.clip(lower=1e-6))


def apply_logdiff(df, lag):
    df_log = safe_log(df)
    df_out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    for col in df.columns:
        if is_topic(col):
            df_out[col] = df_log[col]
        else:
            df_out[col] = df_log[col] - df_log[col].shift(lag)

    return df_out


def process_file(input_path, output_path, lag):
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    df_out = apply_logdiff(df, lag)
    df_out.to_csv(output_path)

    return df_out


if __name__ == "__main__":
    DATA_TEMP.mkdir(parents=True, exist_ok=True)

    process_file(INPUT_MONTHLY, OUT_MONTHLY, YOY_LAG_MONTHLY)
    process_file(INPUT_WEEKLY, OUT_WEEKLY, YOY_LAG_WEEKLY)

    print(f"Done. Log-differenced files saved to: {DATA_TEMP}")