import numpy as np
import pandas as pd

from config import DATA_OUTPUT, OUTPUT_TABLES

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

TABLES_DIR = DATA_OUTPUT
OUT_DIR = OUTPUT_TABLES

COUNTRIES = [
    "BR", "CA", "FR", "DE", "IN", "ID", "IT", "JP",
    "MX", "RU", "ZA", "KR", "TR", "GB", "US"
]

COUNTRIES_NO_TR = [c for c in COUNTRIES if c != "TR"]

SOURCES = {
    "C": ("C_onemodel_LSTM",     None),
    "D": ("D_weekspecific_LSTM", None),
}

WEEK_POSITIONS = [1, 2, 3, 4]


def load_predictions(folder_name, file_prefix=None):
    if file_prefix is None:
        file_prefix = folder_name
    path = TABLES_DIR / folder_name / f"{file_prefix}_test_predictions.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    if "sq_error" not in df.columns:
        df["sq_error"] = (df["actual"] - df["predicted"]) ** 2
    if "week_position" in df.columns:
        df["week_position"] = (
            df["week_position"].astype(str).str.extract(r"(\d+)").astype(int)
        )
    return df


def intersect_predictions_with_week(preds):
    common_keys = None
    for _, df in preds.items():
        keys = set(zip(df["date"], df["country"], df["week_position"]))
        common_keys = keys if common_keys is None else common_keys & keys
    aligned = {}
    for label, df in preds.items():
        df_keys = list(zip(df["date"], df["country"], df["week_position"]))
        mask = [k in common_keys for k in df_keys]
        aligned[label] = df[mask].copy()
    return aligned


def compute_country_week_rmse(df, country, week):
    sub = df[
        (df["country"].astype(str).str.upper() == country.upper())
        & (df["week_position"] == week)
    ]
    if len(sub) == 0:
        return np.nan
    return float(np.sqrt(np.mean(sub["sq_error"].values)))


def avg_week_rmse(df, countries, week):
    values = [compute_country_week_rmse(df, c, week) for c in countries]
    values = [v for v in values if pd.notna(v)]
    return float(np.mean(values)) if values else np.nan


def pct_change(value, benchmark):
    if pd.isna(value) or pd.isna(benchmark) or benchmark == 0:
        return np.nan
    return (value - benchmark) / benchmark * 100


def save_excel(df, filename):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    df.to_excel(path, index=False)
    print(f"Saved: {path}")


def create_table_9():
    preds = {
        label: load_predictions(folder, prefix)
        for label, (folder, prefix) in SOURCES.items()
    }
    aligned = intersect_predictions_with_week(preds)

    rows = []

    for country in tqdm(COUNTRIES, desc="Table 9 (countries)", unit="country"):
        c_w1 = compute_country_week_rmse(aligned["C"], country, 1)
        c_w4 = compute_country_week_rmse(aligned["C"], country, 4)
        d_w1 = compute_country_week_rmse(aligned["D"], country, 1)
        d_w4 = compute_country_week_rmse(aligned["D"], country, 4)

        rows.append({
            "Country":         country,
            "C W1":            c_w1,
            "C W4":            c_w4,
            "C W1 to W4 (%)":  pct_change(c_w4, c_w1),
            "D W1":            d_w1,
            "D W4":            d_w4,
            "D W1 to W4 (%)":  pct_change(d_w4, d_w1),
        })

    for label, countries in [("Average", COUNTRIES), ("Avg excl. TR", COUNTRIES_NO_TR)]:
        c_w1 = avg_week_rmse(aligned["C"], countries, 1)
        c_w4 = avg_week_rmse(aligned["C"], countries, 4)
        d_w1 = avg_week_rmse(aligned["D"], countries, 1)
        d_w4 = avg_week_rmse(aligned["D"], countries, 4)

        rows.append({
            "Country":         label,
            "C W1":            c_w1,
            "C W4":            c_w4,
            "C W1 to W4 (%)":  pct_change(c_w4, c_w1),
            "D W1":            d_w1,
            "D W4":            d_w4,
            "D W1 to W4 (%)":  pct_change(d_w4, d_w1),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    save_excel(create_table_9(), "Table_9_cd_info_gain.xlsx")
    print("Done.")