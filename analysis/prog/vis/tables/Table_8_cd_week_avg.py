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
    return aligned, common_keys


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


def winner_lower(c_value, d_value):
    if pd.isna(c_value) or pd.isna(d_value):
        return np.nan
    return "C" if c_value <= d_value else "D"


def save_excel(df, filename):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    df.to_excel(path, index=False)
    print(f"Saved: {path}")


def create_table_8():
    preds = {
        label: load_predictions(folder, prefix)
        for label, (folder, prefix) in SOURCES.items()
    }
    aligned, common_keys = intersect_predictions_with_week(preds)
    print(f"Common (date, country, week) observations: {len(common_keys)}")

    rows = []

    for label, df in aligned.items():
        values = [avg_week_rmse(df, COUNTRIES, week) for week in WEEK_POSITIONS]
        rows.append({
            "Model": label,
            "W1": values[0],
            "W2": values[1],
            "W3": values[2],
            "W4": values[3],
            "W1 to W4 (%)": pct_change(values[3], values[0]),
        })

    winner_row = {"Model": "Winner"}
    for week in WEEK_POSITIONS:
        c_rmse = avg_week_rmse(aligned["C"], COUNTRIES, week)
        d_rmse = avg_week_rmse(aligned["D"], COUNTRIES, week)
        winner_row[f"W{week}"] = winner_lower(c_rmse, d_rmse)
    winner_row["W1 to W4 (%)"] = ""
    rows.append(winner_row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    save_excel(create_table_8(), "Table_8_cd_week_avg.xlsx")
    print("Done.")