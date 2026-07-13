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

# (key, folder, file_prefix, display_name)
MODELS = [
    ("RW",    "0_RW",    None, "Random Walk"),
    ("AR",    "0_AR",    "AR", "AR(1)"),
    ("LASSO", "B_LASSO", None, "LASSO"),
    ("XGB",   "B_XGB",   None, "XGB"),
    ("LSTM",  "B_LSTM",  None, "LSTM"),
]


def load_predictions(folder_name, file_prefix=None):
    if file_prefix is None:
        file_prefix = folder_name

    path = TABLES_DIR / folder_name / f"{file_prefix}_test_predictions.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    # Ensure error columns exist (in case some model scripts didn't save them)
    if "error" not in df.columns:
        df["error"] = df["actual"] - df["predicted"]
    if "abs_error" not in df.columns:
        df["abs_error"] = df["error"].abs()
    if "sq_error" not in df.columns:
        df["sq_error"] = df["error"] ** 2

    return df


def compute_country_rmse(df, country):
    sub = df[df["country"] == country]
    if len(sub) == 0:
        return np.nan
    return float(np.sqrt(np.mean(sub["sq_error"].values)))


def compute_country_mae(df, country):
    sub = df[df["country"] == country]
    if len(sub) == 0:
        return np.nan
    return float(np.mean(sub["abs_error"].values))


def pct_change(value, benchmark):
    if pd.isna(value) or pd.isna(benchmark) or benchmark == 0:
        return np.nan
    return (value - benchmark) / benchmark * 100


def save_excel(df, filename):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    df.to_excel(path, index=False)
    print(f"Saved: {path}")


def create_table_1():
    # Load predictions for all models
    predictions = {}
    for key, folder, prefix, _ in MODELS:
        predictions[key] = load_predictions(folder, prefix)

    # Find common (date, country) intersection across ALL models
    common_keys = None
    for label, df in predictions.items():
        keys = set(zip(df["date"], df["country"]))
        common_keys = keys if common_keys is None else common_keys & keys

    print(f"Common observations across all models: {len(common_keys)}")

    # Filter each model's predictions to only the common subset
    aligned = {}
    for label, df in predictions.items():
        df_keys = list(zip(df["date"], df["country"]))
        mask = [k in common_keys for k in df_keys]
        aligned[label] = df[mask].copy()

    # Compute per-country RMSE and MAE for each model on the aligned subset
    per_country_rmse = {}
    per_country_mae = {}
    for key, _, _, _ in MODELS:
        rmse_list = [compute_country_rmse(aligned[key], c) for c in COUNTRIES]
        mae_list  = [compute_country_mae(aligned[key], c)  for c in COUNTRIES]
        per_country_rmse[key] = rmse_list
        per_country_mae[key] = mae_list

    # Average across countries
    averages = {}
    for key, _, _, _ in MODELS:
        averages[key] = {
            "rmse": float(np.nanmean(per_country_rmse[key])),
            "mae":  float(np.nanmean(per_country_mae[key])),
        }

    rw_rmse = averages["RW"]["rmse"]
    ar_rmse = averages["AR"]["rmse"]

    rows = []
    for key, _, _, model_name in MODELS:
        rows.append({
            "Model": model_name,
            "RMSE": averages[key]["rmse"],
            "MAE":  averages[key]["mae"],
            "vs RW (%)": pct_change(averages[key]["rmse"], rw_rmse),
            "vs AR (%)": pct_change(averages[key]["rmse"], ar_rmse),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    save_excel(create_table_1(), "Table_4_umidas_avg_metrics.xlsx")
    print("Done.")