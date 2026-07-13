import numpy as np
import pandas as pd
from openpyxl import Workbook

from config import DATA_OUTPUT, OUTPUT_TABLES


TABLES_DIR = DATA_OUTPUT
OUT_DIR = OUTPUT_TABLES

SOURCES = {
    "AR(1)":   ("0_AR",    "AR"),
    "LSTM":    ("B_LSTM",  None),
    "XGBoost": ("B_XGB",   None),
    "LASSO":   ("B_LASSO", None),
}

COUNTRIES = ["BR", "CA", "FR", "DE", "IN", "ID", "IT", "JP",
             "MX", "RU", "ZA", "KR", "TR", "GB", "US"]

OUTPUT_NAME = "Table_11_lstm_xgb.xlsx"


def load_predictions(folder_name, file_prefix=None):
    if file_prefix is None:
        file_prefix = folder_name
    path = TABLES_DIR / folder_name / f"{file_prefix}_test_predictions.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def compute_country_rmse(df, country):
    sub = df[df["country"] == country]
    if len(sub) == 0:
        return np.nan
    return float(np.sqrt(np.mean(sub["sq_error"].values)))


def build_table():
    rmse_data = {}
    for label, (folder, prefix) in SOURCES.items():
        rmse_data[label] = load_predictions(folder, prefix)

    common_keys = None
    for label, df in rmse_data.items():
        keys = set(zip(df["date"], df["country"]))
        common_keys = keys if common_keys is None else common_keys & keys

    aligned = {}
    for label, df in rmse_data.items():
        df_keys = list(zip(df["date"], df["country"]))
        mask = [k in common_keys for k in df_keys]
        aligned[label] = df[mask].copy()

    rows = []
    for country in COUNTRIES:
        row = {"Country": country}
        for label in SOURCES.keys():
            row[label] = compute_country_rmse(aligned[label], country)
        rows.append(row)

    df_table = pd.DataFrame(rows)

    avg_row = {"Country": "Average"}
    for label in SOURCES.keys():
        avg_row[label] = df_table[label].mean()

    no_tr = df_table[df_table["Country"] != "TR"]
    avg_no_tr = {"Country": "Avg excl. TR"}
    for label in SOURCES.keys():
        avg_no_tr[label] = no_tr[label].mean()

    return df_table, avg_row, avg_no_tr


def write_excel(df_table, avg_row, avg_no_tr, out_path):
    full_df = pd.concat([
        df_table,
        pd.DataFrame([avg_row]),
        pd.DataFrame([avg_no_tr]),
    ], ignore_index=True)

    full_df.to_excel(out_path, index=False)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_table, avg_row, avg_no_tr = build_table()
    print(df_table.to_string(index=False))
    print(pd.DataFrame([avg_row, avg_no_tr]).to_string(index=False))
    out_path = OUT_DIR / OUTPUT_NAME
    write_excel(df_table, avg_row, avg_no_tr, out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()