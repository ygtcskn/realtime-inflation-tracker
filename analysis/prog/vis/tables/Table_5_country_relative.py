import numpy as np
import pandas as pd
from openpyxl import Workbook

from config import DATA_OUTPUT, OUTPUT_TABLES


TABLES_DIR = DATA_OUTPUT
OUT_DIR = OUTPUT_TABLES

SOURCES = {
    "RW":   ("0_RW",    None),
    "AR":   ("0_AR",    "AR"),
    "LSTM": ("B_LSTM",  None),
}

COUNTRIES = ["BR", "CA", "FR", "DE", "IN", "ID", "IT", "JP",
             "MX", "RU", "ZA", "KR", "TR", "GB", "US"]

OUTPUT_NAME = "Table_5_country_relative.xlsx"


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
        rmse_rw = compute_country_rmse(aligned["RW"], country)
        rmse_ar = compute_country_rmse(aligned["AR"], country)
        rmse_lstm = compute_country_rmse(aligned["LSTM"], country)
        rows.append({
            "Country": country,
            "RW": rmse_rw,
            "AR(1)": rmse_ar,
            "LSTM": rmse_lstm,
            "LSTM/AR": rmse_lstm / rmse_ar if rmse_ar > 0 else np.nan,
            "LSTM/RW": rmse_lstm / rmse_rw if rmse_rw > 0 else np.nan,
        })

    df_table = pd.DataFrame(rows)

    avg_row = {
        "Country": "Average",
        "RW": df_table["RW"].mean(),
        "AR(1)": df_table["AR(1)"].mean(),
        "LSTM": df_table["LSTM"].mean(),
        "LSTM/AR": df_table["LSTM/AR"].mean(),
        "LSTM/RW": df_table["LSTM/RW"].mean(),
    }

    no_tr = df_table[df_table["Country"] != "TR"]
    avg_no_tr = {
        "Country": "Avg excl. TR",
        "RW": no_tr["RW"].mean(),
        "AR(1)": no_tr["AR(1)"].mean(),
        "LSTM": no_tr["LSTM"].mean(),
        "LSTM/AR": no_tr["LSTM/AR"].mean(),
        "LSTM/RW": no_tr["LSTM/RW"].mean(),
    }

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