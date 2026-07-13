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

THREE_PERIODS = {
    "Pandemic Shock":  ("2020-01-01", "2020-12-31"),
    "Inflation Surge": ("2021-01-01", "2023-12-31"),
    "Normalisation":   ("2024-01-01", "2025-12-31"),
}

THREE_PERIODS_ORDER = [
    "Pandemic Shock", "Inflation Surge", "Normalisation",
]

SOURCES = {
    "AR":   ("0_AR",   "AR"),
    "LSTM": ("B_LSTM", None),
}


def load_predictions(folder_name, file_prefix=None):
    if file_prefix is None:
        file_prefix = folder_name
    path = TABLES_DIR / folder_name / f"{file_prefix}_test_predictions.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    if "sq_error" not in df.columns:
        df["sq_error"] = (df["actual"] - df["predicted"]) ** 2
    return df


def intersect_predictions(preds):
    common_keys = None
    for _, df in preds.items():
        keys = set(zip(df["date"], df["country"]))
        common_keys = keys if common_keys is None else common_keys & keys

    aligned = {}
    for label, df in preds.items():
        df_keys = list(zip(df["date"], df["country"]))
        mask = [k in common_keys for k in df_keys]
        aligned[label] = df[mask].copy()
    return aligned, common_keys


def compute_rmse(df, country=None, date_start=None, date_end=None):
    sub = df
    if country is not None:
        sub = sub[sub["country"].astype(str).str.upper() == country.upper()]
    if date_start is not None:
        sub = sub[sub["date"] >= pd.Timestamp(date_start)]
    if date_end is not None:
        sub = sub[sub["date"] <= pd.Timestamp(date_end)]
    if len(sub) == 0:
        return np.nan
    return float(np.sqrt(np.mean(sub["sq_error"].values)))


def save_excel(df, filename):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    df.to_excel(path, index=False)
    print(f"Saved: {path}")


def create_table_6():
    preds = {
        label: load_predictions(folder, prefix)
        for label, (folder, prefix) in SOURCES.items()
    }

    aligned, common_keys = intersect_predictions(preds)
    print(f"Common observations across AR and LSTM: {len(common_keys)}")

    pred_ar = aligned["AR"]
    pred_lstm = aligned["LSTM"]

    rows = []
    ratios_by_country = {}

    for country in tqdm(COUNTRIES, desc="Table 6 (countries)", unit="country"):
        country_ratios = {}

        for period_name in THREE_PERIODS_ORDER:
            start, end = THREE_PERIODS[period_name]

            lstm_rmse = compute_rmse(pred_lstm, country=country, date_start=start, date_end=end)
            ar_rmse   = compute_rmse(pred_ar,   country=country, date_start=start, date_end=end)

            ratio = (
                lstm_rmse / ar_rmse
                if pd.notna(lstm_rmse) and pd.notna(ar_rmse) and ar_rmse > 0
                else np.nan
            )
            country_ratios[period_name] = ratio

        ratios_by_country[country] = country_ratios

        rows.append({
            "Country": country,
            "Pandemic Shock":  country_ratios["Pandemic Shock"],
            "Inflation Surge": country_ratios["Inflation Surge"],
            "Normalisation":   country_ratios["Normalisation"],
        })

    rows.append({
        "Country": "Average",
        "Pandemic Shock":  np.nanmean([ratios_by_country[c]["Pandemic Shock"]  for c in COUNTRIES]),
        "Inflation Surge": np.nanmean([ratios_by_country[c]["Inflation Surge"] for c in COUNTRIES]),
        "Normalisation":   np.nanmean([ratios_by_country[c]["Normalisation"]   for c in COUNTRIES]),
    })

    rows.append({
        "Country": "Avg excl. TR",
        "Pandemic Shock":  np.nanmean([ratios_by_country[c]["Pandemic Shock"]  for c in COUNTRIES_NO_TR]),
        "Inflation Surge": np.nanmean([ratios_by_country[c]["Inflation Surge"] for c in COUNTRIES_NO_TR]),
        "Normalisation":   np.nanmean([ratios_by_country[c]["Normalisation"]   for c in COUNTRIES_NO_TR]),
    })

    rows.append({
        "Country": "Beats AR (N)",
        "Pandemic Shock":  (
            f"{sum(ratios_by_country[c]['Pandemic Shock'] < 1 for c in COUNTRIES if pd.notna(ratios_by_country[c]['Pandemic Shock']))}/"
            f"{sum(pd.notna(ratios_by_country[c]['Pandemic Shock']) for c in COUNTRIES)}"
        ),
        "Inflation Surge": (
            f"{sum(ratios_by_country[c]['Inflation Surge'] < 1 for c in COUNTRIES if pd.notna(ratios_by_country[c]['Inflation Surge']))}/"
            f"{sum(pd.notna(ratios_by_country[c]['Inflation Surge']) for c in COUNTRIES)}"
        ),
        "Normalisation":   (
            f"{sum(ratios_by_country[c]['Normalisation'] < 1 for c in COUNTRIES if pd.notna(ratios_by_country[c]['Normalisation']))}/"
            f"{sum(pd.notna(ratios_by_country[c]['Normalisation']) for c in COUNTRIES)}"
        ),
    })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    save_excel(create_table_6(), "Table_6_lstm_ar_ratio_phases.xlsx")
    print("Done.")