from io import StringIO

import numpy as np
import pandas as pd
import requests

from config import DATA_FINAL, DATA_RAW

OUTPUT_PATH = DATA_FINAL / "inflation.csv"
RUSSIA_INPUT_PATH = DATA_RAW / "russia_inflation.csv"

G20 = [
    "ARG", "AUS", "BRA", "CAN", "CHN", "FRA", "DEU", "IND", "IDN",
    "ITA", "JPN", "MEX", "RUS", "SAU", "ZAF", "KOR", "TUR", "GBR", "USA"
]

BASE_URL = "https://sdmx.oecd.org/public/rest/data"
SELECTOR = f"{'+'.join(G20)}.M.N.CPI.PA._T.N.GY"
PARAMS = "startPeriod=2004-01&dimensionAtObservation=AllDimensions&format=csvfilewithlabels"

DS_1999 = "OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL,1.0"
DS_2018 = "OECD.SDD.TPS,DSD_PRICES_COICOP2018@DF_PRICES_C2018_ALL,1.0"


def fetch_oecd(dataset):
    url = f"{BASE_URL}/{dataset}/{SELECTOR}?{PARAMS}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    raw = pd.read_csv(StringIO(response.text))

    cols = {c.lower(): c for c in raw.columns}
    country_col = cols.get("location") or cols.get("ref_area")
    date_col = cols.get("time_period")
    value_col = cols.get("obs_value")

    df = raw[[country_col, date_col, value_col]].rename(
        columns={
            country_col: "country",
            date_col: "date",
            value_col: "infl_yoy"
        }
    )

    df["date"] = df["date"].astype(str).str[:7]
    df["tp"] = pd.PeriodIndex(df["date"], freq="M")
    df["infl_yoy"] = pd.to_numeric(df["infl_yoy"], errors="coerce")

    return df[["country", "date", "tp", "infl_yoy"]].dropna(subset=["tp"])


def load_manual_russia_data(path=RUSSIA_INPUT_PATH):
    if not path.exists():
        raise FileNotFoundError(
            "Private Russian inflation supplement not found. "
            f"Create {path} using the schema documented in analysis/data/README.md."
        )

    rus_df = pd.read_csv(path)
    required = {"country", "date", "infl_yoy"}
    missing = required.difference(rus_df.columns)
    if missing:
        raise ValueError(
            f"Russian inflation supplement is missing columns: {sorted(missing)}"
        )

    rus_df = rus_df[["country", "date", "infl_yoy"]].copy()
    rus_df["country"] = rus_df["country"].astype(str).str.upper().str.strip()

    if not rus_df["country"].eq("RUS").all():
        raise ValueError("Russian inflation supplement may contain only country=RUS")

    date_text = rus_df["date"].astype(str).str.strip()
    if not date_text.str.fullmatch(r"\d{4}-(0[1-9]|1[0-2])").all():
        raise ValueError("Russian inflation dates must use monthly YYYY-MM values")

    try:
        rus_df["tp"] = pd.PeriodIndex(date_text, freq="M")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Russian inflation dates must use monthly YYYY-MM values"
        ) from exc

    rus_df["infl_yoy"] = pd.to_numeric(rus_df["infl_yoy"], errors="coerce")
    if rus_df["infl_yoy"].isna().any() or not np.isfinite(rus_df["infl_yoy"]).all():
        raise ValueError("Russian inflation supplement contains a missing/non-numeric value")

    if rus_df.duplicated(["country", "tp"]).any():
        raise ValueError("Russian inflation supplement contains duplicate months")

    rus_df["date"] = rus_df["tp"].astype(str)
    return rus_df


def add_manual_russia_data(df, manual_path=RUSSIA_INPUT_PATH):
    rus_df = load_manual_russia_data(manual_path)

    df = df.copy()
    df["tp"] = pd.PeriodIndex(df["date"], freq="M")

    base_df = df.set_index(["country", "tp"])
    manual_df = rus_df.drop(columns="date").set_index(["country", "tp"])

    df = base_df.combine_first(manual_df).reset_index()
    df["date"] = df["tp"].astype(str)

    return (
        df.sort_values(["country", "tp"], kind="mergesort")
        .drop(columns="tp")
        .reset_index(drop=True)
    )


def fill_usa_october_2025(df):
    df = df.copy()

    try:
        sep = df.loc[
            (df["country"] == "USA") & (df["date"] == "2025-09"),
            "infl_yoy"
        ].iloc[0]

        nov = df.loc[
            (df["country"] == "USA") & (df["date"] == "2025-11"),
            "infl_yoy"
        ].iloc[0]

        if not ((df["country"] == "USA") & (df["date"] == "2025-10")).any():
            df = pd.concat(
                [
                    df,
                    pd.DataFrame([{
                        "country": "USA",
                        "date": "2025-10",
                        "infl_yoy": np.nan
                    }])
                ],
                ignore_index=True
            )

        df.loc[
            (df["country"] == "USA") & (df["date"] == "2025-10"),
            "infl_yoy"
        ] = (sep + nov) / 2

    except Exception:
        pass

    df["tp"] = pd.PeriodIndex(df["date"], freq="M")

    return (
        df.sort_values(["country", "tp"], kind="mergesort")
        .drop(columns="tp")
        .reset_index(drop=True)
    )


def fetch_inflation():
    DATA_FINAL.mkdir(parents=True, exist_ok=True)

    df99 = fetch_oecd(DS_1999)
    df18 = fetch_oecd(DS_2018)

    m99 = df99.drop(columns=["date"]).set_index(["country", "tp"])
    m18 = df18.drop(columns=["date"]).set_index(["country", "tp"])

    df = m99.combine_first(m18).reset_index()
    df["date"] = df["tp"].astype(str)

    df = (
        df.sort_values(["country", "tp"], kind="mergesort")
        .drop(columns="tp")
        .reset_index(drop=True)
    )

    df = add_manual_russia_data(df)
    df = fill_usa_october_2025(df)

    df.to_csv(OUTPUT_PATH, index=False)

    return df


if __name__ == "__main__":
    fetch_inflation()
    print(f"Done. Saved inflation data to: {OUTPUT_PATH}")
