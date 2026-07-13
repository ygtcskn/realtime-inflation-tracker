import io
import warnings

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from config import DATA_RAW, DATA_TEMP, DATA_FINAL

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

warnings.simplefilter(action="ignore", category=FutureWarning)

TEMP_DIR = DATA_TEMP
FINAL_DIR = DATA_FINAL

RAW_OUTPUT_PATH = TEMP_DIR / "financials_weekly_raw.csv"
LOGDIFF_OUTPUT_PATH = FINAL_DIR / "financials_weekly_52w_logdiff.csv"
ZAF_MANUAL_PATH = DATA_RAW / "ZAF.csv"

TEMP_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2003-12-01"
END_DATE = "2026-12-31"

GLOBAL_VARS = ["v_oil", "v_copper", "v_gold", "v_vix", "v_us10y"]
COUNTRY_VARS = ["v_fxrate", "v_stock"]
ALL_VARS = GLOBAL_VARS + COUNTRY_VARS

COUNTRIES = [
    "BRA", "CAN", "FRA", "DEU", "IND", "IDN", "ITA", "JPN",
    "MEX", "RUS", "ZAF", "KOR", "TUR", "GBR", "USA"
]

ISO_MAP = {
    "AR": "ARG", "AU": "AUS", "BR": "BRA", "CA": "CAN", "CN": "CHN",
    "DE": "DEU", "FR": "FRA", "GB": "GBR", "ID": "IDN", "IN": "IND",
    "IT": "ITA", "JP": "JPN", "KR": "KOR", "MX": "MEX", "RU": "RUS",
    "SA": "SAU", "TR": "TUR", "US": "USA", "ZA": "ZAF"
}

FX_MAP = {
    "AU": ("AUDUSD=X", True),
    "DE": ("EURUSD=X", True),
    "FR": ("EURUSD=X", True),
    "IT": ("EURUSD=X", True),
    "GB": ("GBPUSD=X", True),
    "AR": ("ARS=X", False),
    "BR": ("BRL=X", False),
    "CA": ("CAD=X", False),
    "CN": ("CNY=X", False),
    "IN": ("INR=X", False),
    "ID": ("IDR=X", False),
    "JP": ("JPY=X", False),
    "MX": ("MXN=X", False),
    "RU": ("RUB=X", False),
    "SA": ("SAR=X", False),
    "ZA": ("ZAR=X", False),
    "KR": ("KRW=X", False),
    "US": ("DX-Y.NYB", False),
}

STOCK_MAP = {
    "US": "^GSPC",
    "DE": "^GDAXI",
    "FR": "^FCHI",
    "GB": "^FTSE",
    "JP": "^N225",
    "CN": "000001.SS",
    "IN": "^BSESN",
    "BR": "^BVSP",
    "MX": "^MXX",
    "KR": "^KS11",
    "TR": "XU100.IS",
    "ID": "^JKSE",
    "CA": "^GSPTSE",
    "AU": "^AXJO",
    "AR": "^MERV",
    "IT": "FTSEMIB.MI",
    "RU": "IMOEX.ME",
    "ZA": "^J203.JO",
    "SA": "^TASI.SR",
}

GLOBAL_COMMODITIES = {
    "v_oil": "CL=F",
    "v_copper": "HG=F",
    "v_gold": "GC=F",
    "v_vix": "^VIX",
    "v_us10y": "^TNX",
}

STOOQ_OVERRIDES = {
    "TR_FX": ("usdtry", False),
    "RU_STOCK": ("^rts", False),
    "BR_FX": ("usdbrl", False),
    "AU_FX": ("audusd", True),
}


def safe_download_yahoo(ticker):
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)

        if df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.xs("Close", level=0, axis=1)
            except KeyError:
                df = df.iloc[:, 0]
        else:
            df = df["Close"] if "Close" in df.columns else df.iloc[:, 0]

        return df if not isinstance(df, pd.DataFrame) else df.iloc[:, 0]
    except Exception:
        return None


def fetch_stooq_data(ticker, invert=False):
    try:
        url = f"https://stooq.com/q/d/l/?s={ticker}&i=d"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=60)

        if response.status_code != 200:
            return None

        df = pd.read_csv(
            io.StringIO(response.text),
            parse_dates=["Date"],
            index_col="Date",
        )

        df.columns = [c.capitalize() for c in df.columns]

        if "Close" not in df.columns:
            return None

        df = df.sort_index().loc[START_DATE:END_DATE]
        series = pd.to_numeric(df["Close"], errors="coerce").dropna()

        if ticker == "usdtry":
            series = series.apply(lambda x: x / 1_000_000 if x > 100 else x)

        resampled = series.resample("W-FRI").last()

        if invert:
            resampled = 1 / resampled

        return resampled
    except Exception:
        return None


def read_investing_csv(filepath, col_name="v_stock"):
    try:
        df = pd.read_csv(filepath)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        if "Price" not in df.columns:
            return pd.DataFrame(columns=[col_name])
        # Always strip commas — Investing.com CSVs use comma thousands separators.
        # Don't gate on dtype: pandas 2.x may report string columns as 'str'/'string'
        # rather than 'object', which would cause the comma replacement to be skipped.
        df["Price"] = df["Price"].astype(str).str.replace(",", "", regex=False)
        series = pd.to_numeric(df["Price"], errors="coerce").loc[START_DATE:END_DATE]
        resampled = series.resample("W-FRI").last()
        return resampled.to_frame(name=col_name)
    except Exception:
        return pd.DataFrame(columns=[col_name])


def process_series(series, col_name, invert=False):
    if series is None:
        return pd.DataFrame(columns=[col_name])

    resampled = series.resample("W-FRI").last()

    if invert:
        resampled = 1 / resampled

    return resampled.to_frame(name=col_name) if isinstance(resampled, pd.Series) else resampled


def fetch_financials():
    global_df = None

    for name, ticker in GLOBAL_COMMODITIES.items():
        series = safe_download_yahoo(ticker)
        df_global = process_series(series, name)

        if global_df is None:
            global_df = df_global
        else:
            global_df = global_df.join(df_global, how="outer")

    global_df = global_df.loc[START_DATE:]

    all_rows = []
    countries = sorted(list(set(list(FX_MAP.keys()) + ["TR", "ZA", "RU", "BR", "AU"])))

    for country in tqdm(countries, desc="Fetching financials", unit="country"):
        # A. FX
        fx_df = pd.DataFrame(columns=["v_fxrate"])
        fx_override = STOOQ_OVERRIDES.get(f"{country}_FX")

        if fx_override:
            stooq_ticker, invert = fx_override
            fx_series = fetch_stooq_data(stooq_ticker, invert=invert)

            if fx_series is not None:
                fx_df = fx_series.to_frame(name="v_fxrate")
        elif country in FX_MAP:
            ticker, invert = FX_MAP[country]
            fx_df = process_series(
                safe_download_yahoo(ticker),
                "v_fxrate",
                invert=invert,
            )

        # B. STOCK
        stock_df = pd.DataFrame(columns=["v_stock"])

        if country == "ZA":
            if ZAF_MANUAL_PATH.exists():
                stock_df = read_investing_csv(ZAF_MANUAL_PATH)

        if stock_df.empty:
            stock_override = STOOQ_OVERRIDES.get(f"{country}_STOCK")

            if stock_override:
                stock_series = fetch_stooq_data(
                    stock_override[0],
                    invert=stock_override[1],
                )

                if stock_series is not None:
                    stock_df = stock_series.to_frame(name="v_stock")

        if stock_df.empty and country in STOCK_MAP:
            stock_df = process_series(
                safe_download_yahoo(STOCK_MAP[country]),
                "v_stock",
            )

        if country == "ZA" and stock_df.empty:
            stock_df = process_series(
                safe_download_yahoo("EZA"),
                "v_stock",
            )

        # C. Merge
        country_df = global_df.join(fx_df, how="outer").join(stock_df, how="outer")
        country_df = country_df.loc[START_DATE:]
        country_df["country"] = country
        country_df = country_df.reset_index().rename(columns={"index": "date", "Date": "date"})

        all_rows.append(country_df)

    # 3. Combine & Map
    final_df = pd.concat(all_rows, ignore_index=True)
    final_df["country"] = final_df["country"].map(ISO_MAP).fillna(final_df["country"])

    # 4. Final cleaning (matches old script exactly)
    final_df = final_df.sort_values(["country", "date"])

    # A. Forward fill (fix holidays) — only data columns, grouped by country
    data_cols = [c for c in ["v_fxrate", "v_stock", "v_oil", "v_copper", "v_gold", "v_vix", "v_us10y"]
                 if c in final_df.columns]
    final_df[data_cols] = final_df.groupby("country")[data_cols].ffill()

    # B. Trim incomplete end
    last_date = final_df["date"].max()
    last_week_check = final_df[final_df["date"] == last_date]
    if last_week_check[data_cols].isnull().any().any():
        final_df = final_df[final_df["date"] != last_date]

    # C. Drop rows with no FX (start gaps)
    final_df = final_df.dropna(subset=["v_fxrate"])

    final_df.to_csv(RAW_OUTPUT_PATH, index=False)

    return final_df


def create_logdiff_data(df):
    """
    Match old prep_financial.py exactly:
      - Use 'Date' column (capitalized) internally, like the old script
      - Apply np.log(x.replace(0, np.nan).clip(lower=1e-10)).diff(52)
        to BOTH global and country-specific variables, grouped by country
      - Drop rows with NaN (first 52 weeks per country)
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["country", "Date"])

    # Filter to selected countries (matches old prep_financial.py)
    df = df[df["country"].isin(COUNTRIES)]

    # Coerce all variables to numeric (the old script read from CSV, which
    # auto-inferred numeric dtypes; here we're working in-memory so columns
    # may have ended up as object dtype after the outer joins)
    for var in ALL_VARS:
        df[var] = pd.to_numeric(df[var], errors="coerce")

    df_transformed = df[["Date", "country"]].copy()

    # Transform global variables (grouped by country, exactly as in old script)
    for var in GLOBAL_VARS:
        df_transformed[var] = df.groupby("country")[var].transform(
            lambda x: np.log(x.replace(0, np.nan).clip(lower=1e-10)).diff(52)
        )

    # Transform country-specific variables
    for var in COUNTRY_VARS:
        df_transformed[var] = df.groupby("country")[var].transform(
            lambda x: np.log(x.replace(0, np.nan).clip(lower=1e-10)).diff(52)
        )

    # Drop rows with NaN (first 52 weeks per country)
    df_transformed = df_transformed.dropna()

    df_transformed.to_csv(LOGDIFF_OUTPUT_PATH, index=False)

    return df_transformed


if __name__ == "__main__":
    financials_raw = fetch_financials()
    create_logdiff_data(financials_raw)