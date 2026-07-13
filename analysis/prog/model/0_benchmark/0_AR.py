import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import DATA_FINAL, DATA_OUTPUT

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

warnings.filterwarnings("ignore")

INFLATION_DIR = DATA_FINAL
OUT_DIR = DATA_OUTPUT / "0_AR"
INPUT_INFLATION = INFLATION_DIR / "inflation.csv"

AR_LAGS = [1]
TEST_START = "2020-01"
HORIZONS = [0]
MIN_TRAIN = 60
DATE_CUTOFF = "2025-10-01"
START_DATE = "2005-01-01"

COUNTRIES = ["BR", "CA", "FR", "DE", "IN", "ID", "IT", "JP", "MX", "RU", "ZA", "KR", "TR", "GB", "US"]

CODE_MAP_3TO2 = {
    "ARG": "AR", "AUS": "AU", "BRA": "BR", "CAN": "CA", "CHN": "CN",
    "FRA": "FR", "DEU": "DE", "IND": "IN", "IDN": "ID", "ITA": "IT",
    "JPN": "JP", "MEX": "MX", "RUS": "RU", "SAU": "SA", "ZAF": "ZA",
    "KOR": "KR", "TUR": "TR", "GBR": "GB", "USA": "US"
}

MODEL_NAME = "AR"


def load_inflation(path):
    df = pd.read_csv(path)

    if "year_month" in df.columns:
        df["date"] = pd.to_datetime(df["year_month"]) + pd.offsets.MonthEnd(0)
    else:
        df["date"] = pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0)

    if "infl_yoy" not in df.columns and "value" in df.columns:
        df = df.rename(columns={"value": "infl_yoy"})

    if df["country"].str.len().max() == 3:
        df["country"] = df["country"].map(CODE_MAP_3TO2)

    df = df[df["country"].isin(COUNTRIES)]
    df = df[["date", "country", "infl_yoy"]].dropna()
    df = df.sort_values(["country", "date"]).reset_index(drop=True)

    return df


def create_ar_panel(df, lags, horizon=0):
    df = df.sort_values(["country", "date"]).copy()

    if horizon > 0:
        df["target"] = df.groupby("country")["infl_yoy"].shift(-horizon)
    else:
        df["target"] = df["infl_yoy"]

    for lag in lags:
        df[f"lag{lag}"] = df.groupby("country")["infl_yoy"].shift(lag)

    return df.dropna()


def compute_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "R2": np.nan, "MAPE": np.nan, "N": 0}

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan

    mask = y_true != 0
    mape = (
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        if mask.sum() > 0
        else np.nan
    )

    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape, "N": len(y_true)}


def save_results(test_df, train_fits, df_actual, model_name, out_dir, countries):
    prefix = model_name

    test_df = test_df.copy()
    test_df["model"] = model_name
    test_df["set"] = "test"
    test_df["error"] = test_df["actual"] - test_df["predicted"]
    test_df["abs_error"] = test_df["error"].abs()
    test_df["sq_error"] = test_df["error"] ** 2

    train_rows = []

    for key, fits in train_fits.items():
        country = key[0]
        h = key[-1]

        for date, pred in fits.items():
            actual_row = df_actual[
                (df_actual["country"] == country)
                & (df_actual["date"] == pd.Timestamp(date))
            ]

            actual_val = actual_row["infl_yoy"].values[0] if len(actual_row) > 0 else np.nan

            train_rows.append({
                "date": pd.Timestamp(date),
                "country": country,
                "horizon": h,
                "actual": actual_val,
                "predicted": pred
            })

    train_df = pd.DataFrame(train_rows)

    if len(train_df) > 0:
        train_df["model"] = model_name
        train_df["set"] = "train"
        train_df["error"] = train_df["actual"] - train_df["predicted"]
        train_df["abs_error"] = train_df["error"].abs()
        train_df["sq_error"] = train_df["error"] ** 2

    test_df.to_csv(out_dir / f"{prefix}_test_predictions.csv", index=False)

    if len(train_df) > 0:
        train_df.to_csv(out_dir / f"{prefix}_train_predictions.csv", index=False)

    metrics_rows = []

    def add_metrics(scope, country, data):
        if len(data) == 0:
            return

        metrics = compute_metrics(data["actual"].values, data["predicted"].values)

        metrics_rows.append({
            "scope": scope,
            "country": country,
            **metrics,
            "model": model_name
        })

    add_metrics("overall", "ALL", test_df)

    for c in countries:
        add_metrics("country", c, test_df[test_df["country"] == c])

    add_metrics("turkey_analysis", "WITH_TR", test_df)
    add_metrics("turkey_analysis", "WITHOUT_TR", test_df[test_df["country"] != "TR"])

    country_rmses = []
    country_rmses_no_tr = []

    for c in countries:
        country_data = test_df[test_df["country"] == c]

        if len(country_data) > 0:
            rmse = np.sqrt(mean_squared_error(country_data["actual"], country_data["predicted"]))
            country_rmses.append(rmse)

            if c != "TR":
                country_rmses_no_tr.append(rmse)

    metrics_rows.append({
        "scope": "turkey_analysis",
        "country": "AVG_WITH_TR",
        "RMSE": np.mean(country_rmses),
        "MAE": np.nan,
        "R2": np.nan,
        "MAPE": np.nan,
        "N": len(country_rmses),
        "model": model_name
    })

    metrics_rows.append({
        "scope": "turkey_analysis",
        "country": "AVG_WITHOUT_TR",
        "RMSE": np.mean(country_rmses_no_tr),
        "MAE": np.nan,
        "R2": np.nan,
        "MAPE": np.nan,
        "N": len(country_rmses_no_tr),
        "model": model_name
    })

    if len(train_df) > 0:
        add_metrics("train", "ALL", train_df)

        for c in countries:
            add_metrics("train_country", c, train_df[train_df["country"] == c])

    add_metrics("test", "ALL", test_df)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_dir / f"{prefix}_all_metrics.csv", index=False)

    errors_pivot = test_df.pivot_table(
        index="date",
        columns="country",
        values="error",
        aggfunc="first"
    )

    sq_errors_pivot = test_df.pivot_table(
        index="date",
        columns="country",
        values="sq_error",
        aggfunc="first"
    )

    errors_pivot.to_csv(out_dir / f"{prefix}_errors_pivot.csv")
    sq_errors_pivot.to_csv(out_dir / f"{prefix}_sq_errors_pivot.csv")

    return metrics_df


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_infl = load_inflation(INPUT_INFLATION)
    df_infl = df_infl[df_infl["date"] >= pd.Timestamp(START_DATE)]
    df_infl = df_infl[df_infl["date"] < pd.Timestamp(DATE_CUTOFF)]

    all_results = []
    train_fits = defaultdict(dict)
    df_actual = df_infl[["date", "country", "infl_yoy"]].copy()
    feature_cols = [f"lag{lag}" for lag in AR_LAGS]

    for h in HORIZONS:
        df_panel = create_ar_panel(df_infl, AR_LAGS, horizon=h)
        test_dates = sorted(df_panel[df_panel["date"] >= pd.Timestamp(TEST_START)]["date"].unique())

        for t_date in tqdm(test_dates, desc=f"AR h={h}", unit="date"):
            train = df_panel[df_panel["date"] < t_date]
            test = df_panel[df_panel["date"] == t_date]

            if len(train) < MIN_TRAIN or len(test) == 0:
                continue

            X_train = train[feature_cols].values
            y_train = train["target"].values
            X_test = test[feature_cols].values

            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)

            for j, row in test.iterrows():
                all_results.append({
                    "date": row["date"],
                    "country": row["country"],
                    "horizon": h,
                    "actual": row["target"],
                    "predicted": y_pred_test[test.index.get_loc(j)]
                })

            for j, (_, row) in enumerate(train.iterrows()):
                key = (row["country"], h)
                train_fits[key][row["date"]] = y_pred_train[j]

    res_df = pd.DataFrame(all_results)
    res_df["date"] = pd.to_datetime(res_df["date"])

    save_results(res_df, train_fits, df_actual, MODEL_NAME, OUT_DIR, COUNTRIES)

    print(f"Done. Results saved to: {OUT_DIR}")