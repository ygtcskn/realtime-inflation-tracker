import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from config import DATA_FINAL, DATA_OUTPUT

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

warnings.filterwarnings("ignore")

INPUT_PATH = DATA_FINAL / "B_panel.csv"
OUT_DIR = DATA_OUTPUT / "B_XGB"

TEST_START = "2020-01"
HORIZONS = [0]
MIN_TRAIN = 60

XGB_PARAMS = {
    "n_estimators": 142,
    "max_depth": 4,
    "learning_rate": 0.043587256493138944,
    "subsample": 0.7907935569219802,
    "colsample_bytree": 0.9671853292946282,
    "min_child_weight": 10,
    "reg_alpha": 8.14607862876297,
    "reg_lambda": 0.14989126947551112,
    "gamma": 0.4403080947379019,
    "random_state": 42,
    "verbosity": 0,
}

COUNTRIES = ["BR", "CA", "FR", "DE", "IN", "ID", "IT", "JP", "MX", "RU", "ZA", "KR", "TR", "GB", "US"]
RANDOM_SEED = 42
MODEL_NAME = "B_XGB"


def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)


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


def save_results(test_df, train_fit_df, model_name, out_dir, countries):
    prefix = model_name

    test_df = test_df.copy()
    test_df["model"] = model_name
    test_df["set"] = "test"
    test_df["error"] = test_df["actual"] - test_df["predicted"]
    test_df["abs_error"] = test_df["error"].abs()
    test_df["sq_error"] = test_df["error"] ** 2

    train_df = pd.DataFrame()

    if train_fit_df is not None and len(train_fit_df) > 0:
        train_df = train_fit_df.copy()
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
    set_seed()

    df = pd.read_csv(INPUT_PATH)
    df["date"] = pd.to_datetime(df["date"])

    for col in [c for c in df.columns if c.startswith("C_")]:
        if df[col].dtype == "object":
            df[col] = df[col].map({
                "True": 1,
                "False": 0,
                True: 1,
                False: 0
            }).fillna(0).astype(int)

        elif df[col].dtype == "bool":
            df[col] = df[col].astype(int)

    exclude = ["date", "country", "infl_yoy"]
    ar_lag_cols = sorted([c for c in df.columns if c.startswith("infl_yoy_lag")])

    feature_cols = [
        c for c in df.columns
        if c not in exclude
        and not c.startswith("C_")
        and c not in ar_lag_cols
    ]

    dummy_cols = [c for c in df.columns if c.startswith("C_")]

    continuous_cols = feature_cols + ar_lag_cols

    for col in continuous_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[continuous_cols] = df[continuous_cols].fillna(0).astype(np.float32)
    df[dummy_cols] = df[dummy_cols].fillna(0).astype(np.float32)

    df = df.sort_values(["country", "date"]).copy()

    all_results = []
    train_fit_rows = []

    for h in HORIZONS:
        df_h = df.copy()
        df_h["target"] = df_h.groupby("country")["infl_yoy"].shift(-h)
        df_h = df_h.dropna(subset=["target"])

        test_dates = sorted(df_h[df_h["date"] >= pd.Timestamp(TEST_START)]["date"].unique())

        for t_date in tqdm(test_dates, desc=f"XGB h={h}", unit="date"):
            train_data = df_h[df_h["date"] < t_date].copy()
            test_data = df_h[df_h["date"] == t_date].copy()

            if len(train_data) < MIN_TRAIN or len(test_data) == 0:
                continue

            scaler = StandardScaler()

            X_train_cont = scaler.fit_transform(train_data[continuous_cols].values)
            X_test_cont = scaler.transform(test_data[continuous_cols].values)

            X_train = np.hstack([X_train_cont, train_data[dummy_cols].values])
            X_test = np.hstack([X_test_cont, test_data[dummy_cols].values])

            y_train = train_data["target"].values
            y_test = test_data["target"].values

            model = xgb.XGBRegressor(**XGB_PARAMS)
            model.fit(X_train, y_train)

            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)

            for j in range(len(test_data)):
                all_results.append({
                    "date": test_data.iloc[j]["date"],
                    "country": test_data.iloc[j]["country"],
                    "horizon": h,
                    "actual": y_test[j],
                    "predicted": y_pred_test[j]
                })

            for j in range(len(train_data)):
                train_fit_rows.append({
                    "forecast_origin": t_date,
                    "date": train_data.iloc[j]["date"],
                    "country": train_data.iloc[j]["country"],
                    "horizon": h,
                    "actual": y_train[j],
                    "predicted": y_pred_train[j]
                })

    res_df = pd.DataFrame(all_results)
    res_df["date"] = pd.to_datetime(res_df["date"])

    train_fit_df = pd.DataFrame(train_fit_rows)

    if len(train_fit_df) > 0:
        train_fit_df["date"] = pd.to_datetime(train_fit_df["date"])
        train_fit_df["forecast_origin"] = pd.to_datetime(train_fit_df["forecast_origin"])
        train_fit_df = train_fit_df.sort_values("forecast_origin")
        train_fit_df = train_fit_df.drop_duplicates(
            subset=["date", "country", "horizon"],
            keep="last"
        )

    save_results(res_df, train_fit_df, MODEL_NAME, OUT_DIR, COUNTRIES)

    print(f"Done. Results saved to: {OUT_DIR}")