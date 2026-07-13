import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import DATA_FINAL, DATA_OUTPUT

warnings.filterwarnings("ignore")

INPUT_PATH = DATA_FINAL / "B_monthly_panel.csv"
OUT_DIR = DATA_OUTPUT / "0_RW"

TEST_START = "2020-01"
MODEL_NAME = "0_RW"

COUNTRIES = ["BR", "CA", "FR", "DE", "IN", "ID", "IT", "JP", "MX", "RU", "ZA", "KR", "TR", "GB", "US"]


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


def save_results(all_df, model_name, out_dir, countries):
    prefix = model_name

    all_df = all_df.copy()
    all_df["model"] = model_name
    all_df["error"] = all_df["actual"] - all_df["predicted"]
    all_df["abs_error"] = all_df["error"].abs()
    all_df["sq_error"] = all_df["error"] ** 2

    test_df = all_df[all_df["set"] == "test"].copy()
    train_df = all_df[all_df["set"] == "train"].copy()

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

    df = pd.read_csv(INPUT_PATH)
    df["date"] = pd.to_datetime(df["date"])

    df = df[["date", "country", "infl_yoy"]].drop_duplicates(["date", "country"])
    df = df.sort_values(["country", "date"]).copy()
    df = df.dropna(subset=["infl_yoy"])

    df["predicted"] = df.groupby("country")["infl_yoy"].shift(1)
    df = df.dropna(subset=["predicted"]).copy()
    df = df.rename(columns={"infl_yoy": "actual"})

    test_mask = df["date"] >= pd.Timestamp(TEST_START)
    df.loc[test_mask, "set"] = "test"
    df.loc[~test_mask, "set"] = "train"

    save_results(df, MODEL_NAME, OUT_DIR, COUNTRIES)

    print(f"Done. Results saved to: {OUT_DIR}")