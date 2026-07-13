import numpy as np
import pandas as pd
from scipy import stats

from config import DATA_OUTPUT, OUTPUT_TABLES


TABLES_DIR = DATA_OUTPUT
OUT_DIR = OUTPUT_TABLES

MODELS = {
    "B_LSTM":              ("LSTM",     None),
    "B_LASSO":             ("LASSO",    None),
    "B_XGB":               ("XGBoost",  None),
    "0_AR":                ("AR",       "AR"),
    "0_RW":                ("RW",       None),
    "C_onemodel_LSTM":     ("C_model",  None),
    "D_weekspecific_LSTM": ("D_model",  None),
}

COUNTRIES = ["BR", "CA", "FR", "DE", "IN", "ID", "IT", "JP",
             "MX", "RU", "ZA", "KR", "TR", "GB", "US"]

H = 1
OUTPUT_NAME = "Table_10_DM.xlsx"


def dm_test(loss_diffs, h=1):
    d = np.asarray(loss_diffs, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)

    if n < 10:
        return np.nan, np.nan, n

    d_mean = d.mean()
    gamma_0 = np.var(d, ddof=0)
    var_d = gamma_0
    for k in range(1, h):
        gamma_k = np.mean((d[:-k] - d_mean) * (d[k:] - d_mean))
        var_d += 2 * gamma_k

    if var_d <= 0:
        return np.nan, np.nan, n

    dm_raw = d_mean / np.sqrt(var_d / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm = dm_raw * hln
    p_value = 2 * (1 - stats.t.cdf(np.abs(dm), df=n - 1))

    return dm, p_value, n


def sig_marker(p):
    if np.isnan(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def load_predictions(folder_name, file_prefix=None):
    if file_prefix is None:
        file_prefix = folder_name
    path = TABLES_DIR / folder_name / f"{file_prefix}_test_predictions.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    if "horizon" in df.columns and df["horizon"].nunique() > 1:
        df = df[df["horizon"] == H]
    df = df.sort_values(["country", "date"]).reset_index(drop=True)
    return df


def align_pair(df1, df2, name1, name2):
    keys = ["date", "country"]
    if "horizon" in df1.columns and "horizon" in df2.columns:
        keys.append("horizon")
    cols = keys + ["sq_error"]
    return pd.merge(
        df1[cols], df2[cols],
        on=keys,
        suffixes=(f"_{name1}", f"_{name2}"),
        how="inner",
    )


def country_avg_rmse(merged, model_name):
    """Compute RMSE per country, then average across countries (equal weights)."""
    col = f"sq_error_{model_name}"
    per_country_rmse = merged.groupby("country")[col].apply(
        lambda s: float(np.sqrt(np.mean(s)))
    )
    return float(per_country_rmse.mean())


def country_avg_loss_series(merged, model_name):
    """For each date, average squared errors across countries.
    Returns a time-indexed series of country-averaged losses."""
    col = f"sq_error_{model_name}"
    return merged.groupby("date")[col].mean().sort_index()


def country_avg_dm(merged, m1, m2):
    """DM test on the per-date country-averaged loss differential."""
    l1 = country_avg_loss_series(merged, m1)
    l2 = country_avg_loss_series(merged, m2)
    joined = pd.concat([l1, l2], axis=1, join="inner")
    joined.columns = ["l1", "l2"]
    d = (joined["l1"] - joined["l2"]).values
    return dm_test(d, h=H)


def build_country_cd_table(preds):
    if "C_model" not in preds or "D_model" not in preds:
        return pd.DataFrame()

    merged = align_pair(preds["C_model"], preds["D_model"], "C_model", "D_model")
    rows = []

    for country in COUNTRIES:
        sub = merged[merged["country"] == country]
        if len(sub) == 0:
            rows.append({
                "Country": country, "n": 0,
                "RMSE C": np.nan, "RMSE D": np.nan,
                "DM": np.nan, "p": np.nan, "sig": "", "Result": "no data",
            })
            continue

        l_c = sub["sq_error_C_model"].values
        l_d = sub["sq_error_D_model"].values
        d = l_c - l_d

        dm, p, n = dm_test(d, h=H)
        rmse_c = float(np.sqrt(np.mean(l_c)))
        rmse_d = float(np.sqrt(np.mean(l_d)))

        if np.isnan(dm):
            verdict = "insufficient"
        elif p < 0.05:
            verdict = "C better" if dm < 0 else "D better"
        else:
            verdict = "ns"

        rows.append({
            "Country": country,
            "n": n,
            "RMSE C": rmse_c,
            "RMSE D": rmse_d,
            "DM": dm,
            "p": p,
            "sig": sig_marker(p),
            "Result": verdict,
        })

    # Country-averaged summary row
    rmse_c_cavg = country_avg_rmse(merged, "C_model")
    rmse_d_cavg = country_avg_rmse(merged, "D_model")
    dm, p, n = country_avg_dm(merged, "C_model", "D_model")

    if np.isnan(dm):
        verdict = "insufficient"
    elif p < 0.05:
        verdict = "C better" if dm < 0 else "D better"
    else:
        verdict = "ns"

    rows.append({
        "Country": "AVG (country-averaged)",
        "n": n,
        "RMSE C": rmse_c_cavg,
        "RMSE D": rmse_d_cavg,
        "DM": dm,
        "p": p,
        "sig": sig_marker(p),
        "Result": verdict,
    })

    return pd.DataFrame(rows)


def build_pooled_table(preds):
    """Model comparisons using country-averaged RMSEs and per-date country-averaged DM."""
    pairs = [
        ("LSTM", "AR"),
        ("LSTM", "RW"),
        ("LSTM", "LASSO"),
        ("LSTM", "XGBoost"),
    ]

    rows = []
    for m1, m2 in pairs:
        if m1 not in preds or m2 not in preds:
            rows.append({
                "Model 1": m1, "Model 2": m2, "n": 0,
                "RMSE 1 (cavg)": np.nan, "RMSE 2 (cavg)": np.nan,
                "DM": np.nan, "p": np.nan, "sig": "", "Result": "missing",
            })
            continue

        merged = align_pair(preds[m1], preds[m2], m1, m2)
        if len(merged) == 0:
            rows.append({
                "Model 1": m1, "Model 2": m2, "n": 0,
                "RMSE 1 (cavg)": np.nan, "RMSE 2 (cavg)": np.nan,
                "DM": np.nan, "p": np.nan, "sig": "", "Result": "no overlap",
            })
            continue

        rmse_1 = country_avg_rmse(merged, m1)
        rmse_2 = country_avg_rmse(merged, m2)
        dm, p, n = country_avg_dm(merged, m1, m2)

        if np.isnan(dm):
            verdict = "insufficient"
        elif p < 0.05:
            verdict = f"{m1} better" if dm < 0 else f"{m2} better"
        else:
            verdict = "ns"

        rows.append({
            "Model 1": m1,
            "Model 2": m2,
            "n": n,
            "RMSE 1 (cavg)": rmse_1,
            "RMSE 2 (cavg)": rmse_2,
            "DM": dm,
            "p": p,
            "sig": sig_marker(p),
            "Result": verdict,
        })

    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    preds = {}
    for folder, (label, prefix) in MODELS.items():
        try:
            preds[label] = load_predictions(folder, prefix)
            print(f"Loaded {label}: {len(preds[label])} rows")
        except FileNotFoundError as e:
            print(f"! Missing {label}: {e}")

    df_country = build_country_cd_table(preds)
    df_pooled = build_pooled_table(preds)

    print("\n--- C vs D per country (+ country-averaged row) ---")
    print(df_country.to_string(index=False))

    print("\n--- Country-averaged model comparisons ---")
    print(df_pooled.to_string(index=False))

    out_path = OUT_DIR / OUTPUT_NAME
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_country.to_excel(writer, sheet_name="C_vs_D_per_country", index=False)
        df_pooled.to_excel(writer, sheet_name="Model_comparisons_cavg", index=False)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()