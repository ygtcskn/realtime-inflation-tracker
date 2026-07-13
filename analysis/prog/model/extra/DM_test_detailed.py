"""
Diebold-Mariano tests across model pairs.

Comparisons:
  1. UMIDAS LSTM (B) vs LASSO
  2. UMIDAS LSTM (B) vs AR
  3. UMIDAS LSTM (B) vs C model
  4. UMIDAS LSTM (B) vs D model
  5. C vs D model
  6. C vs D model — per country

Reads from:
  output/{MODEL}/{MODEL}_test_predictions.csv

with columns: date, country, horizon, actual, predicted, model, set, error,
              abs_error, sq_error

Convention:
  - DM stat NEGATIVE  → model 1 has lower loss (better forecasts)
  - DM stat POSITIVE  → model 2 has lower loss (better forecasts)
  - p < 0.05          → reject H0 of equal forecast accuracy

Loss function default: squared error (MSE-based DM). Switch with LOSS = "mae".
"""

import numpy as np
import pandas as pd
from scipy import stats

from config import DATA_OUTPUT, OUTPUT_TABLES


# ============================================================
# CONFIG
# ============================================================

# Folder name → (label, file_prefix, week_position).
# week_position filter is applied only if the column exists in the file.
# Set to None to use all rows (or for files without week_position).
MODELS = {
    "B_LSTM":              ("UMIDAS_LSTM", None, None),
    "B_LASSO":             ("LASSO",       None, None),
    "0_AR":                ("AR",          "AR", None),
    "C_onemodel_LSTM":     ("C_model",     None, 4),
    "D_weekspecific_LSTM": ("D_model",     None, 4),
}

# Forecast horizon for HLN small-sample correction.
# If horizon column has multiple values, we filter to this one.
H = 1

# Loss function: "mse" or "mae"
LOSS = "mse"

# Significance levels for the *** ** * markers
SIG_LEVELS = [(0.01, "***"), (0.05, "** "), (0.10, "*  ")]

OUT_DIR = OUTPUT_TABLES / "dm_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DM TEST
# ============================================================

def dm_test(loss_diffs, h=1):
    """
    Diebold-Mariano test with Harvey-Leybourne-Newbold (1997) small-sample
    correction. Takes the loss differential series d_t = L1_t - L2_t directly.

    Returns dict with dm_stat, p_value, n, mean_diff.
    """
    d = np.asarray(loss_diffs, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)

    if n < 10:
        return {"dm_stat": np.nan, "p_value": np.nan, "n": n, "mean_diff": np.nan}

    d_mean = d.mean()

    # Long-run variance: gamma_0 + 2 sum_{k=1}^{h-1} gamma_k
    gamma_0 = np.var(d, ddof=0)
    var_d = gamma_0
    for k in range(1, h):
        gamma_k = np.mean((d[:-k] - d_mean) * (d[k:] - d_mean))
        var_d += 2 * gamma_k

    if var_d <= 0:
        return {"dm_stat": np.nan, "p_value": np.nan, "n": n, "mean_diff": d_mean}

    dm_raw = d_mean / np.sqrt(var_d / n)

    # HLN small-sample correction
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm = dm_raw * hln

    p_value = 2 * (1 - stats.t.cdf(np.abs(dm), df=n - 1))

    return {"dm_stat": dm, "p_value": p_value, "n": n, "mean_diff": d_mean}


# ============================================================
# DATA LOADING
# ============================================================

def load_model_predictions(folder_name, file_prefix=None, week_position=None):
    if file_prefix is None:
        file_prefix = folder_name
    path = DATA_OUTPUT / folder_name / f"{file_prefix}_test_predictions.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    if "horizon" in df.columns and df["horizon"].nunique() > 1:
        df = df[df["horizon"] == H]

    if week_position is not None and "week_position" in df.columns:
        n_before = len(df)
        df = df[df["week_position"] == week_position]
        print(f"    filtered to week_position={week_position}: {n_before} → {len(df)} rows")

    df = df.sort_values(["country", "date"]).reset_index(drop=True)
    return df


def align_pair(df1, df2, name1, name2):
    keys = ["date", "country"]
    if "horizon" in df1.columns and "horizon" in df2.columns:
        keys.append("horizon")

    cols = keys + ["error", "abs_error", "sq_error"]
    merged = pd.merge(
        df1[cols], df2[cols],
        on=keys,
        suffixes=(f"_{name1}", f"_{name2}"),
        how="inner",
    )
    return merged


# ============================================================
# REPORT FORMATTING
# ============================================================

def sig_marker(p):
    for thresh, mark in SIG_LEVELS:
        if p < thresh:
            return mark
    return "   "


def format_pooled_row(name1, name2, result, loss1, loss2):
    dm = result["dm_stat"]
    p = result["p_value"]
    n = result["n"]

    if np.isnan(dm):
        return (f"{name1:>15} vs {name2:<15}  n={n:>5}  "
                f"DM=  NaN  p=  NaN          → insufficient data")

    if p < 0.05:
        verdict = f"{name1} better" if dm < 0 else f"{name2} better"
    else:
        verdict = "no significant difference"

    # Display RMSE for readability when LOSS=mse, otherwise MAE
    if LOSS == "mse":
        d1, d2, label = np.sqrt(loss1), np.sqrt(loss2), "RMSE"
    else:
        d1, d2, label = loss1, loss2, "MAE"

    return (f"{name1:>15} vs {name2:<15}  n={n:>5}  "
            f"DM={dm:+7.3f}  p={p:.4f}{sig_marker(p)}  "
            f"{label}1={d1:.4f}  {label}2={d2:.4f}  → {verdict}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 100)
    print(f"DIEBOLD-MARIANO TESTS  (loss={LOSS}, horizon={H})")
    print("=" * 100)

    preds = {}
    for folder, (label, prefix, week_pos) in MODELS.items():
        try:
            preds[label] = load_model_predictions(folder, prefix, week_pos)
            df = preds[label]
            print(f"  Loaded {label:<15}  rows={len(df):>5}  "
                  f"countries={df['country'].nunique():>3}  "
                  f"dates={df['date'].min().date()} → {df['date'].max().date()}")
        except FileNotFoundError as e:
            print(f"  ! Missing {label} ({folder}): {e}")
        except Exception as e:
            print(f"  ! Error loading {label}: {e}")

    if not preds:
        print("\n  No models loaded. Exiting.")
        return

    pairs = [
        ("UMIDAS_LSTM", "LASSO"),
        ("UMIDAS_LSTM", "AR"),
        ("UMIDAS_LSTM", "C_model"),
        ("UMIDAS_LSTM", "D_model"),
        ("C_model",     "D_model"),
    ]

    loss_col = "sq_error" if LOSS == "mse" else "abs_error"

    # ------------------------------------------------------------
    # Pooled DM tests
    # ------------------------------------------------------------
    print()
    print("=" * 100)
    print(f"POOLED DM TESTS (all countries combined, loss={LOSS})")
    print("=" * 100)
    print("Significance: *** p<0.01, ** p<0.05, * p<0.10")
    print("-" * 100)

    pooled_rows = []
    for m1, m2 in pairs:
        if m1 not in preds or m2 not in preds:
            print(f"  Skipping {m1} vs {m2}: model missing")
            continue

        merged = align_pair(preds[m1], preds[m2], m1, m2)
        if len(merged) == 0:
            print(f"  Skipping {m1} vs {m2}: no overlapping (date, country) pairs")
            continue

        l1 = merged[f"{loss_col}_{m1}"].values
        l2 = merged[f"{loss_col}_{m2}"].values
        d = l1 - l2

        result = dm_test(d, h=H)
        mean_l1 = np.nanmean(l1)
        mean_l2 = np.nanmean(l2)

        print(format_pooled_row(m1, m2, result, mean_l1, mean_l2))

        pooled_rows.append({
            "model1": m1,
            "model2": m2,
            "n": result["n"],
            "dm_stat": result["dm_stat"],
            "p_value": result["p_value"],
            "mean_loss_diff": result["mean_diff"],
            f"mean_{LOSS}_model1": mean_l1,
            f"mean_{LOSS}_model2": mean_l2,
        })

    # ------------------------------------------------------------
    # Per-country DM test for C vs D
    # ------------------------------------------------------------
    print()
    print("=" * 100)
    print("PER-COUNTRY DM TEST: C_model vs D_model")
    print("=" * 100)

    country_rows = []
    if "C_model" in preds and "D_model" in preds:
        merged_cd = align_pair(preds["C_model"], preds["D_model"], "C_model", "D_model")

        print(f"{'country':<10} {'n':>6} {'DM':>10} {'p':>10}{'sig':>4}  "
              f"{'C_'+LOSS:>14} {'D_'+LOSS:>14}  verdict")
        print("-" * 100)

        for country in sorted(merged_cd["country"].unique()):
            sub = merged_cd[merged_cd["country"] == country]
            l_c = sub[f"{loss_col}_C_model"].values
            l_d = sub[f"{loss_col}_D_model"].values
            d = l_c - l_d

            result = dm_test(d, h=H)
            mean_c = np.nanmean(l_c)
            mean_d = np.nanmean(l_d)

            if np.isnan(result["dm_stat"]):
                verdict = "insufficient data"
            elif result["p_value"] < 0.05:
                verdict = "C better" if result["dm_stat"] < 0 else "D better"
            else:
                verdict = "ns"

            print(f"{country:<10} {result['n']:>6} {result['dm_stat']:>10.3f} "
                  f"{result['p_value']:>10.4f}{sig_marker(result['p_value']):>4}  "
                  f"{mean_c:>14.4f} {mean_d:>14.4f}  {verdict}")

            country_rows.append({
                "country": country,
                "n": result["n"],
                "dm_stat": result["dm_stat"],
                "p_value": result["p_value"],
                f"C_{LOSS}": mean_c,
                f"D_{LOSS}": mean_d,
                "verdict": verdict,
            })

    # ------------------------------------------------------------
    # Save tables
    # ------------------------------------------------------------
    if pooled_rows:
        pooled_df = pd.DataFrame(pooled_rows)
        pooled_path = OUT_DIR / f"dm_pooled_{LOSS}.csv"
        pooled_df.to_csv(pooled_path, index=False)
        print(f"\nSaved pooled summary: {pooled_path}")

    if country_rows:
        country_df = pd.DataFrame(country_rows)
        country_path = OUT_DIR / f"dm_C_vs_D_per_country_{LOSS}.csv"
        country_df.to_csv(country_path, index=False)
        print(f"Saved per-country C vs D:  {country_path}")

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()