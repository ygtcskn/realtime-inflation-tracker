import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.filters.hp_filter import hpfilter

from config import DATA_TEMP, OUTPUT_TABLES

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

warnings.filterwarnings("ignore")

INPUT_MONTHLY = DATA_TEMP / "3_monthly_clean.csv"
INPUT_WEEKLY = DATA_TEMP / "3_weekly_clean.csv"

OUT_MONTHLY = DATA_TEMP / "4_monthly_detrended_final.csv"
OUT_WEEKLY = DATA_TEMP / "4_weekly_detrended_final.csv"

PC1_DIR = DATA_TEMP / "preprocessing" / "d_detrend"

COUNTRIES = ["BR", "CA", "FR", "DE", "IN", "ID", "IT", "JP", "MX", "RU", "ZA", "KR", "TR", "GB", "US"]

HP_LAMBDA_MONTHLY = 129600
HP_LAMBDA_WEEKLY = 45697600

def apply_transform(df, inverse=False):
    result = pd.DataFrame(index=df.index)
    func = np.exp if inverse else np.log

    for col in df.columns:
        try:
            result[col] = func(df[col].values)
        except Exception:
            result[col] = df[col].values

    return result


def detrend_country_series(df_country, hp_lambda):
    if df_country.empty:
        return None, None

    df_country = df_country.loc[:, (df_country != df_country.iloc[0]).any()]

    if df_country.empty:
        return None, None

    df_log = apply_transform(df_country, inverse=False)

    trends = {}

    for col in df_log.columns:
        _, trend = hpfilter(df_log[col], lamb=hp_lambda)
        trends[col] = trend

    df_trend = pd.DataFrame(trends, index=df_log.index)

    scaler = StandardScaler()
    trend_scaled = scaler.fit_transform(df_trend)

    pca = PCA(n_components=1)
    pc1 = pd.Series(pca.fit_transform(trend_scaled)[:, 0], index=df_trend.index)

    avg_trend_std = df_trend.mean(axis=1).std()

    if pc1.std() < 1e-6:
        pc1_rescaled = pc1 * 0
    else:
        pc1_rescaled = (pc1 - pc1.mean()) / pc1.std() * avg_trend_std

    df_filtered_log = df_log.sub(pc1_rescaled, axis=0)
    df_final = apply_transform(df_filtered_log, inverse=True)

    scale_factors = df_country.mean() / df_final.mean().replace(0, np.nan)
    df_final = df_final * scale_factors.fillna(1.0)

    return df_final, pc1_rescaled


def process_dataset(input_path, output_path, hp_lambda, freq_label):
    df_in = pd.read_csv(input_path, index_col=0, parse_dates=True)
    detrended_dfs = []

    for country in tqdm(COUNTRIES, desc=f"Detrending ({freq_label})", unit="country"):
        cols = [c for c in df_in.columns if c.startswith(f"{country}_")]

        if not cols:
            continue

        df_clean, pc1 = detrend_country_series(df_in[cols], hp_lambda)

        if df_clean is not None:
            detrended_dfs.append(df_clean)

            pc1_df = pd.DataFrame({
                "date": pc1.index,
                "pc1_trend": pc1.values
            })

            pc1_df.to_csv(
                PC1_DIR / f"{country}_{freq_label.lower()}_pc1_trend.csv",
                index=False
            )

    if not detrended_dfs:
        return pd.DataFrame()

    df_final = pd.concat(detrended_dfs, axis=1).sort_index()
    df_final.to_csv(output_path)

    return df_final


if __name__ == "__main__":
    DATA_TEMP.mkdir(parents=True, exist_ok=True)
    PC1_DIR.mkdir(parents=True, exist_ok=True)

    process_dataset(
        INPUT_MONTHLY,
        OUT_MONTHLY,
        HP_LAMBDA_MONTHLY,
        "Monthly"
    )

    process_dataset(
        INPUT_WEEKLY,
        OUT_WEEKLY,
        HP_LAMBDA_WEEKLY,
        "Weekly"
    )

    print(f"Done. Detrended files saved to: {DATA_TEMP}")