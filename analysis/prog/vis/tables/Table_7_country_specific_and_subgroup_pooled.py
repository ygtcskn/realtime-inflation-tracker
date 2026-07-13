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

B_LSTM_COUSPE_DIR = DATA_OUTPUT / "B_LSTM_couspe"

COUNTRY_SPECIFIC_FILE = (
    B_LSTM_COUSPE_DIR
    / "country_specific"
    / "B_LSTM_couspe_country_specific_all_metrics.csv"
)

SUBGROUP_POOLED_DIR = B_LSTM_COUSPE_DIR / "subgroup_pooled"

COUNTRIES = [
    "BR", "CA", "FR", "DE", "IN", "ID", "IT", "JP",
    "MX", "RU", "ZA", "KR", "TR", "GB", "US"
]

SUBGROUPS = {
    "G7": ["CA", "FR", "DE", "IT", "JP", "GB", "US"],
    "BRICS": ["BR", "RU", "IN", "ZA"],
    "Emerging": ["BR", "IN", "ID", "MX", "RU", "ZA", "KR", "TR"],
    "Advanced": ["CA", "FR", "DE", "IT", "JP", "GB", "US"],
    "Europe": ["FR", "DE", "IT", "GB"],
    "Asia": ["IN", "ID", "JP", "KR"],
    "Americas": ["BR", "CA", "MX", "US"],
}


def load_metrics(folder_name):
    folder = TABLES_DIR / folder_name

    if not folder.exists():
        print(f"Missing folder: {folder}")
        return None

    files = [
        f for f in folder.iterdir()
        if "metric" in f.name.lower() and f.suffix.lower() == ".csv"
    ]

    if not files:
        print(f"No metrics file found in: {folder}")
        return None

    df = pd.read_csv(files[0])
    df.columns = df.columns.str.lower()

    return df


def load_metrics_file(path):
    if not path.exists():
        print(f"Missing metrics file: {path}")
        return None

    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()

    return df


def load_metrics_from_path(folder):
    if not folder.exists():
        print(f"Missing folder: {folder}")
        return None

    files = [
        f for f in folder.rglob("*.csv")
        if "metric" in f.name.lower()
    ]

    if not files:
        print(f"No metrics file found in: {folder}")
        return None

    df = pd.read_csv(files[0])
    df.columns = df.columns.str.lower()

    return df


def get_country_value(mdf, country, column):
    if mdf is None or column not in mdf.columns:
        return np.nan

    if "country" not in mdf.columns:
        return np.nan

    rows = mdf[mdf["country"].astype(str).str.upper() == country.upper()]

    if len(rows) == 0:
        return np.nan

    if "scope" in mdf.columns:
        scope_priority = [
            "country",
            "test_country",
            "overall",
            "test_overall",
        ]

        for scope in scope_priority:
            scoped = rows[rows["scope"].astype(str).str.lower() == scope]

            if len(scoped) > 0:
                rows = scoped
                break

    return float(rows[column].values[0]) if len(rows) > 0 else np.nan


def get_overall_value(mdf, column):
    if mdf is None or column not in mdf.columns:
        return np.nan

    if "scope" in mdf.columns:
        scope_priority = [
            "overall",
            "test_overall",
            "country",
            "test_country",
        ]

        for scope in scope_priority:
            scoped = mdf[mdf["scope"].astype(str).str.lower() == scope]

            if len(scoped) > 0:
                return float(scoped[column].values[0])

    if len(mdf) > 0:
        return float(mdf[column].values[0])

    return np.nan


def avg_country_value(mdf, countries, column):
    values = [get_country_value(mdf, country, column) for country in countries]
    values = [value for value in values if pd.notna(value)]

    return float(np.mean(values)) if values else np.nan


def get_subgroup_pooled_rmse(group_name, group_countries):
    group_folder = SUBGROUP_POOLED_DIR / group_name
    mdf = load_metrics_from_path(group_folder)

    if mdf is None:
        return np.nan

    value = get_overall_value(mdf, "rmse")

    if pd.isna(value):
        value = avg_country_value(mdf, group_countries, "rmse")

    return value


def save_excel(df, filename):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    path = OUT_DIR / filename
    df.to_excel(path, index=False)

    print(f"Saved: {path}")


def create_table_15():
    m_pooled = load_metrics("B_LSTM")
    m_country_specific = load_metrics_file(COUNTRY_SPECIFIC_FILE)

    rows = []

    for country in tqdm(COUNTRIES, desc="Table 7 (countries)", unit="country"):
        pooled = get_country_value(m_pooled, country, "rmse")
        country_specific = get_country_value(m_country_specific, country, "rmse")

        if pd.notna(pooled) and pd.notna(country_specific):
            winner = "Pooled" if pooled <= country_specific else "Country-specific"
        else:
            winner = np.nan

        rows.append({
            "Country / Group": country,
            "Pooled LSTM": pooled,
            "Alternative LSTM": country_specific,
            "Alternative type": "Country-specific",
            "Winner": winner,
        })

    for group_name, group_countries in SUBGROUPS.items():
        pooled = avg_country_value(m_pooled, group_countries, "rmse")
        subgroup_pooled = get_subgroup_pooled_rmse(group_name, group_countries)

        if pd.notna(pooled) and pd.notna(subgroup_pooled):
            winner = "Pooled" if pooled <= subgroup_pooled else "Subgroup-pooled"
        else:
            winner = np.nan

        rows.append({
            "Country / Group": group_name,
            "Pooled LSTM": pooled,
            "Alternative LSTM": subgroup_pooled,
            "Alternative type": "Subgroup-pooled",
            "Winner": winner,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    save_excel(create_table_15(), "Table_7_country_specific_and_subgroup_pooled.xlsx")
    print("Done.")