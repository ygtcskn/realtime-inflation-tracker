import re
import warnings

import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from config import DATA_FINAL, DATA_OUTPUT

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

warnings.filterwarnings("ignore")

INPUT_PATH = DATA_FINAL / "B_panel.csv"
OUT_DIR = DATA_OUTPUT / "B_LSTM" / "shap_rolling"

TEST_START = "2020-01"
HORIZON = 0
MIN_TRAIN = 60

HIDDEN_SIZE = 128
NUM_LAYERS = 2
FC_HIDDEN = 128
SEQ_LEN = 23
DROPOUT = 0.06118313036234152
LR = 0.0010912874710507497
BATCH_SIZE = 16
EPOCHS = 100
WEIGHT_DECAY = 0.00039678762125960887
GRAD_CLIP = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

WINDOW_STEP = 1
N_BACKGROUND = 300
N_EXPLAIN = None
TOP_N = 20

COUNTRIES = ["BR", "CA", "FR", "DE", "IN", "ID", "IT", "JP", "MX", "RU", "ZA", "KR", "TR", "GB", "US"]


class PanelDataset(Dataset):
    def __init__(self, X_seq, X_static, y):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.X_static = torch.tensor(X_static, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X_seq[i], self.X_static[i], self.y[i]


class LSTMModel(nn.Module):
    def __init__(self, n_seq, n_static, hidden, layers, dropout, fc_hidden):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_seq,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden + n_static, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1)
        )

    def forward(self, x_seq, x_static):
        out, _ = self.lstm(x_seq)
        return self.fc(torch.cat([out[:, -1, :], x_static], dim=1)).squeeze(1)


class LSTMWrapper(nn.Module):
    def __init__(self, model, seq_len, n_seq):
        super().__init__()

        self.model = model
        self.seq_len = seq_len
        self.n_seq = n_seq
        self.seq_size = seq_len * n_seq

    def forward(self, x):
        x_seq = x[:, :self.seq_size].reshape(-1, self.seq_len, self.n_seq)
        x_static = x[:, self.seq_size:]
        return self.model(x_seq, x_static).unsqueeze(1)


def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_sequences(df, seq_cols, static_cols, h, seq_len):
    data = []
    df = df.sort_values(["country", "date"]).copy()
    df["target"] = df.groupby("country")["infl_yoy"].shift(-h)
    df = df.dropna(subset=["target"])

    for country, grp in df.groupby("country"):
        grp = grp.sort_values("date").reset_index(drop=True)

        if len(grp) <= seq_len:
            continue

        X_seq = grp[seq_cols].values.astype(np.float32)
        X_static = grp[static_cols].values.astype(np.float32)
        y = grp["target"].values.astype(np.float32)
        dates = grp["date"].values

        for i in range(len(grp) - seq_len):
            end_idx = i + seq_len - 1

            data.append({
                "date": dates[end_idx],
                "country": country,
                "X_seq": X_seq[i:i + seq_len].copy(),
                "X_static": X_static[end_idx].copy(),
                "y": float(y[end_idx])
            })

    return data


def scale_static(X_static_train, X_static_test, n_continuous):
    scaler = StandardScaler()

    train_scaled = X_static_train.copy()
    test_scaled = X_static_test.copy()

    train_scaled[:, :n_continuous] = scaler.fit_transform(X_static_train[:, :n_continuous])
    test_scaled[:, :n_continuous] = scaler.transform(X_static_test[:, :n_continuous])

    return train_scaled.astype(np.float32), test_scaled.astype(np.float32)


def train_model(X_seq, X_static, y_scaled, n_seq, n_static):
    set_seed()

    dataset = PanelDataset(
        np.ascontiguousarray(X_seq),
        np.ascontiguousarray(X_static),
        np.ascontiguousarray(y_scaled)
    )

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator
    )

    model = LSTMModel(
        n_seq,
        n_static,
        HIDDEN_SIZE,
        NUM_LAYERS,
        DROPOUT,
        FC_HIDDEN
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    criterion = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5
    )

    model.train()

    for _ in tqdm(range(EPOCHS), desc="Epoch", leave=False):
        epoch_loss = 0.0
        batches = 0

        for xb, x_static, yb in dataloader:
            xb = xb.to(DEVICE)
            x_static = x_static.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            pred = model(xb, x_static)
            loss = criterion(pred, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        scheduler.step(epoch_loss / max(batches, 1))

    return model


def flatten_inputs(X_seq, X_static):
    return np.hstack([X_seq.reshape(X_seq.shape[0], -1), X_static]).astype(np.float32)


def get_base_variable(feature):
    match = re.match(r"^(.+)_w([1-4])$", str(feature))
    return match.group(1) if match else feature


def get_week_position(feature):
    match = re.match(r"^.+_w([1-4])$", str(feature))
    return int(match.group(1)) if match else np.nan


def categorize_feature(feature):
    feature = str(feature)

    if feature.startswith("C_"):
        return "Country Dummies"

    if feature.startswith("infl_yoy_lag"):
        return "AR Lags"

    if feature.startswith("v_"):
        return "Financial Variables"

    return "Google Trends"


def aggregate_shap_over_timesteps(shap_values, n_seq, seq_len):
    n_samples = shap_values.shape[0]
    seq_size = seq_len * n_seq

    seq_shap = shap_values[:, :seq_size].reshape(n_samples, seq_len, n_seq)
    seq_agg = seq_shap.mean(axis=1)

    static_shap = shap_values[:, seq_size:]

    return np.hstack([seq_agg, static_shap])


def summarize_abs_shap(df, feature_cols):
    rows = []

    for feature in feature_cols:
        values = df[feature].abs().values

        rows.append({
            "feature": feature,
            "mean_abs_shap": np.mean(values),
            "median_abs_shap": np.median(values),
            "category": categorize_feature(feature),
            "week": get_week_position(feature),
            "base_variable": get_base_variable(feature),
        })

    return pd.DataFrame(rows)


def prepare_data():
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

    seq_cols = [
        c for c in df.columns
        if c not in exclude
        and not c.startswith("C_")
        and c not in ar_lag_cols
    ]

    dummy_cols = sorted([c for c in df.columns if c.startswith("C_")])
    static_cols = ar_lag_cols + dummy_cols

    for col in seq_cols + ar_lag_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[seq_cols] = df[seq_cols].fillna(0).astype(np.float32)
    df[ar_lag_cols] = df[ar_lag_cols].fillna(0).astype(np.float32)
    df[dummy_cols] = df[dummy_cols].fillna(0).astype(np.float32)

    return df, seq_cols, static_cols, ar_lag_cols


def run_shap():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df, seq_cols, static_cols, ar_lag_cols = prepare_data()

    full_seq = create_sequences(df, seq_cols, static_cols, HORIZON, SEQ_LEN)

    if not full_seq:
        raise ValueError("No sequences were created.")

    meta_dates = pd.to_datetime([row["date"] for row in full_seq])
    test_dates = sorted(set(date for date in meta_dates if date >= pd.Timestamp(TEST_START)))
    shap_dates = test_dates[::WINDOW_STEP]

    if not shap_dates:
        raise ValueError("No SHAP dates found.")

    feature_names = list(seq_cols) + list(static_cols)
    n_seq = len(seq_cols)
    n_static = len(static_cols)
    n_ar = len(ar_lag_cols)

    all_shap_rows = []

    for i, t_date in tqdm(enumerate(shap_dates), desc="SHAP rolling", unit="date", total=len(shap_dates)):
        train_idx = [
            j for j, row in enumerate(full_seq)
            if pd.Timestamp(row["date"]) < t_date
        ]

        test_idx = [
            j for j, row in enumerate(full_seq)
            if pd.Timestamp(row["date"]) == t_date
        ]

        if len(train_idx) < MIN_TRAIN or not test_idx:
            continue

        X_seq_train = np.stack([full_seq[j]["X_seq"] for j in train_idx]).astype(np.float32)
        X_static_train = np.stack([full_seq[j]["X_static"] for j in train_idx]).astype(np.float32)
        y_train = np.array([full_seq[j]["y"] for j in train_idx], dtype=np.float32)

        X_seq_test = np.stack([full_seq[j]["X_seq"] for j in test_idx]).astype(np.float32)
        X_static_test = np.stack([full_seq[j]["X_static"] for j in test_idx]).astype(np.float32)
        countries_test = np.array([full_seq[j]["country"] for j in test_idx])

        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten().astype(np.float32)

        seq_scaler = StandardScaler()
        n_obs, seq_len, n_features = X_seq_train.shape

        X_seq_train_scaled = seq_scaler.fit_transform(
            X_seq_train.reshape(-1, n_features)
        ).reshape(n_obs, seq_len, n_features).astype(np.float32)

        X_seq_test_scaled = seq_scaler.transform(
            X_seq_test.reshape(-1, n_features)
        ).reshape(X_seq_test.shape[0], seq_len, n_features).astype(np.float32)

        X_static_train_scaled, X_static_test_scaled = scale_static(
            X_static_train,
            X_static_test,
            n_ar
        )

        model = train_model(
            X_seq_train_scaled,
            X_static_train_scaled,
            y_train_scaled,
            n_features,
            n_static
        )

        model.eval()

        wrapper = LSTMWrapper(model, SEQ_LEN, n_seq).to(DEVICE)
        wrapper.eval()

        X_train_flat = flatten_inputs(X_seq_train_scaled, X_static_train_scaled)
        X_test_flat = flatten_inputs(X_seq_test_scaled, X_static_test_scaled)

        np.random.seed(RANDOM_SEED + i)

        background_idx = np.random.choice(
            len(X_train_flat),
            min(N_BACKGROUND, len(X_train_flat)),
            replace=False
        )

        background = torch.tensor(X_train_flat[background_idx]).to(DEVICE)

        if N_EXPLAIN is not None:
            explain_idx = np.random.choice(
                len(X_test_flat),
                min(N_EXPLAIN, len(X_test_flat)),
                replace=False
            )
        else:
            explain_idx = np.arange(len(X_test_flat))

        explain = torch.tensor(X_test_flat[explain_idx]).to(DEVICE)

        with torch.backends.cudnn.flags(enabled=False):
            explainer = shap.GradientExplainer(wrapper, background)
            shap_values = explainer.shap_values(explain)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = np.array(shap_values)

        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 0]

        shap_values_agg = aggregate_shap_over_timesteps(
            shap_values,
            n_seq,
            SEQ_LEN
        )

        for sample_idx in range(len(explain_idx)):
            row = {
                "date": pd.Timestamp(t_date),
                "country": countries_test[explain_idx[sample_idx]],
            }

            for feature_idx, feature_name in enumerate(feature_names):
                row[feature_name] = shap_values_agg[sample_idx, feature_idx]

            all_shap_rows.append(row)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    shap_df = pd.DataFrame(all_shap_rows)

    if shap_df.empty:
        raise ValueError("SHAP output is empty.")

    shap_df["date"] = pd.to_datetime(shap_df["date"])

    shap_df.to_csv(OUT_DIR / "B_LSTM_shap_rolling_raw.csv", index=False)

    overall = summarize_abs_shap(shap_df, feature_names)
    overall = overall.sort_values("mean_abs_shap", ascending=False)
    overall.to_csv(OUT_DIR / "B_LSTM_shap_overall.csv", index=False)

    base_importance = (
        overall.groupby("base_variable")
        .agg(
            total_mean_shap=("mean_abs_shap", "sum"),
            avg_mean_shap=("mean_abs_shap", "mean"),
            total_median_shap=("median_abs_shap", "sum"),
            avg_median_shap=("median_abs_shap", "mean"),
            n_weeks=("mean_abs_shap", "count"),
            category=("category", "first"),
        )
        .sort_values("total_mean_shap", ascending=False)
        .reset_index()
    )

    base_importance.to_csv(OUT_DIR / "B_LSTM_shap_base_variables.csv", index=False)

    base_importance_exogenous = base_importance[
        ~base_importance["base_variable"].str.startswith("C_")
        & ~base_importance["base_variable"].str.startswith("infl_yoy_lag")
    ].copy()

    base_importance_exogenous.to_csv(
        OUT_DIR / "B_LSTM_shap_base_variables_exogenous.csv",
        index=False
    )

    weekly_features = overall[overall["week"].notna()].copy()

    week_agg = (
        weekly_features.groupby("week")
        .agg(
            mean_of_means=("mean_abs_shap", "mean"),
            sum_of_means=("mean_abs_shap", "sum"),
            mean_of_medians=("median_abs_shap", "mean"),
            sum_of_medians=("median_abs_shap", "sum"),
            count=("mean_abs_shap", "count"),
        )
        .reset_index()
    )

    week_agg.to_csv(OUT_DIR / "B_LSTM_shap_week_aggregate.csv", index=False)

    week_decomp_rows = []

    for base_variable in base_importance_exogenous.head(TOP_N)["base_variable"]:
        for week in [1, 2, 3, 4]:
            feature = f"{base_variable}_w{week}"

            if feature in overall["feature"].values:
                row = overall[overall["feature"] == feature].iloc[0]

                week_decomp_rows.append({
                    "base_variable": base_variable,
                    "week": week,
                    "mean_abs_shap": row["mean_abs_shap"],
                    "median_abs_shap": row["median_abs_shap"],
                })

    week_decomp = pd.DataFrame(week_decomp_rows)
    week_decomp.to_csv(OUT_DIR / "B_LSTM_shap_week_decomposition.csv", index=False)

    shap_df["year_month"] = shap_df["date"].dt.to_period("M").astype(str)

    time_mean = (
        shap_df.groupby("year_month")[feature_names]
        .apply(lambda x: x.abs().mean())
        .reset_index()
    )

    time_median = (
        shap_df.groupby("year_month")[feature_names]
        .apply(lambda x: x.abs().median())
        .reset_index()
    )

    time_mean.to_csv(OUT_DIR / "B_LSTM_shap_time_varying_mean.csv", index=False)
    time_median.to_csv(OUT_DIR / "B_LSTM_shap_time_varying_median.csv", index=False)

    country_rows = []

    for country in COUNTRIES:
        country_data = shap_df[shap_df["country"] == country]

        if country_data.empty:
            continue

        for feature in feature_names:
            values = country_data[feature].abs().values

            country_rows.append({
                "country": country,
                "feature": feature,
                "mean_abs_shap": np.mean(values),
                "median_abs_shap": np.median(values),
                "category": categorize_feature(feature),
                "base_variable": get_base_variable(feature),
            })

    country_df = pd.DataFrame(country_rows)
    country_df.to_csv(OUT_DIR / "B_LSTM_shap_by_country.csv", index=False)

    return shap_df


if __name__ == "__main__":
    set_seed()
    run_shap()
    print(f"Done. SHAP outputs saved to: {OUT_DIR}")