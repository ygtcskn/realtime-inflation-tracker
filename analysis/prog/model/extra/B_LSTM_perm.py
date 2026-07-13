
import re
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
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
OUT_DIR = DATA_OUTPUT / "B_LSTM_perm"

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

N_PERMUTATIONS = 50


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
            dropout=dropout if layers > 1 else 0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden + n_static, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, x_seq, x_static):
        out, _ = self.lstm(x_seq)
        return self.fc(torch.cat([out[:, -1, :], x_static], dim=1)).squeeze(1)


def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
                "y": float(y[end_idx]),
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
        np.ascontiguousarray(y_scaled),
    )

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    model = LSTMModel(
        n_seq,
        n_static,
        HIDDEN_SIZE,
        NUM_LAYERS,
        DROPOUT,
        FC_HIDDEN,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
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


def predict(model, X_seq, X_static):
    model.eval()

    with torch.no_grad():
        preds = model(
            torch.tensor(X_seq, dtype=torch.float32).to(DEVICE),
            torch.tensor(X_static, dtype=torch.float32).to(DEVICE),
        )

    return preds.cpu().numpy()


def permutation_importance(model, X_seq, X_static, y_true, seq_cols, static_cols, y_scaler):
    y_pred_base = y_scaler.inverse_transform(
        predict(model, X_seq, X_static).reshape(-1, 1)
    ).flatten()

    base_rmse = np.sqrt(mean_squared_error(y_true, y_pred_base))
    results = []

    for feature_idx, feature in enumerate(seq_cols):
        increases = []

        for _ in range(N_PERMUTATIONS):
            X_perm = X_seq.copy()
            perm_idx = np.random.permutation(X_perm.shape[0])
            X_perm[:, :, feature_idx] = X_perm[perm_idx, :, feature_idx]

            y_pred_perm = y_scaler.inverse_transform(
                predict(model, X_perm, X_static).reshape(-1, 1)
            ).flatten()

            rmse_perm = np.sqrt(mean_squared_error(y_true, y_pred_perm))
            increases.append(rmse_perm - base_rmse)

        results.append({
            "feature": feature,
            "path": "Sequential",
            "mean_rmse_increase": float(np.mean(increases)),
            "std_rmse_increase": float(np.std(increases)),
            "base_rmse": float(base_rmse),
        })

    for feature_idx, feature in enumerate(static_cols):
        increases = []

        for _ in range(N_PERMUTATIONS):
            X_perm = X_static.copy()
            perm_idx = np.random.permutation(X_perm.shape[0])
            X_perm[:, feature_idx] = X_perm[perm_idx, feature_idx]

            y_pred_perm = y_scaler.inverse_transform(
                predict(model, X_seq, X_perm).reshape(-1, 1)
            ).flatten()

            rmse_perm = np.sqrt(mean_squared_error(y_true, y_pred_perm))
            increases.append(rmse_perm - base_rmse)

        results.append({
            "feature": feature,
            "path": "Static",
            "mean_rmse_increase": float(np.mean(increases)),
            "std_rmse_increase": float(np.std(increases)),
            "base_rmse": float(base_rmse),
        })

    return results


def summarize_importance(raw_df):
    summary = (
        raw_df.groupby(["feature", "path", "category", "week_position"], dropna=False)
        .agg(
            mean_rmse_increase=("mean_rmse_increase", "mean"),
            median_rmse_increase=("mean_rmse_increase", "median"),
            std_across_windows=("mean_rmse_increase", "std"),
            mean_perm_std=("std_rmse_increase", "mean"),
            n_windows=("mean_rmse_increase", "size"),
        )
        .reset_index()
    )

    summary["base_variable"] = summary["feature"].apply(get_base_variable)
    summary = summary.sort_values("mean_rmse_increase", ascending=False).reset_index(drop=True)

    return summary


def summarize_base_variables(summary):
    base_summary = (
        summary.groupby("base_variable", dropna=False)
        .agg(
            mean_rmse_increase=("mean_rmse_increase", "sum"),
            median_rmse_increase=("median_rmse_increase", "sum"),
            n_components=("feature", "count"),
        )
        .reset_index()
    )

    base_summary["category"] = base_summary["base_variable"].apply(categorize_feature)
    base_summary["week_position"] = base_summary["base_variable"].apply(get_week_position)
    base_summary = base_summary.sort_values("mean_rmse_increase", ascending=False).reset_index(drop=True)

    return base_summary


def summarize_categories(summary):
    temp = summary.copy()
    temp["value_used"] = temp["mean_rmse_increase"].clip(lower=0)

    return (
        temp.groupby("category", dropna=False)
        .agg(
            total_rmse_increase=("value_used", "sum"),
            avg_rmse_increase=("value_used", "mean"),
            n_features=("feature", "count"),
        )
        .reset_index()
        .sort_values("total_rmse_increase", ascending=False)
    )


def summarize_weeks(summary):
    temp = summary[summary["week_position"].notna()].copy()

    if temp.empty:
        return pd.DataFrame()

    temp["value_used"] = temp["mean_rmse_increase"].clip(lower=0)

    return (
        temp.groupby("week_position", dropna=False)
        .agg(
            total_rmse_increase=("value_used", "sum"),
            avg_rmse_increase=("value_used", "mean"),
            n_features=("feature", "count"),
        )
        .reset_index()
        .sort_values("week_position")
    )


def save_summaries(raw_df, prefix):
    summary = summarize_importance(raw_df)
    base_summary = summarize_base_variables(summary)

    feature_exogenous = summary[
        ~summary["feature"].str.startswith("C_")
        & ~summary["feature"].str.startswith("infl_yoy_lag")
    ].copy()

    base_exogenous = base_summary[
        ~base_summary["base_variable"].str.startswith("C_")
        & ~base_summary["base_variable"].str.startswith("infl_yoy_lag")
    ].copy()

    by_category = summarize_categories(summary)
    by_week = summarize_weeks(summary)

    summary.to_csv(OUT_DIR / f"{prefix}_feature_summary.csv", index=False)
    base_summary.to_csv(OUT_DIR / f"{prefix}_base_variable_summary.csv", index=False)
    feature_exogenous.to_csv(OUT_DIR / f"{prefix}_feature_summary_exogenous.csv", index=False)
    base_exogenous.to_csv(OUT_DIR / f"{prefix}_base_variable_summary_exogenous.csv", index=False)
    by_category.to_csv(OUT_DIR / f"{prefix}_by_category.csv", index=False)
    by_week.to_csv(OUT_DIR / f"{prefix}_by_week.csv", index=False)

    return summary, base_summary


def prepare_data():
    df = pd.read_csv(INPUT_PATH)
    df["date"] = pd.to_datetime(df["date"])

    for col in [c for c in df.columns if c.startswith("C_")]:
        if df[col].dtype == "object":
            df[col] = df[col].map({
                "True": 1,
                "False": 0,
                True: 1,
                False: 0,
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


def run_permutation():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df, seq_cols, static_cols, ar_lag_cols = prepare_data()
    full_seq = create_sequences(df, seq_cols, static_cols, HORIZON, SEQ_LEN)

    if not full_seq:
        raise ValueError("No sequences were created.")

    meta_dates = pd.to_datetime([row["date"] for row in full_seq])
    test_dates = sorted(set(date for date in meta_dates if date >= pd.Timestamp(TEST_START)))

    if not test_dates:
        raise ValueError("No test dates found.")

    all_rows = []
    n_ar = len(ar_lag_cols)

    for t_date in tqdm(test_dates, desc="Permutation importance", unit="date"):
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
        y_test = np.array([full_seq[j]["y"] for j in test_idx], dtype=np.float32)

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
            len(static_cols)
        )

        perm_results = permutation_importance(
            model,
            X_seq_test_scaled,
            X_static_test_scaled,
            y_test,
            seq_cols,
            static_cols,
            y_scaler
        )

        for result in perm_results:
            feature = result["feature"]

            all_rows.append({
                "date": pd.Timestamp(t_date),
                "feature": feature,
                "base_variable": get_base_variable(feature),
                "path": result["path"],
                "category": categorize_feature(feature),
                "week_position": get_week_position(feature),
                "mean_rmse_increase": result["mean_rmse_increase"],
                "std_rmse_increase": result["std_rmse_increase"],
                "base_rmse": result["base_rmse"],
                "n_test_obs": len(y_test),
                "horizon": HORIZON,
            })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    raw_df = pd.DataFrame(all_rows)

    if raw_df.empty:
        raise ValueError("Permutation output is empty.")

    raw_df["date"] = pd.to_datetime(raw_df["date"])
    raw_df.to_csv(OUT_DIR / "B_LSTM_permutation_importance_rolling_raw.csv", index=False)

    save_summaries(raw_df, "B_LSTM_perm_overall")

    return raw_df


if __name__ == "__main__":
    set_seed()
    run_permutation()
    print(f"Done. Permutation outputs saved to: {OUT_DIR}")