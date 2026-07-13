import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from config import DATA_FINAL, DATA_OUTPUT

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

warnings.filterwarnings("ignore")

INPUT_PATH = DATA_FINAL / "B_panel.csv"
OUT_DIR = DATA_OUTPUT / "B_LSTM_couspe"

TEST_START = "2020-01"
HORIZONS = [0]
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

COUNTRIES = ["BR", "CA", "FR", "DE", "IN", "ID", "IT", "JP", "MX", "RU", "ZA", "KR", "TR", "GB", "US"]

SUBGROUPS = {
    "G7": ["CA", "FR", "DE", "IT", "JP", "GB", "US"],
    "BRICS": ["BR", "RU", "IN", "ZA"],
    "Emerging": ["BR", "IN", "ID", "MX", "RU", "ZA", "KR", "TR"],
    "Advanced": ["CA", "FR", "DE", "IT", "JP", "GB", "US"],
    "Europe": ["FR", "DE", "IT", "GB"],
    "Asia": ["IN", "ID", "JP", "KR"],
    "Americas": ["BR", "CA", "MX", "US"],
}

RUN_COUNTRY_SPECIFIC = True
RUN_SUBGROUP_POOLED = True


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
    if n_continuous == 0:
        return X_static_train.astype(np.float32), X_static_test.astype(np.float32)

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


def predict(model, X_seq, X_static):
    model.eval()

    with torch.no_grad():
        preds = model(
            torch.tensor(X_seq, dtype=torch.float32).to(DEVICE),
            torch.tensor(X_static, dtype=torch.float32).to(DEVICE)
        )

    return preds.cpu().numpy()


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


def run_expanding_window(df, seq_cols, static_cols, n_ar, horizons):
    all_results = []

    for h in horizons:
        full_seq = create_sequences(df, seq_cols, static_cols, h, SEQ_LEN)

        if not full_seq:
            continue

        meta_dates = pd.to_datetime([row["date"] for row in full_seq])
        test_dates = sorted(set(date for date in meta_dates if date >= pd.Timestamp(TEST_START)))

        for t_date in tqdm(test_dates, desc="Expanding window", unit="date", leave=False):
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

            test_meta = [
                (full_seq[j]["date"], full_seq[j]["country"])
                for j in test_idx
            ]

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

            y_pred = y_scaler.inverse_transform(
                predict(model, X_seq_test_scaled, X_static_test_scaled).reshape(-1, 1)
            ).flatten()

            for j in range(len(test_idx)):
                all_results.append({
                    "date": test_meta[j][0],
                    "country": test_meta[j][1],
                    "horizon": h,
                    "actual": float(y_test[j]),
                    "predicted": float(y_pred[j])
                })

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return all_results


def save_results(test_df, model_name, out_dir, countries):
    test_df = test_df.copy()
    test_df["model"] = model_name
    test_df["set"] = "test"
    test_df["error"] = test_df["actual"] - test_df["predicted"]
    test_df["abs_error"] = test_df["error"].abs()
    test_df["sq_error"] = test_df["error"] ** 2

    test_df.to_csv(out_dir / f"{model_name}_test_predictions.csv", index=False)

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

    for country in countries:
        add_metrics("country", country, test_df[test_df["country"] == country])

    add_metrics("turkey_analysis", "WITH_TR", test_df)
    add_metrics("turkey_analysis", "WITHOUT_TR", test_df[test_df["country"] != "TR"])

    country_rmses = []
    country_rmses_no_tr = []

    for country in countries:
        country_data = test_df[test_df["country"] == country]

        if len(country_data) > 0:
            rmse = np.sqrt(mean_squared_error(country_data["actual"], country_data["predicted"]))
            country_rmses.append(rmse)

            if country != "TR":
                country_rmses_no_tr.append(rmse)

    metrics_rows.append({
        "scope": "turkey_analysis",
        "country": "AVG_WITH_TR",
        "RMSE": np.mean(country_rmses) if country_rmses else np.nan,
        "MAE": np.nan,
        "R2": np.nan,
        "MAPE": np.nan,
        "N": len(country_rmses),
        "model": model_name
    })

    metrics_rows.append({
        "scope": "turkey_analysis",
        "country": "AVG_WITHOUT_TR",
        "RMSE": np.mean(country_rmses_no_tr) if country_rmses_no_tr else np.nan,
        "MAE": np.nan,
        "R2": np.nan,
        "MAPE": np.nan,
        "N": len(country_rmses_no_tr),
        "model": model_name
    })

    add_metrics("test", "ALL", test_df)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_dir / f"{model_name}_all_metrics.csv", index=False)

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

    errors_pivot.to_csv(out_dir / f"{model_name}_errors_pivot.csv")
    sq_errors_pivot.to_csv(out_dir / f"{model_name}_sq_errors_pivot.csv")

    return metrics_df


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

    for col in seq_cols + ar_lag_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[seq_cols] = df[seq_cols].fillna(0).astype(np.float32)
    df[ar_lag_cols] = df[ar_lag_cols].fillna(0).astype(np.float32)
    df[dummy_cols] = df[dummy_cols].fillna(0).astype(np.float32)

    return df, seq_cols, ar_lag_cols, dummy_cols


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed()

    df, seq_cols, ar_lag_cols, dummy_cols = prepare_data()
    n_ar = len(ar_lag_cols)

    if RUN_COUNTRY_SPECIFIC:
        country_out = OUT_DIR / "country_specific"
        country_out.mkdir(parents=True, exist_ok=True)

        all_country_results = []

        for country in tqdm(COUNTRIES, desc="Country-specific", unit="country"):
            df_country = df[df["country"] == country].copy()
            results = run_expanding_window(
                df_country,
                seq_cols,
                ar_lag_cols,
                n_ar,
                HORIZONS
            )
            all_country_results.extend(results)

        if all_country_results:
            country_df = pd.DataFrame(all_country_results)
            country_df["date"] = pd.to_datetime(country_df["date"])
            save_results(country_df, "B_LSTM_couspe_country_specific", country_out, COUNTRIES)

    if RUN_SUBGROUP_POOLED:
        subgroup_out = OUT_DIR / "subgroup_pooled"
        subgroup_out.mkdir(parents=True, exist_ok=True)

        for subgroup_name, subgroup_countries in tqdm(SUBGROUPS.items(), desc="Subgroup-pooled", unit="group"):
            df_subgroup = df[df["country"].isin(subgroup_countries)].copy()

            subgroup_dummy_cols = [
                f"C_{country}"
                for country in subgroup_countries
                if f"C_{country}" in dummy_cols
            ]

            static_cols = ar_lag_cols + subgroup_dummy_cols

            results = run_expanding_window(
                df_subgroup,
                seq_cols,
                static_cols,
                n_ar,
                HORIZONS
            )

            if results:
                result_df = pd.DataFrame(results)
                result_df["date"] = pd.to_datetime(result_df["date"])

                subgroup_dir = subgroup_out / subgroup_name
                subgroup_dir.mkdir(parents=True, exist_ok=True)

                save_results(
                    result_df,
                    f"B_LSTM_couspe_{subgroup_name}",
                    subgroup_dir,
                    subgroup_countries
                )

    print(f"Done. Results saved to: {OUT_DIR}")