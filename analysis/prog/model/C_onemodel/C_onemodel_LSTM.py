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

INPUT_PATH = DATA_FINAL / "C_panel.csv"
OUT_DIR = DATA_OUTPUT / "C_onemodel_LSTM"

TEST_START = "2020-01"
HORIZONS = [0]
MIN_TRAIN = 60

HIDDEN_SIZE = 128
NUM_LAYERS = 2
FC_HIDDEN = 128
SEQ_LEN = 14
DROPOUT = 0.3021576121968087
LR = 0.0005741998556902038
BATCH_SIZE = 32
EPOCHS = 100
WEIGHT_DECAY = 0.00021199372995063797
GRAD_CLIP = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42
MODEL_NAME = "C_onemodel_LSTM"

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


def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_sequences(df, seq_cols, static_cols, h, seq_len):
    data = []
    df = df.sort_values(["country", "week_position", "date"]).copy()
    df["target"] = df.groupby(["country", "week_position"])["infl_yoy"].shift(-h)
    df = df.dropna(subset=["target"])

    for (country, wp), grp in df.groupby(["country", "week_position"]):
        grp = grp.sort_values("date").reset_index(drop=True)

        if len(grp) <= seq_len:
            continue

        v_seq = grp[seq_cols].values.astype(np.float32)
        v_static = grp[static_cols].values.astype(np.float32)
        v_y = grp["target"].values.astype(np.float32)
        dates = grp["date"].values

        for i in range(len(grp) - seq_len):
            end_idx = i + seq_len - 1

            data.append({
                "date": dates[end_idx],
                "country": country,
                "week_position": int(wp),
                "X_seq": v_seq[i:i + seq_len].copy(),
                "X_static": v_static[end_idx].copy(),
                "y": float(v_y[end_idx])
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

        scheduler.step(epoch_loss / batches)

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
        wp = key[1]
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
                "week_position": wp,
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

    def add_metrics(scope, country, data, week_position=np.nan):
        if len(data) == 0:
            return

        metrics = compute_metrics(data["actual"].values, data["predicted"].values)

        row = {
            "scope": scope,
            "country": country,
            **metrics,
            "model": model_name
        }

        if not pd.isna(week_position):
            row["week_position"] = week_position

        metrics_rows.append(row)

    add_metrics("overall", "ALL", test_df)

    for c in countries:
        add_metrics("country", c, test_df[test_df["country"] == c])

    if "week_position" in test_df.columns:
        for wp in sorted(test_df["week_position"].unique()):
            wp_data = test_df[test_df["week_position"] == wp]
            add_metrics("week", "ALL", wp_data, week_position=wp)

            for c in countries:
                add_metrics(
                    "week_country",
                    c,
                    wp_data[wp_data["country"] == c],
                    week_position=wp
                )

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

    if "week_position" in test_df.columns:
        for wp in sorted(test_df["week_position"].unique()):
            wp_errors = test_df[test_df["week_position"] == wp].pivot_table(
                index="date",
                columns="country",
                values="error",
                aggfunc="first"
            )

            wp_errors.to_csv(out_dir / f"{prefix}_errors_pivot_W{int(wp)}.csv")

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

    exclude = ["date", "country", "week_position", "infl_yoy"]
    ar_lag_cols = sorted([c for c in df.columns if c.startswith("infl_yoy_lag")])

    seq_cols = [
        c for c in df.columns
        if c not in exclude
        and not c.startswith("C_")
        and c not in ar_lag_cols
    ]

    dummy_cols = [c for c in df.columns if c.startswith("C_")]

    static_cols = ar_lag_cols + ["week_position"] + dummy_cols
    n_ar = len(ar_lag_cols)

    for col in seq_cols + ar_lag_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[seq_cols] = df[seq_cols].fillna(0).astype(np.float32)
    df[ar_lag_cols] = df[ar_lag_cols].fillna(0).astype(np.float32)
    df["week_position"] = df["week_position"].astype(np.float32)
    df[dummy_cols] = df[dummy_cols].fillna(0).astype(np.float32)

    all_results = []
    train_fits = defaultdict(dict)
    df_actual = df[["date", "country", "infl_yoy"]].drop_duplicates().copy()

    for h in HORIZONS:
        full_seq = create_sequences(df, seq_cols, static_cols, h, SEQ_LEN)

        if not full_seq:
            continue

        meta_dates = pd.to_datetime([d["date"] for d in full_seq])
        test_dates = sorted(set(d for d in meta_dates if d >= pd.Timestamp(TEST_START)))

        for t_date in tqdm(test_dates, desc=f"C_LSTM h={h}", unit="date"):
            tr_idx = [
                j for j, d in enumerate(full_seq)
                if pd.Timestamp(d["date"]) < t_date
            ]

            te_idx = [
                j for j, d in enumerate(full_seq)
                if pd.Timestamp(d["date"]) == t_date
            ]

            if len(tr_idx) < MIN_TRAIN or not te_idx:
                continue

            Xsq_tr = np.stack([full_seq[j]["X_seq"] for j in tr_idx]).astype(np.float32)
            Xst_tr = np.stack([full_seq[j]["X_static"] for j in tr_idx]).astype(np.float32)
            y_tr = np.array([full_seq[j]["y"] for j in tr_idx], dtype=np.float32)

            m_tr = [
                (full_seq[j]["date"], full_seq[j]["country"], full_seq[j]["week_position"])
                for j in tr_idx
            ]

            Xsq_te = np.stack([full_seq[j]["X_seq"] for j in te_idx]).astype(np.float32)
            Xst_te = np.stack([full_seq[j]["X_static"] for j in te_idx]).astype(np.float32)
            y_te = np.array([full_seq[j]["y"] for j in te_idx], dtype=np.float32)

            m_te = [
                (full_seq[j]["date"], full_seq[j]["country"], full_seq[j]["week_position"])
                for j in te_idx
            ]

            sy = StandardScaler()
            y_tr_s = sy.fit_transform(y_tr.reshape(-1, 1)).flatten().astype(np.float32)

            sx = StandardScaler()
            n_obs, seq_len, n_features = Xsq_tr.shape

            Xsq_tr_s = sx.fit_transform(
                Xsq_tr.reshape(-1, n_features)
            ).reshape(n_obs, seq_len, n_features).astype(np.float32)

            Xsq_te_s = sx.transform(
                Xsq_te.reshape(-1, n_features)
            ).reshape(Xsq_te.shape[0], seq_len, n_features).astype(np.float32)

            Xst_tr_s, Xst_te_s = scale_static(Xst_tr, Xst_te, n_ar)

            model = train_model(
                Xsq_tr_s,
                Xst_tr_s,
                y_tr_s,
                n_features,
                len(static_cols)
            )

            yp_te = sy.inverse_transform(
                predict(model, Xsq_te_s, Xst_te_s).reshape(-1, 1)
            ).flatten()

            yp_tr = sy.inverse_transform(
                predict(model, Xsq_tr_s, Xst_tr_s).reshape(-1, 1)
            ).flatten()

            for j in range(len(te_idx)):
                all_results.append({
                    "date": m_te[j][0],
                    "country": m_te[j][1],
                    "week_position": m_te[j][2],
                    "horizon": h,
                    "actual": y_te[j],
                    "predicted": yp_te[j]
                })

            for j in range(len(tr_idx)):
                train_fits[(m_tr[j][1], m_tr[j][2], h)][m_tr[j][0]] = yp_tr[j]

    res_df = pd.DataFrame(all_results)
    res_df["date"] = pd.to_datetime(res_df["date"])

    save_results(res_df, train_fits, df_actual, MODEL_NAME, OUT_DIR, COUNTRIES)

    print(f"Done. Results saved to: {OUT_DIR}")