import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA_OUTPUT, OUTPUT_GRAPHS

warnings.filterwarnings("ignore")

OUT_DIR = OUTPUT_GRAPHS
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_DIR = DATA_OUTPUT / "C_onemodel_LSTM"
D_DIR = DATA_OUTPUT / "D_weekspecific_LSTM"

C_PRED_PATH = C_DIR / "C_onemodel_LSTM_test_predictions.csv"
D_PRED_PATH = D_DIR / "D_weekspecific_LSTM_test_predictions.csv"

GRAPH9_PNG = OUT_DIR / "Graph_9_Nowcast_CD.png"
GRAPH9_PDF = OUT_DIR / "Graph_9_Nowcast_CD.pdf"

ALL_COUNTRIES = ["BR", "CA", "FR", "DE", "IN", "ID", "IT", "JP", "MX", "RU", "ZA", "KR", "GB", "US"]
NOWCAST_COUNTRIES = ["US", "DE", "JP", "KR"]
WEEK_POSITIONS = [1, 2, 3, 4]

WEEK_LABELS = {
    1: "Week 1",
    2: "Week 2",
    3: "Week 3",
    4: "Week 4",
}

WEEK_COLORS = {
    1: "#4285F4",
    2: "#DB4437",
    3: "#F4B400",
    4: "#0F9D58",
}

COUNTRY_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896"
]

COUNTRY_COLOR_MAP = {
    country: COUNTRY_COLORS[i]
    for i, country in enumerate(ALL_COUNTRIES)
}

BG_COLOR = "#F0F0F0"
NEUTRAL_FIG = "#FFFFFF"
NEUTRAL_EDGE = "#D0D0D0"
NEUTRAL_TEXT = "#222222"

plt.rcParams.update({
    "figure.facecolor": NEUTRAL_FIG,
    "axes.edgecolor": NEUTRAL_EDGE,
    "axes.labelcolor": NEUTRAL_TEXT,
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "text.color": NEUTRAL_TEXT,
    "grid.color": "#FFFFFF",
    "grid.alpha": 0.75,
    "grid.linewidth": 0.9,
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "legend.fontsize": 10,
    "legend.framealpha": 0.95,
    "legend.edgecolor": NEUTRAL_EDGE,
    "legend.facecolor": NEUTRAL_FIG,
})


def save_figure(fig, png_path, pdf_path):
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def load_prediction_data():
    if not C_PRED_PATH.exists():
        raise FileNotFoundError(f"File not found: {C_PRED_PATH}")

    if not D_PRED_PATH.exists():
        raise FileNotFoundError(f"File not found: {D_PRED_PATH}")

    c_df = pd.read_csv(C_PRED_PATH)
    d_df = pd.read_csv(D_PRED_PATH)

    c_df["date"] = pd.to_datetime(c_df["date"])
    d_df["date"] = pd.to_datetime(d_df["date"])

    c_df["model"] = "C_onemodel_LSTM"
    d_df["model"] = "D_weekspecific_LSTM"

    df = pd.concat([c_df, d_df], ignore_index=True)

    required_cols = {"date", "country", "actual", "predicted", "week_position"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in prediction files: {missing}")

    df["week_position"] = pd.to_numeric(df["week_position"], errors="coerce")
    df = df.dropna(subset=["week_position"]).copy()
    df["week_position"] = df["week_position"].astype(int)

    return df


def plot_graph9_nowcast_revision_grid(df, countries=NOWCAST_COUNTRIES):
    fig, axes = plt.subplots(
        len(countries),
        2,
        figsize=(16, 5 * len(countries)),
        sharex=True,
        sharey=True,
        facecolor=NEUTRAL_FIG
    )

    if len(countries) == 1:
        axes = axes.reshape(1, -1)

    legend_items = {}
    df_sub = df[df["country"].isin(countries)].copy()

    y_min = min(df_sub["actual"].min(), df_sub["predicted"].min())
    y_max = max(df_sub["actual"].max(), df_sub["predicted"].max())
    pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
    y_lim = (y_min - pad, y_max + pad)

    models = ["C_onemodel_LSTM", "D_weekspecific_LSTM"]

    for r, country in enumerate(countries):
        for c, model in enumerate(models):
            ax = axes[r, c]
            ax.set_facecolor(BG_COLOR)

            sub = df[
                (df["country"] == country)
                & (df["model"] == model)
            ].copy().sort_values(["date", "week_position"])

            if sub.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{country} — {model}")
                continue

            actual_ts = (
                sub[["date", "actual"]]
                .drop_duplicates()
                .sort_values("date")
            )

            line_actual, = ax.plot(
                actual_ts["date"],
                actual_ts["actual"],
                color="#333333",
                lw=2.2,
                zorder=10,
                label="Actual"
            )

            legend_items["Actual"] = line_actual

            for week in WEEK_POSITIONS:
                week_df = sub[sub["week_position"] == week].sort_values("date")

                if not week_df.empty:
                    line_week, = ax.plot(
                        week_df["date"],
                        week_df["predicted"],
                        color=WEEK_COLORS[week],
                        lw=1.7,
                        alpha=0.95,
                        label=WEEK_LABELS[week],
                        zorder=5
                    )

                    legend_items[WEEK_LABELS[week]] = line_week

            title = "One-Model-Fits-All" if model == "C_onemodel_LSTM" else "Week-Specific"

            ax.set_title(f"{country} — {title}", color=NEUTRAL_TEXT, fontweight="bold")
            ax.set_ylim(y_lim)
            ax.grid(True, alpha=0.4)
            ax.xaxis.set_major_locator(mdates.YearLocator(1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
            ax.set_ylabel("Inflation YoY (%)" if c == 0 else "")
            ax.set_xlabel("Date" if r == len(countries) - 1 else "")
            ax.tick_params(axis="both", labelsize=10, colors="#555555")

    fig.legend(
        list(legend_items.values()),
        list(legend_items.keys()),
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        frameon=True
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    save_figure(fig, GRAPH9_PNG, GRAPH9_PDF)


if __name__ == "__main__":
    plot_graph9_nowcast_revision_grid(load_prediction_data())
    print(f"Done. Graph saved to: {OUT_DIR}")
