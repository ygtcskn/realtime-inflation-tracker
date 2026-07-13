import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from config import DATA_OUTPUT, OUTPUT_GRAPHS

warnings.filterwarnings("ignore")

OUT_DIR = OUTPUT_GRAPHS
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_DIR = DATA_OUTPUT / "C_onemodel_LSTM"
D_DIR = DATA_OUTPUT / "D_weekspecific_LSTM"

C_METRICS_PATH = C_DIR / "C_onemodel_LSTM_all_metrics.csv"
D_METRICS_PATH = D_DIR / "D_weekspecific_LSTM_all_metrics.csv"

GRAPH8_PNG = OUT_DIR / "Graph_8_CD_rmse_pct_change.png"
GRAPH8_PDF = OUT_DIR / "Graph_8_CD_rmse_pct_change.pdf"

ALL_COUNTRIES = ["BR", "CA", "FR", "DE", "IN", "ID", "IT", "JP", "MX", "RU", "ZA", "KR", "GB", "US"]
WEEK_POSITIONS = [1, 2, 3, 4]

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


def load_weekly_country_rmse(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    week_country = df[df["scope"] == "week_country"].copy()

    required_cols = {"country", "week_position", "RMSE"}
    missing = required_cols - set(week_country.columns)

    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    rmse_dict = {}

    for _, row in week_country.iterrows():
        country = row["country"]
        week = int(row["week_position"])

        if country in ALL_COUNTRIES and week in WEEK_POSITIONS:
            rmse_dict[(country, week)] = row["RMSE"]

    return rmse_dict


def build_rmse_matrix(rmse_dict):
    matrix = np.full((len(ALL_COUNTRIES), len(WEEK_POSITIONS)), np.nan)

    for i, country in enumerate(ALL_COUNTRIES):
        for j, week in enumerate(WEEK_POSITIONS):
            matrix[i, j] = rmse_dict.get((country, week), np.nan)

    return matrix


def plot_graph8_rmse_pct_change():
    mat_c = build_rmse_matrix(load_weekly_country_rmse(C_METRICS_PATH))
    mat_d = build_rmse_matrix(load_weekly_country_rmse(D_METRICS_PATH))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 6),
        sharey=False,
        facecolor=NEUTRAL_FIG
    )

    x = np.array(WEEK_POSITIONS)

    panels = [
        (axes[0], mat_c, "One-Model-Fits-All"),
        (axes[1], mat_d, "Week-Specific"),
    ]

    for ax, matrix, title in panels:
        ax.set_facecolor(BG_COLOR)
        baseline = matrix[:, 0]

        for i, country in enumerate(ALL_COUNTRIES):
            if np.isnan(baseline[i]) or baseline[i] == 0:
                pct_change = np.full(len(WEEK_POSITIONS), np.nan)
            else:
                pct_change = (matrix[i] / baseline[i] - 1.0) * 100.0

            ax.plot(
                x,
                pct_change,
                color=COUNTRY_COLOR_MAP[country],
                marker="o",
                lw=1.7,
                ms=4.5,
                label=country,
                zorder=3
            )

        ax.axhline(0, color="#333333", lw=1.6, ls="--", alpha=0.85, zorder=4)

        ylo, yhi = ax.get_ylim()
        pad = (yhi - ylo) * 0.15 if np.isfinite(yhi - ylo) else 1.0

        ax.set_ylim(ylo - pad, yhi + pad)
        ax.axhspan(0, yhi + pad, color="#e74c3c", alpha=0.04, zorder=0)
        ax.axhspan(ylo - pad, 0, color="#2ecc71", alpha=0.04, zorder=0)

        for week in WEEK_POSITIONS:
            if week % 2 == 0:
                ax.axvspan(week - 0.5, week + 0.5, color="grey", alpha=0.05, zorder=0)

        ax.set_xticks(WEEK_POSITIONS)
        ax.set_xticklabels([f"Week {week}" for week in WEEK_POSITIONS])
        ax.set_title(title, color=NEUTRAL_TEXT, fontweight="bold")
        ax.tick_params(axis="both", labelsize=10, colors="#555555")
        ax.grid(axis="y", alpha=0.4)
        ax.grid(axis="x", alpha=0.25)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.1f%%"))
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.supylabel("% change in RMSE (vs Week 1)", fontsize=9, color=NEUTRAL_TEXT)
    plt.tight_layout(rect=[0.03, 0.03, 1, 1])

    save_figure(fig, GRAPH8_PNG, GRAPH8_PDF)


if __name__ == "__main__":
    plot_graph8_rmse_pct_change()
    print(f"Done. Graph saved to: {OUT_DIR}")
