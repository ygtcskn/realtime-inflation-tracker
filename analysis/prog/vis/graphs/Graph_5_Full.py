import warnings

import matplotlib
matplotlib.use("Agg")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from config import DATA_OUTPUT, OUTPUT_GRAPHS

warnings.filterwarnings("ignore")

BASE_DIR = DATA_OUTPUT / "B_LSTM"
TEST_PATH = BASE_DIR / "B_LSTM_test_predictions.csv"
TRAIN_PATH = BASE_DIR / "B_LSTM_train_predictions.csv"

GRAPH5_PNG = OUTPUT_GRAPHS / "Graph_5_Full.png"
GRAPH5_PDF = OUTPUT_GRAPHS / "Graph_5_Full.pdf"

HORIZON = 0
TEST_START = pd.Timestamp("2020-01-01")

COUNTRIES_TO_PLOT = ["US", "DE", "FR", "GB", "IT", "JP", "CA", "KR", "TR", "BR"]

BG_COLOR = "#FFFFFF"
NEUTRAL_FIG = "#FFFFFF"
NEUTRAL_EDGE = "#D0D0D0"
NEUTRAL_TEXT = "#222222"
ACTUAL_COLOR = "#000000"
PREDICTED_COLOR = "#2ca02c"

plt.rcParams.update({
    "figure.facecolor": NEUTRAL_FIG,
    "axes.facecolor": BG_COLOR,
    "axes.edgecolor": NEUTRAL_EDGE,
    "axes.labelcolor": NEUTRAL_TEXT,
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "text.color": NEUTRAL_TEXT,
    "grid.color": "#E6E6E6",
    "grid.alpha": 0.75,
    "grid.linewidth": 0.9,
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#D0D0D0",
    "legend.facecolor": "#FFFFFF",
})


def save_figure(fig, png_path, pdf_path):
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02, facecolor=fig.get_facecolor())
    plt.close(fig)


def style_axis(ax, title):
    ax.set_title(title, fontsize=11, pad=6, color=NEUTRAL_TEXT)
    ax.grid(False)
    ax.set_facecolor(BG_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelrotation=45, labelsize=8, width=1.0, length=4)
    ax.tick_params(axis="y", labelsize=8, width=1.0, length=4)


def add_shared_legend(fig):
    handles = [
        Line2D([0], [0], color=ACTUAL_COLOR, lw=1.2, label="Actual"),
        Line2D([0], [0], color=PREDICTED_COLOR, lw=1.2, label="Predicted"),
        Line2D([0], [0], color="red", lw=1.0, linestyle="--", label="Test start"),
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.005)
    )


def plot_graph_5_full_panel(test_df, train_df):
    fig, axes = plt.subplots(5, 2, figsize=(8.27, 9.27), sharex=True, sharey=False, facecolor=NEUTRAL_FIG)
    axes = axes.flatten()

    for i, (ax, country) in enumerate(zip(axes, COUNTRIES_TO_PLOT)):
        test_c = test_df[(test_df["country"] == country) & (test_df["horizon"] == HORIZON)].copy().sort_values("date")
        train_c = train_df[(train_df["country"] == country) & (train_df["horizon"] == HORIZON)].copy().sort_values("date")

        actual_c = pd.concat(
            [train_c[["date", "actual"]], test_c[["date", "actual"]]],
            axis=0
        ).drop_duplicates(subset=["date"]).sort_values("date")

        if not actual_c.empty:
            ax.plot(actual_c["date"], actual_c["actual"], color=ACTUAL_COLOR, linewidth=1.2)

        if not test_c.empty:
            ax.plot(test_c["date"], test_c["predicted"], color=PREDICTED_COLOR, linewidth=1.2)

        ax.axvline(TEST_START, color="red", linestyle="--", linewidth=1.0, alpha=0.9)
        style_axis(ax, country)

        if i < 8:
            ax.tick_params(axis="x", labelbottom=False)

    fig.supylabel("Inflation YoY (%)", fontsize=8, x=0.01)
    add_shared_legend(fig)
    plt.tight_layout(rect=[-0.02, 0.05, 0.98, 0.98], h_pad=0.8, w_pad=0.8)

    save_figure(fig, GRAPH5_PNG, GRAPH5_PDF)


def plot_graph_5_full_country(test_df, train_df, country):
    test_c = test_df[(test_df["country"] == country) & (test_df["horizon"] == HORIZON)].copy().sort_values("date")
    train_c = train_df[(train_df["country"] == country) & (train_df["horizon"] == HORIZON)].copy().sort_values("date")

    if test_c.empty and train_c.empty:
        return

    actual_c = pd.concat(
        [train_c[["date", "actual"]], test_c[["date", "actual"]]],
        axis=0
    ).drop_duplicates(subset=["date"]).sort_values("date")

    fig, ax = plt.subplots(figsize=(8.27, 4.2), facecolor=NEUTRAL_FIG)

    if not actual_c.empty:
        ax.plot(actual_c["date"], actual_c["actual"], color=ACTUAL_COLOR, linewidth=1.3, label="Actual")

    if not test_c.empty:
        ax.plot(test_c["date"], test_c["predicted"], color=PREDICTED_COLOR, linewidth=1.3, label="Predicted")

    ax.axvline(TEST_START, color="red", linestyle="--", linewidth=1.0, alpha=0.9, label="Test start")
    style_axis(ax, country)
    ax.set_ylabel("Inflation YoY (%)")
    ax.set_xlabel("Date")
    ax.legend(loc="best", frameon=True, fontsize=8)

    save_figure(
        fig,
        OUTPUT_GRAPHS / f"Graph_5_Full_{country}.png",
        OUTPUT_GRAPHS / f"Graph_5_Full_{country}.pdf"
    )


def plot_graph_5_full():
    if not TEST_PATH.exists() or not TRAIN_PATH.exists():
        return

    test_df = pd.read_csv(TEST_PATH)
    train_df = pd.read_csv(TRAIN_PATH)

    test_df["date"] = pd.to_datetime(test_df["date"])
    train_df["date"] = pd.to_datetime(train_df["date"])

    plot_graph_5_full_panel(test_df, train_df)

    for country in tqdm(COUNTRIES_TO_PLOT, desc="Graph 5 (full) per country", unit="country"):
        plot_graph_5_full_country(test_df, train_df, country)


if __name__ == "__main__":
    OUTPUT_GRAPHS.mkdir(parents=True, exist_ok=True)
    plot_graph_5_full()
    print("Done. Graph saved.")
