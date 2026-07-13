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

GRAPH52_PNG = OUTPUT_GRAPHS / "Graph_5_2_Test_Only.png"
GRAPH52_PDF = OUTPUT_GRAPHS / "Graph_5_2_Test_Only.pdf"

HORIZON = 0

# Main 4x1 grid
COUNTRIES_TO_PLOT = ["US", "DE", "FR", "KR"]

# Separate country graphs for all countries
ALL_COUNTRIES = [
    "BR", "CA", "FR", "DE", "IN", "ID", "IT", "JP",
    "MX", "RU", "ZA", "KR", "GB", "US", "TR"
]

BG_COLOR = "#FFFFFF"
NEUTRAL_FIG = "#FFFFFF"
NEUTRAL_EDGE = "#D0D0D0"
NEUTRAL_TEXT = "#222222"
ACTUAL_COLOR = "#000000"

# Google-style palette, extended for all countries
COUNTRY_COLORS = {
    "US": "#4285F4",
    "DE": "#DB4437",
    "FR": "#F4B400",
    "KR": "#0F9D58",

    "BR": "#7BAAF7",
    "CA": "#E67C73",
    "IN": "#F7CB4D",
    "ID": "#57BB8A",
    "IT": "#3367D6",
    "JP": "#C5221F",
    "MX": "#F29900",
    "RU": "#188038",
    "ZA": "#AECBFA",
    "GB": "#F28B82",
    "TR": "#FBBC04",
}

COUNTRY_NAMES = {
    "BR": "Brazil",
    "CA": "Canada",
    "FR": "France",
    "DE": "Germany",
    "IN": "India",
    "ID": "Indonesia",
    "IT": "Italy",
    "JP": "Japan",
    "MX": "Mexico",
    "RU": "Russia",
    "ZA": "South Africa",
    "KR": "South Korea",
    "GB": "United Kingdom",
    "US": "United States",
    "TR": "Turkey",
}

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
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#D0D0D0",
    "legend.facecolor": "#FFFFFF",
})


def save_figure(fig, png_path, pdf_path):
    png_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor=fig.get_facecolor()
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor=fig.get_facecolor()
    )

    plt.close(fig)


def style_axis(ax, title):
    ax.set_title(title, fontsize=11, pad=6, color=NEUTRAL_TEXT)
    ax.grid(False)
    ax.set_facecolor(BG_COLOR)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    # show years only on x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax.tick_params(axis="x", labelrotation=45, labelsize=8, width=1.0, length=4)
    ax.tick_params(axis="y", labelsize=8, width=1.0, length=4)


def add_shared_legend(fig):
    handles = [Line2D([0], [0], color=ACTUAL_COLOR, lw=1.2, label="Actual")]

    for country in COUNTRIES_TO_PLOT:
        handles.append(
            Line2D(
                [0], [0],
                color=COUNTRY_COLORS.get(country, "#4285F4"),
                lw=1.2,
                label=country
            )
        )

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.01)
    )


def plot_graph_5_test_only_panel(test_df):
    fig, axes = plt.subplots(
        4, 1,
        figsize=(8.27, 10.5),
        sharex=True,
        sharey=False,
        facecolor=NEUTRAL_FIG
    )

    if len(COUNTRIES_TO_PLOT) == 1:
        axes = [axes]

    for i, (ax, country) in enumerate(zip(axes, COUNTRIES_TO_PLOT)):
        test_c = (
            test_df[
                (test_df["country"] == country) &
                (test_df["horizon"] == HORIZON)
            ]
            .copy()
            .sort_values("date")
        )

        country_color = COUNTRY_COLORS.get(country, "#4285F4")

        if not test_c.empty:
            ax.plot(
                test_c["date"],
                test_c["actual"],
                color=ACTUAL_COLOR,
                linewidth=1.2
            )

            ax.plot(
                test_c["date"],
                test_c["predicted"],
                color=country_color,
                linewidth=1.2
            )

        style_axis(ax, COUNTRY_NAMES.get(country, country))

        if i < len(COUNTRIES_TO_PLOT) - 1:
            ax.tick_params(axis="x", labelbottom=False)

    fig.supylabel("Inflation YoY (%)", fontsize=11, x=0.01)
    add_shared_legend(fig)

    plt.tight_layout(rect=[-0.02, 0.05, 0.98, 0.98], h_pad=1.0)

    save_figure(fig, GRAPH52_PNG, GRAPH52_PDF)


def plot_graph_5_test_only_country(test_df, country):
    test_c = (
        test_df[
            (test_df["country"] == country) &
            (test_df["horizon"] == HORIZON)
        ]
        .copy()
        .sort_values("date")
    )

    if test_c.empty:
        print(f"Skipping {country}: no data.")
        return

    country_color = COUNTRY_COLORS.get(country, "#4285F4")
    country_name = COUNTRY_NAMES.get(country, country)

    # same size as graph_10_realtracking.py standalone graphs
    fig, ax = plt.subplots(figsize=(16, 5), facecolor=NEUTRAL_FIG)

    ax.plot(
        test_c["date"],
        test_c["actual"],
        color=ACTUAL_COLOR,
        linewidth=1.3,
        label="Actual"
    )

    ax.plot(
        test_c["date"],
        test_c["predicted"],
        color=country_color,
        linewidth=1.3,
        label="Predicted"
    )

    style_axis(ax, country_name)
    ax.set_ylabel("Inflation YoY (%)", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)

    ax.legend(
        loc="best",
        frameon=True,
        fontsize=10
    )

    save_figure(
        fig,
        OUTPUT_GRAPHS / f"Graph_5_2_Test_Only_{country}.png",
        OUTPUT_GRAPHS / f"Graph_5_2_Test_Only_{country}.pdf"
    )


def plot_graph_5_test_only():
    if not TEST_PATH.exists():
        print(f"File not found: {TEST_PATH}")
        return

    test_df = pd.read_csv(TEST_PATH)
    test_df["date"] = pd.to_datetime(test_df["date"])

    # Main 4x1 panel
    plot_graph_5_test_only_panel(test_df)

    # Separate graphs for all countries
    for country in tqdm(ALL_COUNTRIES, desc="Graph 5 per country", unit="country"):
        plot_graph_5_test_only_country(test_df, country)


if __name__ == "__main__":
    OUTPUT_GRAPHS.mkdir(parents=True, exist_ok=True)
    plot_graph_5_test_only()
    print("Done. Graphs saved.")