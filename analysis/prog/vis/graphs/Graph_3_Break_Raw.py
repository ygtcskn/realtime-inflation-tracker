import matplotlib.pyplot as plt
import pandas as pd

from config import DATA_TEMP, OUTPUT_GRAPHS

OUT_DIR = OUTPUT_GRAPHS
OUT_DIR.mkdir(parents=True, exist_ok=True)

BREAK_YEARS = [2011, 2016, 2022]

GRAPH3_PNG = OUT_DIR / "Graph_3_Break_Raw.png"
GRAPH3_PDF = OUT_DIR / "Graph_3_Break_Raw.pdf"


def apply_style(ax, yticks, ylabel_size=13, ytick_size=14, xtick_size=14):
    ax.set_yticks(yticks)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.tick_params(axis="y", labelsize=ytick_size)   # Y-axis numbers
    ax.tick_params(axis="x", labelsize=xtick_size)   # X-axis numbers
    if ax.get_ylabel():
        ax.yaxis.label.set_size(ylabel_size)         # Y-axis title


def save_figure(fig, png_path, pdf_path):
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def graph_3():
    df_raw = pd.read_csv(
        DATA_TEMP / "1_monthly_raw.csv",
        index_col=0,
        parse_dates=True
    )

    df_adj = pd.read_csv(
        DATA_TEMP / "2_monthly_breakadj.csv",
        index_col=0,
        parse_dates=True
    )

    raw_series = df_raw["US_shopping"].dropna()
    adj_series = df_adj["US_shopping"].dropna()

    fig, ax = plt.subplots(figsize=(16, 6))

    ax.plot(
        raw_series.index,
        raw_series.values,
        linewidth=2.5,
        color="#BDBDBD",
        alpha=0.6,
        label="Raw Series",
        zorder=3
    )

    ax.plot(
        adj_series.index,
        adj_series.values,
        linewidth=2.5,
        color="#4285F4",
        label="Break-Adjusted Series",
        zorder=2
    )

    for year in BREAK_YEARS:
        ax.axvline(
            pd.Timestamp(f"{year}-01-01"),
            linestyle="--",
            color="#EA4335",
            alpha=0.9
        )

    ax.set_ylabel("Search Interest")
    apply_style(ax, [20, 40, 60, 80, 100])
    ax.legend(loc="upper right", fontsize=12)

    save_figure(fig, GRAPH3_PNG, GRAPH3_PDF)


if __name__ == "__main__":
    graph_3()
    print(f"Done. Graph saved to: {OUT_DIR}")