import matplotlib.pyplot as plt
import pandas as pd

from config import DATA_TEMP, OUTPUT_GRAPHS

OUT_DIR = OUTPUT_GRAPHS
OUT_DIR.mkdir(parents=True, exist_ok=True)

GRAPH2_PNG = OUT_DIR / "Graph_2_Splice.png"
GRAPH2_PDF = OUT_DIR / "Graph_2_Splice.pdf"


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


def graph_2():
    df_monthly = pd.read_csv(
        DATA_TEMP / "1_monthly_raw.csv",
        index_col=0,
        parse_dates=True
    )

    df_spliced = pd.read_csv(
        DATA_TEMP / "1_weekly_spliced_raw.csv",
        index_col=0,
        parse_dates=True
    )

    monthly_series = df_monthly["US_shopping"].dropna()
    spliced_series = df_spliced["US_shopping"].dropna()

    fig, ax = plt.subplots(figsize=(16, 6))

    overlaps = [
        ("2008-01-01", "2009-01-01"),
        ("2011-01-01", "2012-01-01"),
        ("2015-01-01", "2016-01-01"),
        ("2019-01-01", "2020-01-01"),
        ("2023-01-01", "2024-01-01"),
    ]

    for start, end in overlaps:
        ax.axvspan(
            pd.Timestamp(start),
            pd.Timestamp(end),
            color="gray",
            alpha=0.15,
            zorder=0
        )

    ax.plot(
        monthly_series.index,
        monthly_series.values,
        linewidth=2.5,
        color="#4285F4",
        label="Monthly Backbone"
    )

    ax.plot(
        spliced_series.index,
        spliced_series.values,
        linewidth=1,
        alpha=0.9,
        color="#EA4335",
        label="Spliced Weekly"
    )

    ax.set_ylabel("Search Interest")
    apply_style(ax, [40, 60, 80, 100])
    ax.legend(fontsize=12)

    save_figure(fig, GRAPH2_PNG, GRAPH2_PDF)


if __name__ == "__main__":
    graph_2()
    print(f"Done. Graph saved to: {OUT_DIR}")