import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from config import DATA_TEMP, OUTPUT_GRAPHS

OUT_DIR = OUTPUT_GRAPHS
OUT_DIR.mkdir(parents=True, exist_ok=True)

PC1_PATH = DATA_TEMP / "preprocessing" / "d_detrend" / "US_weekly_pc1_trend.csv"

GRAPH4_PNG = OUT_DIR / "Graph_4_Panel_ABC.png"
GRAPH4_PDF = OUT_DIR / "Graph_4_Panel_ABC.pdf"


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


def graph_4():
    df_raw_weekly = pd.read_csv(
        DATA_TEMP / "3_weekly_clean.csv",
        index_col=0,
        parse_dates=True
    )

    df_det_weekly = pd.read_csv(
        DATA_TEMP / "4_weekly_detrended_final.csv",
        index_col=0,
        parse_dates=True
    )

    df_pc1 = pd.read_csv(PC1_PATH, parse_dates=["date"]).set_index("date")

    categories = [
        "shopping",
        "travel",
        "finance",
        "real_estate",
        "jobs",
        "restaurants",
        "autos_vehicles",
        "health",
        "computers_electronics",
        "home_garden"
    ]

    series = [f"US_{category}" for category in categories]

    grey_colors = plt.cm.Greys(np.linspace(0.35, 0.85, len(series)))
    color_map = cm.get_cmap("tab10", len(series))

    existing_series = [
        col for col in series
        if col in df_raw_weekly.columns and col in df_det_weekly.columns
    ]

    all_vals = pd.concat(
        [df_raw_weekly[existing_series], df_det_weekly[existing_series]],
        axis=0
    ).values.flatten()

    all_vals = all_vals[~pd.isna(all_vals)]

    y_min = np.floor(all_vals.min() / 10) * 10
    y_max = np.ceil(all_vals.max() / 10) * 10
    yticks_ab = list(np.arange(y_min, y_max + 1, 20))

    if len(yticks_ab) < 3:
        yticks_ab = list(np.linspace(y_min, y_max, 4))

    pc1_min = df_pc1["pc1_trend"].min()
    pc1_max = df_pc1["pc1_trend"].max()

    pc1_low = np.floor(pc1_min * 2) / 2
    pc1_high = np.ceil(pc1_max * 2) / 2
    yticks_c = list(np.arange(pc1_low, pc1_high + 0.01, 0.5))

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 2)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    pos_a = ax_a.get_position()
    pos_c = ax_c.get_position()
    new_width = pos_a.width
    new_x = 0.5 - new_width / 2
    ax_c.set_position([new_x, pos_c.y0, new_width, pos_c.height])

    for i, col in enumerate(series):
        if col in df_raw_weekly.columns:
            ax_a.plot(
                df_raw_weekly.index,
                df_raw_weekly[col],
                color=grey_colors[i],
                linewidth=1
            )

    ax_a.set_title("Panel A: Raw Categories", fontsize=20)
    ax_a.set_ylabel("Search Interest")
    apply_style(ax_a, yticks_ab)

    for i, col in enumerate(series):
        if col in df_det_weekly.columns:
            ax_b.plot(
                df_det_weekly.index,
                df_det_weekly[col],
                color=color_map(i),
                linewidth=1
            )

    ax_b.set_title("Panel B: Detrended Categories", fontsize=20)
    apply_style(ax_b, yticks_ab)

    ax_c.plot(
        df_pc1.index,
        df_pc1["pc1_trend"],
        color="#4285F4",
        linewidth=2.5
    )

    ax_c.set_title("Panel C: US Weekly Secular Trend (PC1)", fontsize=20)
    ax_c.set_ylabel("Trend")
    apply_style(ax_c, yticks_c)

    fig.tight_layout()
    fig.savefig(GRAPH4_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(GRAPH4_PDF, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    graph_4()
    print(f"Done. Graph saved to: {OUT_DIR}")