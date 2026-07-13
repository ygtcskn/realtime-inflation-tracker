import matplotlib.pyplot as plt
import pandas as pd

from config import DATA_RAW, OUTPUT_GRAPHS

OUT_DIR = OUTPUT_GRAPHS
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_MONTHLY_FILE = DATA_RAW / "raw_monthly" / "10048_US_m_04_25.csv"

GRAPH1_PNG = OUT_DIR / "Graph_1_Housepriceindex.png"
GRAPH1_PDF = OUT_DIR / "Graph_1_Housepriceindex.pdf"


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


def graph_1():
    df = pd.read_csv(RAW_MONTHLY_FILE, skiprows=1)
    df.columns = ["date", "shopping"]
    df["date"] = pd.to_datetime(df["date"])
    df["shopping"] = pd.to_numeric(df["shopping"], errors="coerce")

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(df["date"], df["shopping"], linewidth=2)

    ax.set_ylabel("Search Interest")
    apply_style(ax, [0, 20, 40, 60, 80, 100])

    save_figure(fig, GRAPH1_PNG, GRAPH1_PDF)


if __name__ == "__main__":
    graph_1()
    print(f"Done. Graph saved to: {OUT_DIR}")