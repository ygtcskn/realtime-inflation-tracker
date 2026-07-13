import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # noqa: F401

from config import DATA_FINAL, DATA_OUTPUT, OUTPUT_GRAPHS

warnings.filterwarnings("ignore")

OUTPUT_GRAPHS.mkdir(parents=True, exist_ok=True)

C_PRED_PATH = DATA_OUTPUT / "C_onemodel_LSTM" / "C_onemodel_LSTM_test_predictions.csv"

INPUT_PATH = DATA_FINAL / "C_panel.csv"

GRAPH10_ALL_PNG = OUTPUT_GRAPHS / "Graph_10_all_weekly_tracking.png"
GRAPH10_ALL_PDF = OUTPUT_GRAPHS / "Graph_10_all_weekly_tracking.pdf"


ALL_COUNTRIES = [
    "BR", "CA", "FR", "DE", "IN", "ID", "IT", "JP",
    "MX", "RU", "ZA", "KR", "GB", "US", "TR"
]

GRAPH10_GRID_COUNTRIES = ["US", "DE", "MX", "KR", "GB", "FR", "IT", "CA"]

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

# Per-country headroom multiplier (default 0.18). Bump for countries with
# tall peaks where the upper-right legend still overlaps the lines.
HEADROOM_OVERRIDES = {
    "JP": 0.60,
    "TR": 0.60,
}

WEEK_POSITIONS = [1, 2, 3, 4]

WEEK_COLORS = {
    1: "#4285F4",
    2: "#DB4437",
    3: "#F4B400",
    4: "#0F9D58",
}

DATE_COL = "date"
COUNTRY_COL = "country"
PRED_COL = "predicted"
WEEK_COL = "week_position"

INFL_DATE_COL = "date"
INFL_COUNTRY_COL = "country"
INFL_ACTUAL_COL = "infl_yoy"

START_DATE = "2022-01-01"
END_DATE = "2024-12-31"

X_LABEL_EVERY_STANDALONE = 4
X_LABEL_EVERY_GRID = 12

BG_COLOR = "#F0F0F0"
NEUTRAL_FIG = "#FFFFFF"
NEUTRAL_EDGE = "#D0D0D0"
NEUTRAL_TEXT = "#222222"

ACTUAL_COLOR = "#333333"
PATH_COLOR = "#7F8C8D"

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
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "legend.fontsize": 12,
    "legend.framealpha": 0.95,
    "legend.edgecolor": NEUTRAL_EDGE,
    "legend.facecolor": NEUTRAL_FIG,
})


def save_fig(fig, png_path, pdf_path, dpi=300):
    png_path = Path(png_path)
    pdf_path = Path(pdf_path)

    png_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=fig.get_facecolor()
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor=fig.get_facecolor()
    )

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

    plt.close(fig)


def clean_week_position(series):
    return (
        series.astype(str)
        .str.replace("week_", "", regex=False)
        .str.replace("Week_", "", regex=False)
        .str.replace("week", "", regex=False)
        .str.replace("Week", "", regex=False)
        .str.replace("W", "", regex=False)
        .str.replace("w", "", regex=False)
        .str.replace(" ", "", regex=False)
        .astype(int)
    )


def apply_weekly_x_axis(ax, df_plot, label_every=4, show_labels=True):
    axis_df = (
        df_plot[["x_week", "x_label"]]
        .drop_duplicates()
        .sort_values("x_week")
    )

    ticks = axis_df["x_week"].tolist()
    labels = axis_df["x_label"].tolist()

    ax.set_xticks(ticks[::label_every])

    if show_labels:
        ax.set_xticklabels(
            labels[::label_every],
            rotation=45,
            ha="right",
            fontsize=11
        )
    else:
        ax.set_xticklabels([])

    ax.set_xlim(df_plot["x_week"].min() - 1, df_plot["x_week"].max() + 1)


def prepare_graph10_data():
    pred = pd.read_csv(C_PRED_PATH)

    pred[DATE_COL] = pd.to_datetime(pred[DATE_COL])
    pred[WEEK_COL] = clean_week_position(pred[WEEK_COL])

    pred = pred[
        (pred[COUNTRY_COL].isin(ALL_COUNTRIES))
        & (pred[DATE_COL] >= START_DATE)
        & (pred[DATE_COL] <= END_DATE)
        & (pred[WEEK_COL].isin(WEEK_POSITIONS))
    ].copy()

    if pred.empty:
        raise ValueError("No prediction data found for Graph 10.")

    pred["month"] = pred[DATE_COL].dt.to_period("M")

    infl = pd.read_csv(INPUT_PATH)

    infl[INFL_DATE_COL] = pd.to_datetime(infl[INFL_DATE_COL])

    infl = infl[
        (infl[INFL_COUNTRY_COL].isin(ALL_COUNTRIES))
        & (infl[INFL_DATE_COL] >= START_DATE)
        & (infl[INFL_DATE_COL] <= END_DATE)
    ].copy()

    if infl.empty:
        raise ValueError("No inflation data found for Graph 10.")

    infl["month"] = infl[INFL_DATE_COL].dt.to_period("M")

    infl_monthly = (
        infl
        .sort_values(INFL_DATE_COL)
        .drop_duplicates([INFL_COUNTRY_COL, "month"])
        [[INFL_COUNTRY_COL, "month", INFL_ACTUAL_COL]]
        .copy()
    )

    infl_monthly = infl_monthly.rename(
        columns={INFL_COUNTRY_COL: COUNTRY_COL}
    )

    df = pred.merge(
        infl_monthly,
        on=[COUNTRY_COL, "month"],
        how="left"
    )

    df = df.dropna(subset=[INFL_ACTUAL_COL])

    if df.empty:
        raise ValueError(
            "After merging predictions with inflation, no Graph 10 observations remain."
        )

    all_country_frames = []

    for country in ALL_COUNTRIES:
        sub = df[df[COUNTRY_COL] == country].copy()

        if sub.empty:
            continue

        sub = sub.sort_values(["month", WEEK_COL]).copy()

        months_sorted = sorted(sub["month"].unique())
        month_to_index = {m: i for i, m in enumerate(months_sorted)}

        sub["month_index"] = sub["month"].map(month_to_index)
        sub["x_week"] = sub["month_index"] * 4 + (sub[WEEK_COL] - 1)

        sub["week_label"] = "W" + sub[WEEK_COL].astype(str)
        sub["x_label"] = sub["month"].astype(str) + " " + sub["week_label"]

        actual_anchor = (
            sub[sub[WEEK_COL] == 4]
            [["month", "x_week", INFL_ACTUAL_COL]]
            .drop_duplicates("month")
            .sort_values("x_week")
            .rename(columns={INFL_ACTUAL_COL: "actual_monthly"})
        )

        weekly_axis = (
            sub[["month", WEEK_COL, "x_week", "x_label"]]
            .drop_duplicates()
            .sort_values("x_week")
            .copy()
        )

        weekly_actual = weekly_axis.merge(
            actual_anchor[["x_week", "actual_monthly"]],
            on="x_week",
            how="left"
        )

        weekly_actual["actual_inflation_weekly"] = (
            weekly_actual["actual_monthly"]
            .interpolate(method="linear")
            .ffill()
            .bfill()
        )

        sub = sub.merge(
            weekly_actual[["x_week", "actual_inflation_weekly"]],
            on="x_week",
            how="left"
        )

        all_country_frames.append(sub)

    if not all_country_frames:
        raise ValueError("No country-specific Graph 10 data created.")

    return pd.concat(all_country_frames, ignore_index=True)


def add_graph10_country_panel(
    ax,
    df_plot,
    country,
    show_ylabel=True,
    show_xlabel=True,
    label_every=4,
    show_xlabels=True,
    small_panel=False
):
    ax.set_facecolor(BG_COLOR)

    sub = df_plot[df_plot[COUNTRY_COL] == country].copy()

    country_name = COUNTRY_NAMES.get(country, country)

    if sub.empty:
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax.transAxes
        )
        ax.set_title(country_name, fontsize=13)
        return {}

    sub = sub.sort_values("x_week")

    legend_items = {}

    line_actual, = ax.plot(
        sub["x_week"],
        sub["actual_inflation_weekly"],
        color=ACTUAL_COLOR,
        lw=2.2 if not small_panel else 1.9,
        zorder=10,
        label="Interpolated weekly inflation"
    )
    legend_items["Interpolated weekly inflation"] = line_actual

    line_path, = ax.plot(
        sub["x_week"],
        sub[PRED_COL],
        color=PATH_COLOR,
        lw=1.5 if not small_panel else 1.2,
        alpha=0.9,
        zorder=4,
        label="Weekly nowcast path"
    )
    legend_items["Weekly nowcast path"] = line_path

    for week in WEEK_POSITIONS:
        week_sub = sub[sub[WEEK_COL] == week].copy()

        dots = ax.scatter(
            week_sub["x_week"],
            week_sub[PRED_COL],
            color=WEEK_COLORS[week],
            s=26 if not small_panel else 18,
            zorder=6,
            label=f"Week {week}"
        )
        legend_items[f"Week {week}"] = dots

    ax.set_title(
        f"{country_name} — Real-Time Inflation Tracking",
        color=NEUTRAL_TEXT,
        fontweight="bold",
        fontsize=13
    )

    ax.set_ylabel("Inflation YoY (%)" if show_ylabel else "", fontsize=13)
    ax.set_xlabel("")

    ax.grid(True, alpha=0.4)
    ax.tick_params(axis="both", labelsize=11, colors="#555555")

    apply_weekly_x_axis(
        ax,
        sub,
        label_every=label_every,
        show_labels=show_xlabels
    )

    return legend_items


def plot_graph10_country_standalone(df_graph10, country):
    sub = df_graph10[df_graph10[COUNTRY_COL] == country].copy()

    if sub.empty:
        print(f"Skipping Graph 10 for {country}: no data.")
        return

    fig, ax = plt.subplots(
        figsize=(16, 5),
        facecolor=NEUTRAL_FIG
    )

    add_graph10_country_panel(
        ax,
        df_graph10,
        country,
        show_ylabel=True,
        show_xlabel=True,
        label_every=X_LABEL_EVERY_STANDALONE,
        show_xlabels=True,
        small_panel=False
    )

    # Add headroom above the data so the legend in the upper-right doesn't overlap lines.
    headroom = HEADROOM_OVERRIDES.get(country, 0.18)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + (ymax - ymin) * headroom)

    ax.legend(
        loc="upper right",
        frameon=True,
        fontsize=12
    )

    plt.tight_layout()

    png = OUTPUT_GRAPHS / f"Graph_10_weekly_tracking_{country}.png"
    pdf = OUTPUT_GRAPHS / f"Graph_10_weekly_tracking_{country}.pdf"

    save_fig(fig, png, pdf)


def plot_graph10_all_grid(df_graph10, countries=GRAPH10_GRID_COUNTRIES):
    nrows = 4
    ncols = 2

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(18, 4.2 * nrows),
        sharex=False,
        sharey=False,
        facecolor=NEUTRAL_FIG
    )

    axes = np.array(axes).reshape(nrows, ncols)

    legend_items = {}

    for i, country in enumerate(countries[:8]):
        r = i // ncols
        c = i % ncols

        ax = axes[r, c]

        items = add_graph10_country_panel(
            ax,
            df_graph10,
            country,
            show_ylabel=True,
            show_xlabel=(r == nrows - 1),
            label_every=X_LABEL_EVERY_GRID,
            show_xlabels=(r == nrows - 1),
            small_panel=True
        )

        legend_items.update(items)

    fig.legend(
        list(legend_items.values()),
        list(legend_items.keys()),
        ncol=6,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        frameon=True,
        fontsize=14,
        markerscale=1.25,
        handlelength=2.2,
        handletextpad=0.7,
        borderpad=0.6,
        labelspacing=0.6
    )

    plt.tight_layout(rect=[0, 0.045, 1, 1])
    save_fig(fig, GRAPH10_ALL_PNG, GRAPH10_ALL_PDF)


if __name__ == "__main__":
    print("Preparing Graph 10 data...")
    graph10_data = prepare_graph10_data()

    print("\nCreating Graph 10 standalone country graphs...")
    for country in ALL_COUNTRIES:
        plot_graph10_country_standalone(graph10_data, country)

    print("\nCreating Graph 10 all-grid 4x2...")
    plot_graph10_all_grid(graph10_data, countries=GRAPH10_GRID_COUNTRIES)

    print("\nDone.")
