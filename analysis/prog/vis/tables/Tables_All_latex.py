import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from config import OUTPUT_TABLES

TABLES_PY_DIR = Path(__file__).resolve().parent

OUT_PATH = OUTPUT_TABLES / "Tables_All.tex"

SUMMARY_LABELS = {
    "Average",
    "Avg excl. TR",
    "Beats AR (N)",
    "Winner",
    "Median",
}


# ============================================================
# Helpers
# ============================================================

def load_module(filename):
    path = TABLES_PY_DIR / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def esc_cell(s):
    """
    Escape normal text/data values.
    Keeps intentional LaTeX commands untouched.
    """
    s = str(s)

    if any(cmd in s for cmd in [
        r"\textbf",
        r"\textit",
        r"$",
        r"\Delta",
        r"\rightarrow",
        r"\%",
    ]):
        return s

    for old, new in [
        ("&", r"\&"),
        ("%", r"\%"),
        ("#", r"\#"),
        ("_", r"\_"),
    ]:
        s = s.replace(old, new)

    return s


def esc_header(s):
    return esc_cell(s)


def fmt(val, dec=4):
    if val is None:
        return "---"

    if isinstance(val, float) and np.isnan(val):
        return "---"

    if isinstance(val, (float, np.floating)):
        return f"{val:.{dec}f}"

    if isinstance(val, (int, np.integer)) and not isinstance(val, bool):
        return str(int(val))

    return esc_cell(str(val))


def add_row_asterisk(value):
    return f"{value}$^{{*}}$"


def table_latex(
    df,
    col_decs,
    caption,
    label,
    note="",
    col_align=None,
    resize=False,
    small=False,
    clearpage=True,
    column_renames=None,
    use_asterisk_lowest=False,
    lowest_cols=None,
    full_width=False,
):
    """
    Manual LaTeX style:
    - [H]
    - hline
    - optional resizebox
    - optional full-width tabular*
    - automatically fixes column alignment length
    """

    if column_renames:
        df = df.rename(columns=column_renames)
        col_decs = {column_renames.get(k, k): v for k, v in col_decs.items()}

    cols = list(df.columns)
    ncols = len(cols)

    # Automatically match alignment with number of columns
    if col_align is None:
        col_align = "l" + "c" * (ncols - 1)
    else:
        simple_align = (
            col_align
            .replace("@{\\extracolsep{\\fill}}", "")
            .replace("@{}", "")
            .replace("|", "")
        )
        simple_align = "".join(ch for ch in simple_align if ch in "lcrpmbX")
        if len(simple_align) != ncols:
            col_align = "l" + "c" * (ncols - 1)

    first_col = cols[0]

    lines = [
        r"\begin{table}[H]",
        r"\centering",
    ]

    if small:
        lines.append(r"\small")

    lines.extend([
        f"\\caption{{{caption}}}",
        f"\\label{{tab:{label}}}",
    ])

    if resize:
        lines.append(r"\resizebox{\textwidth}{!}{%")

    if full_width:
        lines.append(
            rf"\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}{col_align}@{{}}}}"
        )
    else:
        lines.append(rf"\begin{{tabular}}{{{col_align}}}")

    lines.extend([
        r"\hline",
        " & ".join(esc_header(c) for c in cols) + r" \\",
        r"\hline",
    ])

    for _, row in df.iterrows():
        first_val = str(row[first_col])

        if first_val in SUMMARY_LABELS:
            lines.append(r"\hline")

        cells = []

        for c in cols:
            cell = fmt(row[c], col_decs.get(c, 4))

            if (
                use_asterisk_lowest
                and lowest_cols
                and c in lowest_cols
                and first_val not in SUMMARY_LABELS
            ):
                numeric_values = []
                for lc in lowest_cols:
                    try:
                        numeric_values.append(float(row[lc]))
                    except Exception:
                        numeric_values.append(np.nan)

                try:
                    current_val = float(row[c])
                    min_val = np.nanmin(numeric_values)

                    if np.isclose(current_val, min_val, equal_nan=False):
                        cell = add_row_asterisk(cell)
                except Exception:
                    pass

            cells.append(cell)

        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\hline")

    if note:
        lines.append(rf"\multicolumn{{{ncols}}}{{l}}{{\small {note}}} \\")
        lines.append(r"\hline")

    if full_width:
        lines.append(r"\end{tabular*}")
    else:
        lines.append(r"\end{tabular}")

    if resize:
        lines.append(r"}")

    lines.append(r"\end{table}")

    if clearpage:
        lines.append("")
        lines.append(r"\clearpage")

    lines.append("")
    return "\n".join(lines)


# ============================================================
# Table builders
# ============================================================

def build_table_4():
    """
    Table 1 in your manual LaTeX:
    Average country-specific RMSE, MAE, MAPE.
    """
    mod = load_module("Table_4_umidas_avg_metrics.py")
    df = mod.create_table_1()

    col_decs = {
        "RMSE": 4,
        "MAE": 4,
        "MAPE": 2,
        "vs RW (%)": 2,
        "vs AR (%)": 2,
    }

    return table_latex(
        df,
        col_decs,
        caption=(
            "Average country-specific RMSE, MAE, and MAPE under the U-MIDAS "
            "configuration (Approach B). Percentage changes relative "
            "to Random Walk and AR(1) benchmarks."
        ),
        label="umidas-avg-rmse",
        col_align="lccccc",
    )


def build_table_5():
    """
    Table 2 in your manual LaTeX:
    Per-country RMSE and ratios.
    """
    mod = load_module("Table_5_country_relative.py")
    df_table, avg_row, avg_no_tr = mod.build_table()
    df = pd.concat([df_table, pd.DataFrame([avg_row, avg_no_tr])], ignore_index=True)

    col_decs = {
        "RW": 4,
        "AR(1)": 4,
        "LSTM": 4,
        "LSTM/AR": 2,
        "LSTM/RW": 2,
    }

    return table_latex(
        df,
        col_decs,
        caption=(
            "Per-country RMSE for the pooled LSTM model under the U-MIDAS "
            "configuration (Approach B), compared against Random Walk and AR(1) "
            "benchmarks. Asterisk (*) marks the lowest RMSE within each row. "
            "Ratio columns report LSTM RMSE relative to AR(1) and Random Walk, "
            "where values below 1 indicate improvement over the benchmark."
        ),
        label="umidas-per-country",
        note=(
            "* indicates the lowest RMSE in the row. "
            "LSTM beats AR in 9/15 countries and RW in 10/15 countries."
        ),
        col_align="lccccc",
        resize=True,
        use_asterisk_lowest=True,
        lowest_cols=["RW", "AR(1)", "LSTM"],
    )


def build_table_6():
    """
    Table 21 in your manual LaTeX:
    LSTM/AR ratios across phases.
    """
    mod = load_module("Table_6_lstm_ar_ratio_phases.py")
    df = mod.create_table_20()

    rename_map = {
        "Normalisation": "Normalization",
    }

    col_decs = {
        "Pandemic Shock": 2,
        "Inflation Surge": 2,
        "Normalisation": 2,
        "Normalization": 2,
    }

    return table_latex(
        df,
        col_decs,
        caption=(
            "LSTM RMSE relative to AR(1) across three phases of the "
            "inflation cycle under the U-MIDAS configuration. "
            "Values below 1.00 indicate that LSTM outperforms AR(1). "
            "All periods use out-of-sample predictions (test)."
        ),
        label="lstm-ar-ratio-phases-compact",
        col_align="lccc",
        small=True,
        column_renames=rename_map,
    )


def build_table_7():
    """
    Table 15 in your manual LaTeX:
    Pooled vs country-specific/subgroup comparison.
    This one is made wider so the Winner column stays at the end.
    """
    mod = load_module("Table_7_country_specific_and_subgroup_pooled.py")
    df = mod.create_table_15()

    rename_map = {
        "Pooled LSTM": "Pooled",
        "Alternative LSTM": "Country-Sp.",
    }

    col_decs = {
        "Pooled": 4,
        "Country-Sp.": 4,
    }

    return table_latex(
        df,
        col_decs,
        caption=(
            "Comparison of pooled and country-specific LSTM models under the "
            "U-MIDAS configuration. Country rows report country-level RMSEs. "
            "Subgroup rows report average RMSE over the countries in that subgroup. "
            "The row \\textit{wotur} compares the main pooled model with the model "
            "estimated without Turkey. Bold indicates the lower RMSE in each row."
        ),
        label="country-subgroup-combined",
        col_align="lccc",
        small=True,
        column_renames=rename_map,
        full_width=True,
    )


def build_table_8():
    """
    Table 8 in your manual LaTeX:
    C vs D average RMSE by week position.
    """
    mod = load_module("Table_8_cd_week_avg.py")
    df = mod.create_table_1_cd()

    rename_map = {
        "W1 to W4 (%)": r"W1$\rightarrow$W4 (\%)",
    }

    col_decs = {
        "W1": 4,
        "W2": 4,
        "W3": 4,
        "W4": 4,
        r"W1$\rightarrow$W4 (\%)": 2,
    }

    return table_latex(
        df,
        col_decs,
        caption=(
            "Average country-specific RMSE by week position: one-model-fits-all "
            "(Approach C) vs week-specific models (Approach D). W1 = earliest "
            "forecast, W4 = all weekly data available."
        ),
        label="c-vs-d",
        note="Benchmarks: RW = 0.8664, AR(1) = 0.8762",
        col_align="lccccc",
        column_renames=rename_map,
    )


def build_table_9():
    """
    Additional table:
    C-D information gain.
    Formatted like your manual LaTeX tables.
    """
    mod = load_module("Table_9_cd_info_gain.py")
    df = mod.create_table_6()

    col_decs = {
        "C W1": 4,
        "C W4": 4,
        "C W1 to W4 (%)": 2,
        "D W1": 4,
        "D W4": 4,
        "D W1 to W4 (%)": 2,
    }

    return table_latex(
        df,
        col_decs,
        caption="Country-level information gain from W1 to W4: models C and D.",
        label="cd-info-gain",
        resize=True,
    )


def build_table_10():
    """
    DM tables.
    Fixed: alignment now automatically matches the number of DataFrame columns,
    so Result/Winner/Sig. no longer moves to a new row.
    """
    mod = load_module("Table_10_DM.py")

    preds = {}
    for folder, (label, prefix) in mod.MODELS.items():
        try:
            preds[label] = mod.load_predictions(folder, prefix)
        except FileNotFoundError as e:
            print(f"  ! Skipping {label}: {e}")

    df_country = mod.build_country_cd_table(preds)
    df_pooled = mod.build_pooled_table(preds)

    country_renames = {
        "RMSE_C": "RMSE C",
        "RMSE_D": "RMSE D",
        "p_value": "p",
        "verdict": "Result",
        "sig": "Sig.",
    }

    pooled_renames = {
        "Model_1": "Model 1",
        "Model_2": "Model 2",
        "RMSE_1": "RMSE 1",
        "RMSE_2": "RMSE 2",
        "p_value": "p",
        "verdict": "Result",
        "sig": "Sig.",
    }

    decs_country = {
        "RMSE C": 4,
        "RMSE D": 4,
        "DM": 3,
        "p": 3,
    }

    decs_pooled = {
        "RMSE 1": 4,
        "RMSE 2": 4,
        "DM": 3,
        "p": 3,
    }

    tex_country = table_latex(
        df_country,
        decs_country,
        caption=(
            "Diebold--Mariano test: C (one-model) vs. D "
            "(week-specific) per country."
        ),
        label="dm-c-vs-d-country",
        note=(
            "DM statistic uses Harvey--Leybourne--Newbold correction. "
            r"Significance: $^{*}$p$<$0.10, $^{**}$p$<$0.05, $^{***}$p$<$0.01. "
            r"\textit{Result} shows the preferred model at the 5\% level; "
            r"\textit{ns} = not significant."
        ),
        resize=True,
        small=True,
        column_renames=country_renames,
    )

    tex_pooled = table_latex(
        df_pooled,
        decs_pooled,
        caption="Diebold--Mariano test: pooled pairwise model comparisons.",
        label="dm-pooled-comparisons",
        note=(
            "DM statistic uses Harvey--Leybourne--Newbold correction. "
            r"Significance: $^{*}$p$<$0.10, $^{**}$p$<$0.05, $^{***}$p$<$0.01."
        ),
        resize=True,
        small=True,
        column_renames=pooled_renames,
    )

    return tex_country + "\n" + tex_pooled


def build_table_11():
    """
    Per-country RMSE comparison across four models:
    AR(1), LSTM, XGBoost, and LASSO.
    """
    mod = load_module("Table_11_lstm_xgb.py")
    df_table, avg_row, avg_no_tr = mod.build_table()
    df = pd.concat([df_table, pd.DataFrame([avg_row, avg_no_tr])], ignore_index=True)

    col_decs = {
        "AR(1)": 4,
        "LSTM": 4,
        "XGBoost": 4,
        "LASSO": 4,
    }

    return table_latex(
        df,
        col_decs,
        caption=(
            "Per-country RMSE across four models: AR(1), LSTM, XGBoost, "
            "and LASSO. Asterisk (*) marks the lowest RMSE within each row."
        ),
        label="model-comparison",
        note="* indicates the lowest RMSE in the row.",
        col_align="lcccc",
        use_asterisk_lowest=True,
        lowest_cols=["AR(1)", "LSTM", "XGBoost", "LASSO"],
    )


# ============================================================
# Build order
# ============================================================

BUILDERS = [
    ("Table 1", build_table_4),
    ("Table 2", build_table_5),
    ("Table 21", build_table_6),
    ("Table 15", build_table_7),
    ("Table 8", build_table_8),
    ("Additional Table: C-D information gain", build_table_9),
    ("Additional Table: DM tests", build_table_10),
    ("Table 11: Model comparison", build_table_11),
]


# ============================================================
# LaTeX document header/footer
# ============================================================

HEADER = r"""\documentclass[12pt]{article}
\usepackage[a4paper,margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{caption}
\usepackage{array}
\usepackage{float}
\usepackage{graphicx}

\begin{document}

"""

FOOTER = r"""
\end{document}
"""


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)

    blocks = [HEADER]

    for name, builder in BUILDERS:
        print(f"Building {name}...", end=" ", flush=True)

        try:
            blocks.append("% ============================================================\n")
            blocks.append(f"% {name}\n")
            blocks.append("% ============================================================\n")
            blocks.append(builder())
            print("ok")

        except Exception as exc:
            print(f"FAILED: {exc}")
            blocks.append(f"% {name}: generation failed - {exc}\n\n")

    blocks.append(FOOTER)

    tex = "\n".join(blocks)
    OUT_PATH.write_text(tex, encoding="utf-8")

    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
