import os
import subprocess
import sys
import logging
from datetime import datetime

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from config import ROOT_DIR, PROG_DIR

PYTHON_EXE = sys.executable

LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / f"master_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

SCRIPTS = [
    PROG_DIR / "prep" / "financial" / "financial.py",
    PROG_DIR / "prep" / "financial" / "inflation.py",

    PROG_DIR / "prep" / "gt" / "1_splice.py",
    PROG_DIR / "prep" / "gt" / "2_breakadj.py",
    PROG_DIR / "prep" / "gt" / "3_denoise.py",
    PROG_DIR / "prep" / "gt" / "4_detrend.py",
    PROG_DIR / "prep" / "gt" / "5_logdiff.py",
    PROG_DIR / "prep" / "gt" / "6_panel.py",

    PROG_DIR / "model" / "0_benchmark" / "0_AR.py",
    PROG_DIR / "model" / "0_benchmark" / "0_RW.py",

    PROG_DIR / "model" / "B_umidas" / "B_LASSO.py",
    PROG_DIR / "model" / "B_umidas" / "B_LSTM.py",
    PROG_DIR / "model" / "B_umidas" / "B_XGB.py",

    PROG_DIR / "model" / "C_onemodel" / "C_onemodel_LSTM.py",
    PROG_DIR / "model" / "D_weekspecific" / "D_weekspecific_LSTM.py",

    PROG_DIR / "model" / "extra" / "B_LSTM_couspe.py",
    PROG_DIR / "model" / "extra" / "B_LSTM_perm.py",
    PROG_DIR / "model" / "extra" / "B_LSTM_shap.py",
    PROG_DIR / "model" / "extra" / "B_LSTM_wotur.py",

    PROG_DIR / "vis" / "graphs" / "Graph_1_Housepriceindex.py",
    PROG_DIR / "vis" / "graphs" / "Graph_2_Splice.py",
    PROG_DIR / "vis" / "graphs" / "Graph_3_Break_Raw.py",
    PROG_DIR / "vis" / "graphs" / "Graph_4_Panel_ABC.py",
    PROG_DIR / "vis" / "graphs" / "Graph_5_2_Test_Only.py",
    PROG_DIR / "vis" / "graphs" / "Graph_5_Full.py",
    PROG_DIR / "vis" / "graphs" / "Graph_6_SHAP_Beeswarm_Exogenous.py",
    PROG_DIR / "vis" / "graphs" / "Graph_7_2_Permutation_Period_Comparison.py",
    PROG_DIR / "vis" / "graphs" / "Graph_7_Permutation_Period_Comparison.py",
    PROG_DIR / "vis" / "graphs" / "Graph_8_CD_rmse_pct_change.py",
    PROG_DIR / "vis" / "graphs" / "Graph_9_Nowcast_CD.py",
    PROG_DIR / "vis" / "graphs" / "Graph_10_realtracking.py",

    PROG_DIR / "vis" / "tables" / "Table_4_umidas_avg_metrics.py",
    PROG_DIR / "vis" / "tables" / "Table_5_country_relative.py",
    PROG_DIR / "vis" / "tables" / "Table_6_lstm_ar_ratio_phases.py",
    PROG_DIR / "vis" / "tables" / "Table_7_country_specific_and_subgroup_pooled.py",
    PROG_DIR / "vis" / "tables" / "Table_8_cd_week_avg.py",
    PROG_DIR / "vis" / "tables" / "Table_9_cd_info_gain.py",
    PROG_DIR / "vis" / "tables" / "Table_10_DM.py",
    PROG_DIR / "vis" / "tables" / "Tables_All_latex.py",
]


def run_script(script_path):
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    print("\n" + "=" * 80)
    print(f"Running: {script_path}")
    print("=" * 80)

    result = subprocess.run(
        [PYTHON_EXE, str(script_path)],
        cwd=str(ROOT_DIR),
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT_DIR)}
    )

    if result.returncode != 0:
        raise RuntimeError(f"Script failed: {script_path}")

    logging.info(f"Finished: {script_path}")


def main():
    os.chdir(ROOT_DIR)

    logging.info("MASTER PIPELINE STARTED")

    for script in tqdm(SCRIPTS, desc="Pipeline", unit="script"):
        run_script(script)

    logging.info("MASTER PIPELINE COMPLETE")

    print("\n" + "=" * 80)
    print("MASTER PIPELINE COMPLETE")
    print(f"Log saved to: {LOG_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
