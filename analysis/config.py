from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
PROG_DIR = ROOT_DIR / "prog"

DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_TEMP = ROOT_DIR / "data" / "temp"
DATA_FINAL = ROOT_DIR / "data" / "final"
DATA_OUTPUT = ROOT_DIR / "output"

OUTPUT_TABLES = ROOT_DIR / "tables"
OUTPUT_GRAPHS = ROOT_DIR / "graphs"
