# Analysis code

This directory contains the archival Python implementation used for the thesis. It preserves the original research-oriented folder structure while this guide provides a shorter technical entry point.

## Component map

| Path | Responsibility |
|---|---|
| `prog/prep/financial/` | Retrieve and transform market series and inflation |
| `prog/prep/gt/` | Splice Google Trends downloads, adjust breaks, denoise, detrend, transform, and build panels |
| `prog/model/0_benchmark/` | Random-walk and pooled autoregressive baselines |
| `prog/model/B_umidas/` | U-MIDAS LASSO, XGBoost, and LSTM experiments |
| `prog/model/C_onemodel/` | One-model-fits-all weekly-update LSTM |
| `prog/model/D_weekspecific/` | Separate LSTM models for weekly information sets |
| `prog/model/extra/` | Country/subgroup experiments, SHAP, permutation importance, and DM tests |
| `prog/vis/` | Thesis tables and figures |
| `master.py` | Original sequential orchestrator |
| `config.py` | Thesis-era path definitions |

For the transformation graph and input/output contracts, see [`docs/pipeline.md`](../docs/pipeline.md) and [`docs/data.md`](../docs/data.md).

## Intended stage order

1. Financial-market and inflation preparation.
2. Google Trends splicing and break adjustment.
3. Denoising, detrending, seasonal transformation, and panel construction.
4. Random-walk and AR(1) benchmarks.
5. U-MIDAS LASSO, XGBoost, and LSTM models.
6. Weekly-update LSTM variants.
7. Robustness, interpretability, statistical tests, figures, and tables.

Each stage reads or writes files beneath `analysis/data/` or generated-output directories. Those files are intentionally excluded from the public repository.

## Setup

Use Python 3.10-3.12, create a clean environment, and install the direct runtime dependencies:

~~~powershell
cd analysis
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
$env:PYTHONPATH = (Get-Location).Path
~~~

The unqualified `torch` requirement is portable and sufficient for CPU execution. For NVIDIA CUDA or AMD ROCm acceleration, use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) to install the build matching the operating system, Python version, and compute platform before installing `requirements.txt`. The requirements file does not include `torchvision` or `torchaudio` because the project does not import them.

Paths in [`config.py`](config.py) are derived from that file's location, so the checkout itself is portable. Run commands with `analysis/` as the working directory and keep that directory on `PYTHONPATH` so imports and outputs resolve consistently. In Bash or Zsh, the equivalent is `export PYTHONPATH="$PWD"`.

## Execution status

`master.py` is useful as an execution-order record, but this public snapshot is not a verified one-command run. It expects private data and does not invoke every diagnostic script.

For a new run:

1. reconstruct and validate the documented input schemas;
2. run one preprocessing stage at a time;
3. inspect row counts, country coverage, date ranges, missingness, and output paths;
4. run benchmarks before the higher-cost models;
5. save predictions and run metadata under a versioned experiment identifier.

Do not run the full pipeline against unreviewed inputs: several stages can take substantial time, fetch remote data, or overwrite derived files.

## Outputs

The scripts write derived panels, predictions, metrics, logs, figures, and tables to local subdirectories. Generated outputs are ignored by Git. The published numerical summary in [`docs/results.md`](../docs/results.md) is transcribed from the submitted thesis rather than asserted as a fresh execution of this snapshot.
