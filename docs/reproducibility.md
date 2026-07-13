# Reproducibility

## Scope

This release provides code traceability rather than a self-contained replication package. The source and transformation logic are visible, but the underlying Google Trends histories and other input data are excluded. Exact numerical reproduction therefore requires independently obtaining compatible source vintages and reconstructing the expected directory layout.

| Reproduction goal | Status | Reason |
|---|---|---|
| Inspect model and preprocessing logic | Available | Python source is included |
| Inspect the intended execution order | Available with caveats | `analysis/master.py` preserves the thesis workflow, but private inputs are not included |
| Rebuild financial and inflation inputs | Partial | Fetch scripts exist; providers can revise histories or change access behavior |
| Rebuild Google Trends inputs from scratch | Not self-contained | The repository begins from locally collected raw histories and does not include the collection archive |
| Regenerate thesis tables exactly | Not guaranteed | Private inputs, source vintages, and some historical scripts/artifacts are absent |
| Validate thesis conclusions | Possible as a new study | Requires fresh data acquisition, vintage-aware preprocessing, and a clean rerun |

## Environment

The direct runtime dependencies are listed in [`analysis/requirements.txt`](../analysis/requirements.txt). The file uses bounded compatibility ranges rather than a machine-specific environment freeze and targets Python 3.10-3.12. It includes pandas, NumPy, SciPy, scikit-learn, statsmodels, XGBoost, PyTorch, SHAP, yfinance, Matplotlib, openpyxl, requests, and tqdm. Transitive, notebook, and unrelated audio/vision packages are intentionally omitted.

The `torch>=2.2,<3` entry is platform-neutral and supports CPU execution. For NVIDIA CUDA or AMD ROCm acceleration, first use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) to install the build matching the operating system, Python version, and compute platform; then install `requirements.txt`. Because exact results can still vary within the declared ranges, record a resolved environment for an empirical rerun with `python -m pip freeze > requirements-lock.txt`.

A PowerShell setup from the repository root is:

~~~powershell
cd analysis
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
$env:PYTHONPATH = (Get-Location).Path
~~~

## Local configuration

[`analysis/config.py`](../analysis/config.py) now derives its root from its own file location:

~~~python
ROOT_DIR = Path(__file__).resolve().parent
~~~

This removes the original machine-specific Windows path. Run scripts with `analysis/` as the working directory so imports and relative output paths resolve consistently.

## Determinism

The model scripts set fixed seeds in their experiment configuration, and the main LSTM runs use seed 42. This reduces ordinary run-to-run variation but does not guarantee bit-for-bit determinism across PyTorch, CUDA, operating-system, or dependency versions.

## Archival orchestration status

The top-level orchestrator expresses the intended order, but it is not currently a one-command public reproduction:

- the data files it expects are intentionally absent;
- some later diagnostic/table scripts are not included in the orchestrator;
- hyperparameter values are embedded in model scripts, while the Optuna search workflow described in the thesis is not included.

For code review, use [pipeline.md](pipeline.md) as the authoritative map of surviving components. For a new empirical run, execute and validate one stage at a time rather than treating `python master.py` as a verified turnkey command.

## Thesis versus current snapshot

The submitted thesis states that financial Friday observations were shifted two days to Sunday before weekly alignment. The current panel-building script explicitly leaves the dates unchanged. The thesis also describes preprocessing within a no-future-information evaluation design, whereas several stored preprocessing steps in this snapshot are fit over complete histories before model evaluation.

These differences do not rewrite the submitted paper. They define the boundary of the archived implementation and the tests needed before claiming a fresh real-time replication. See [limitations.md](limitations.md) for the production-oriented interpretation.

## Recommended replication protocol

1. Record provider, retrieval date, geographic code, query/category identifier, units, timezone, and revision vintage for every source series.
2. Create contract tests for the schemas in [data.md](data.md).
3. Validate configuration-derived input and output locations on a clean checkout.
4. Refit every data-dependent transformation inside each expanding training window.
5. Add checks for source coverage, missing countries, five-week months, and sequence endpoints.
6. Recreate hyperparameter search with stored study metadata and fixed temporal folds.
7. Regenerate predictions and tables from a single versioned run manifest.
8. Compare the new outputs with thesis-reported values and explain material deviations.

## Citation context

The underlying study is:

> Ahmet Yigit Coskun (2026). *Tracking Inflation in the G20 at High Frequency and in Real Time: A Machine Learning Approach*. Economics M.Sc. thesis, Friedrich-Alexander-Universität Erlangen-Nürnberg.

The thesis PDF itself is not bundled in this code-only release.
