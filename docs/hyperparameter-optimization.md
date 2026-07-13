# Hyperparameter optimization

## Scope and provenance

The LSTM and XGBoost models were tuned with Optuna in a separate thesis workspace before the final forecast evaluation. This page is based on an inspection of the surviving tuning scripts, study databases, configuration files, trial tables, and best-parameter files from that private archive.

The optimization code and its generated artifacts are not included in this repository. They depend on private model panels, contain machine-specific paths, and were not prepared as a portable release. The repository instead retains the selected parameters in the executable model scripts and documents the historical search here. The values below were read from archived artifacts; the studies were not rerun during this documentation pass.

## Tuned model variants

The tuning workspace distinguishes three information designs:

| Code | Design | Optimization role |
|---|---|---|
| A | Monthly simple-average panel | Monthly comparison model |
| B | U-MIDAS weekly panel | Main four-week mixed-frequency model |
| C | One-Model-Fits-All panel | Joint model across week positions 1-4 |

The archived LSTM studies also distinguish two architectural treatments of lagged inflation:

- `LSTM`: Google Trends, financial variables, and lagged inflation enter the sequential LSTM path; country indicators remain static.
- `ARX_LSTM`: Google Trends and financial variables enter the LSTM path, while lagged inflation enters the static fully connected path with the country indicators. In C, `week_position` is also static.

Study names ending in `_vs` use separately constructed variable-selected panels. A LassoCV stage fitted to the Approach-A data before 2020 selected 20 Google Trends variables and four financial variables; that 24-variable set was then mapped to the A/B/C panel designs. The private `*_fin_vs.csv` panels are not part of this repository. Their scores should therefore not be compared with full-panel scores as if only the estimator had changed.

No separate D optimization script survives in the inspected workspace. The current Week-Specific implementation reuses the B configuration because D's week-4 information set matches the B design.

## Temporal validation design

Hyperparameters were selected before the January 2020 test period. The retained LSTM studies and the later-generation A/B/C XGBoost scripts use three expanding validation blocks:

| Fold | Training observations | Validation observations |
|---:|---|---|
| 1 | Dates before 1 January 2017 | Calendar year 2017 |
| 2 | Dates before 1 January 2018 | Calendar year 2018 |
| 3 | Dates before 1 January 2019 | Calendar year 2019 |

For the retained LSTM studies and later-generation XGBoost scripts, the objective is minimized and is calculated as follows:

1. fit the model on observations strictly earlier than the validation block;
2. calculate RMSE separately for each represented country;
3. average country RMSEs within the fold, giving each country equal weight;
4. average the three fold scores.

For C LSTM-family studies, the objective is evaluated on week-position 4 predictions, although the joint model is trained with all four week positions. The C XGBoost tuner instead scores all week positions.

The LSTM scripts use contemporaneous tracking horizon `h = 0`, require at least 60 training sequences, and fit feature and target scalers on the corresponding training fold. The XGBoost scripts require at least 60 training rows and use raw, zero-filled inputs without feature or target scaling.

This validation design preserves time order and keeps the 2020-2025 test period outside hyperparameter selection. In the LSTM studies, the same validation blocks are used for early stopping and trial scoring, so the archived score is a model-selection criterion rather than an unbiased estimate from a nested validation procedure. The XGBoost tuner does not use early stopping.

## Optuna configuration

The LSTM studies use:

- 150 trials per study;
- minimization with Optuna's Tree-structured Parzen Estimator sampler;
- random seed 42;
- a maximum of 75 epochs per fold;
- early-stopping patience of 10 epochs;
- gradient clipping at 1.0;
- SQLite study storage with resume and backup behavior;
- CUDA when available, otherwise CPU.

Trials run sequentially and the archived scripts do not configure a pruner. Study names and databases must be versioned carefully: loading an existing SQLite study under the same name can combine trials produced by different code or data revisions.

### LSTM search space

| Hyperparameter | Search definition |
|---|---|
| Hidden size | `32`, `64`, or `128` |
| Sequence length | Integer from `6` to `24` months |
| LSTM layers | Integer from `1` to `2` |
| Fully connected size | `32`, `64`, or `128` |
| Dropout | Continuous from `0.0` to `0.5` |
| Learning rate | Log-uniform from `1e-4` to `3e-3` |
| Batch size | `16`, `32`, or `64` |
| Weight decay | Log-uniform from `1e-5` to `1e-2` |

### XGBoost search space

The archived A/B/C XGBoost scripts use 100 trials, TPE sampling, seed 42, and the same three validation blocks.

| Hyperparameter | Search definition |
|---|---|
| Trees | Integer from `100` to `800` |
| Maximum depth | Integer from `2` to `8` |
| Learning rate | Log-uniform from `0.005` to `0.2` |
| Subsample | Continuous from `0.5` to `1.0` |
| Column sample by tree | Continuous from `0.5` to `1.0` |
| Minimum child weight | Integer from `1` to `20` |
| L1 regularization | Log-uniform from `1e-3` to `10` |
| L2 regularization | Log-uniform from `1e-3` to `10` |
| Gamma | Continuous from `0.0` to `2.0` |

An older, separately archived v2 generation of A/B/C XGBoost studies used pooled RMSE and a narrower search space. Those results are not combined with the later equal-country study documented here because the objectives and search domains are not directly comparable.

## Surviving LSTM study results

Each study listed below has 150 rows in its archived trial table. Best CV RMSE is the mean of the three equal-country fold scores described above.

| Study | Best CV RMSE | Hidden | Sequence | Layers | FC | Dropout | Learning rate | Batch | Weight decay |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_ARX_LSTM | 0.419237 | 128 | 18 | 2 | 128 | 0.188223 | 0.00142335 | 16 | 0.000162879 |
| B_ARX_LSTM | 0.419510 | 128 | 23 | 2 | 128 | 0.061183 | 0.00109129 | 16 | 0.000396788 |
| B_ARX_LSTM_vs | 0.395828 | 32 | 7 | 1 | 128 | 0.365835 | 0.00104939 | 32 | 0.00498476 |
| B_LSTM | 1.000783 | 128 | 18 | 2 | 64 | 0.449791 | 0.00260245 | 32 | 0.000081411 |
| B_LSTM_vs | 0.538913 | 128 | 10 | 1 | 64 | 0.148109 | 0.00103950 | 16 | 0.00559223 |
| C_ARX_LSTM | 0.418843 | 128 | 23 | 2 | 128 | 0.307189 | 0.000674892 | 16 | 0.000510096 |
| C_LSTM | 0.917129 | 32 | 23 | 1 | 64 | 0.000059 | 0.000205706 | 64 | 0.00136849 |
| C_LSTM_vs | 0.492227 | 64 | 12 | 1 | 32 | 0.280492 | 0.00229752 | 16 | 0.00119453 |

The workspace contains additional A/B/C study scripts for which a complete matching output folder is not present in the inspected optimization directory. This table therefore reports only study results supported by a surviving trial table and best-parameter artifact.

## Selected XGBoost result

The surviving B XGBoost study contains 100 trials. Its best trial is trial 73 with a CV RMSE of 0.494162:

| Hyperparameter | Selected value |
|---|---:|
| Trees | 142 |
| Maximum depth | 4 |
| Learning rate | 0.0435873 |
| Subsample | 0.790794 |
| Column sample by tree | 0.967185 |
| Minimum child weight | 10 |
| L1 regularization | 8.146079 |
| L2 regularization | 0.149891 |
| Gamma | 0.440308 |

The A and C XGBoost tuning scripts survive, but their matching current result artifacts were not found alongside the inspected B study. No A or C best score is asserted here. The surviving C tuner also includes `week_position` twice in its feature matrix, so it should be corrected before any future C study.

## Relationship to the current model scripts

| Current implementation | Archived tuning provenance |
|---|---|
| [`B_umidas/B_LSTM.py`](../analysis/prog/model/B_umidas/B_LSTM.py) | Its eight tuned constants exactly match the archived `B_ARX_LSTM` best trial. |
| [`D_weekspecific/D_weekspecific_LSTM.py`](../analysis/prog/model/D_weekspecific/D_weekspecific_LSTM.py) | Reuses the same `B_ARX_LSTM` configuration. |
| [`B_umidas/B_XGB.py`](../analysis/prog/model/B_umidas/B_XGB.py) | Its nine XGBoost constants exactly match the archived `B_ARX_XGB` best trial. |
| [`C_onemodel/C_onemodel_LSTM.py`](../analysis/prog/model/C_onemodel/C_onemodel_LSTM.py) | Its constants match an earlier archived `C_ARX_LSTM` best trial with CV RMSE 0.426904. A later surviving C study selected a different configuration with CV RMSE 0.418843; the current code preserves the earlier selected configuration. |

This distinction matters for traceability: the documentation records the parameters actually retained in the executable scripts and does not silently replace them with a later archived trial.

The B XGBoost parameter values match exactly, but the fitting protocols are not identical: the archived tuner uses the zero-filled inputs directly, while the current forecast script standardizes continuous inputs inside each training window. Tree split ordering is generally invariant to linear rescaling, but this remains a reproducibility difference that should be recorded.

## Audit qualifications

- The variable-selection stage used the full pre-2020 sample, including the 2017-2019 Optuna validation years. The lower `_vs` scores therefore include feature-selection leakage and are exploratory rather than unbiased validation gains. A new study should repeat selection inside every fold.
- C LSTM-family trials use all four week positions for model fitting and early-stopping loss, but rank trials using equal-country RMSE on week 4 only. Archived C LSTM `n_val` fields count all four weekly rows, and the reported training-versus-validation diagnostic is not like-for-like.
- LSTM early stopping and hyperparameter selection reuse the same validation blocks. The archived LSTM CV values are selection criteria; the untouched 2020-2025 period is the subsequent test boundary.
- The sequence-building loop in the archived tuners omits the final otherwise eligible sequence. This is the same endpoint issue identified for the retained model scripts in [limitations](limitations.md).
- The historical optimization log spans multiple runner and script versions. The study database, configuration, trial table, and selected model constants provide stronger provenance than the append-only log alone.

## Archived outputs and reproduction boundary

The separate workspace generated, depending on study:

- the full trial table and best-trial CSV;
- `best_params.json` and a LaTeX parameter table;
- per-fold validation scores;
- a SQLite Optuna study and dated backups;
- convergence, parameter-importance, parallel-coordinate, and RMSE-distribution plots;
- training-versus-validation and learning-curve diagnostics;
- configuration metadata containing folds, search spaces, and input paths.

These artifacts are evidence that tuning was conducted, but they are not a portable replication package. A clean rerun would require the private historical panels, portable path configuration, Optuna as an additional dependency, a resolved software environment, and substantial compute for the recurrent-model studies. It should also retain the study database and an immutable run manifest so that the final model constants can be traced to one designated study version.
