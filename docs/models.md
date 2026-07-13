# Model catalog

All models estimate current-month year-on-year inflation. The catalogue deliberately moves from persistence-based baselines to linear selection, nonlinear trees, and recurrent sequence learning.

## Repository mapping

| Directory | Model | Role |
|---|---|---|
| `analysis/prog/model/0_benchmark/` | Random walk and AR(1) | Univariate baselines |
| `analysis/prog/model/B_umidas/` | LASSO, XGBoost, and LSTM | Main U-MIDAS comparison |
| `analysis/prog/model/C_onemodel/` | One-Model-Fits-All LSTM | Joint weekly tracker |
| `analysis/prog/model/D_weekspecific/` | Week-Specific LSTMs | Separate model per weekly information set |
| `analysis/prog/model/extra/` | SHAP, permutation importance, pooling comparisons, and DM tests | Interpretation and robustness |

## Benchmarks

### Random walk

The latest observed inflation value becomes the forecast:

$$
\hat y_t = y_{t-1}.
$$

This is a demanding benchmark because inflation is persistent.

### AR(1)

The autoregressive benchmark estimates inflation from its first lag:

$$
y_t = \alpha + \phi y_{t-1} + \varepsilon_t.
$$

A common lag order is used across countries to preserve comparability.

## LASSO

LASSO is the linear high-dimensional benchmark. Its L1 penalty shrinks weak coefficients toward zero and can remove predictors entirely, which is useful when hundreds of search and financial variables may be redundant.

The current script searches 20 penalty values on a logarithmic grid with date-aware validation rather than a random split.

## XGBoost

XGBoost supplies a nonlinear tree-based comparison. Trees are added sequentially to correct earlier residuals, while depth, row/column sampling, and L1/L2 penalties constrain complexity.

The selected values retained in [`B_XGB.py`](../analysis/prog/model/B_umidas/B_XGB.py) are:

| Hyperparameter | Value |
|---|---:|
| Trees | 142 |
| Maximum depth | 4 |
| Learning rate | 0.0436 |
| Subsample | 0.7908 |
| Column sample | 0.9672 |
| Minimum child weight | 10 |
| L1 regularization | 8.1461 |
| L2 regularization | 0.1499 |
| Gamma | 0.4403 |

## LSTM

The LSTM learns nonlinear relationships across a sequence of historical country-month observations.

It separates:

- **sequential inputs:** Google Trends and financial features;
- **static inputs:** lagged inflation and country indicators, plus week position in Model C.

The final hidden state from a two-layer LSTM is concatenated with the static inputs. The combined representation passes through a fully connected layer with ReLU and dropout before producing the inflation estimate.

Training uses mean squared error, Adam, gradient clipping, a learning-rate scheduler, and seed 42.

### Current snapshot configuration

| Hyperparameter | B: U-MIDAS | C: One-Model-Fits-All | D: Week-Specific |
|---|---:|---:|---:|
| Hidden size | 128 | 128 | 128 |
| LSTM layers | 2 | 2 | 2 |
| Fully connected size | 128 | 128 | 128 |
| Sequence length | 23 | 14 | 23 |
| Dropout | 0.0612 | 0.3022 | 0.0612 |
| Learning rate | 0.001091 | 0.000574 | 0.001091 |
| Batch size | 16 | 32 | 16 |
| Weight decay | 0.000397 | 0.000212 | 0.000397 |
| Epochs | 100 | 100 | 100 |
| Gradient clip | 1.0 | 1.0 | 1.0 |

The D models reuse the B configuration because the D week-4 information set matches B. C is tuned separately because all four within-month states are learned jointly.

## Hyperparameter-search design

The archived LSTM studies use 150 Optuna trials with a Tree-structured Parzen Estimator; the surviving XGBoost study uses 100. The objective is the average country-level RMSE across expanding validation blocks for 2017, 2018, and 2019, giving every country equal weight. Hyperparameters are selected before the January 2020 test period and held fixed during evaluation.

The selected constants remain in this repository, while the Optuna executables and study storage remain in a separate private archive. [Hyperparameter optimization](hyperparameter-optimization.md) documents the inspected search spaces, study design, surviving results, and the connection between archived best trials and the current model scripts.

## Why compare several model classes?

- Random walk and AR(1) test whether high-frequency signals add value beyond persistence.
- LASSO tests whether a sparse linear relationship is enough.
- XGBoost tests nonlinear interactions without sequence modelling.
- LSTM tests whether temporal representation learning adds further value.

This makes complexity accountable: a flexible model is useful only if it improves genuinely out-of-sample forecasts.
