# Evaluation design

The submitted study uses an expanding-window pseudo-out-of-sample design. Random cross-validation is avoided because it would violate time ordering.

## Main forecast exercise

The intended evaluation:

1. begins with observations from January 2005 through December 2019;
2. estimates each country for January 2020;
3. adds the newly observed month to the training sample;
4. refits the model;
5. repeats through September 2025.

At each forecast date, pooled models estimate all 15 countries. Feature and target scalers inside the model scripts are fitted only on the corresponding training window. The target is monthly year-on-year inflation; high-frequency regressors represent current-month information, while inflation enters with a one-month lag.

The current snapshot has upstream preprocessing and sequence-boundary qualifications to this intended design; see [limitations.md](limitations.md).

## Hyperparameter validation

The thesis selects hyperparameters before the test period with three expanding validation blocks:

| Fold | Validation block |
|---|---|
| 1 | 2017 |
| 2 | 2018 |
| 3 | 2019 |

Each fold uses only earlier observations for training. The stated Optuna objective is the mean of country-specific RMSEs, giving every country equal weight.

See [hyperparameter optimization](hyperparameter-optimization.md) for the exact search spaces, trial counts, C-LSTM week-4 objective, and archived best-trial evidence.

## Accuracy metrics

For country $i$:

$$
RMSE_i =
\sqrt{\frac{1}{T_i}\sum_{t=1}^{T_i}(y_{i,t}-\hat y_{i,t})^2}.
$$

The headline score is the unweighted country average:

$$
\overline{RMSE} = \frac{1}{N}\sum_{i=1}^{N}RMSE_i.
$$

This differs from one pooled RMSE over every country-observation. Equal-country averaging prevents countries with more observations from dominating the headline score.

MAE is also reported:

$$
MAE_i = \frac{1}{T_i}\sum_{t=1}^{T_i}|y_{i,t}-\hat y_{i,t}|.
$$

RMSE emphasizes large misses; MAE is more resistant to outliers and remains in inflation-percentage-point units.

## Diebold-Mariano comparisons

The Diebold-Mariano test compares squared-error losses:

$$
d_t=e_{1,t}^2-e_{2,t}^2.
$$

A negative statistic favors Model 1; a positive value favors Model 2. The implementation applies the Harvey-Leybourne-Newbold small-sample correction.

The result is evidence about these forecast sequences, not a universal model ranking. It depends on the period, horizon, estimation window, and loss function. U-MIDAS comparisons are summarized in [results.md](results.md). The C-versus-D country tests require recomputation because of an alignment issue described in [limitations.md](limitations.md).

## Economic regimes

| Period | Dates |
|---|---|
| Pandemic shock | January-December 2020 |
| Inflation surge | January 2021-December 2023 |
| Normalization | January 2024-September 2025 |

This split examines whether model performance changes between a sudden shock, an extended inflation surge, and a more stable phase.

## Weekly tracking

Models C and D are evaluated at every within-month week position. The exercise asks:

- Does accuracy improve from week 1 to week 4 as information arrives?
- Is one joint model across all weekly states more effective than four smaller models?

Both aggregate and country-level RMSEs are reported.

## Evidence status

The numerical values in [results.md](results.md) are historical thesis outputs, not artifacts regenerated during this documentation pass. A fresh validation requires rebuilding private inputs, making preprocessing vintage-aware, correcting the audited alignment/boundary items, and rerunning the complete experiment.
