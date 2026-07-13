# Thesis-reported results

> [!IMPORTANT]
> The values below are transcribed from the submitted master's thesis. The underlying data are not distributed, and the metrics were not regenerated for this code release. Read them with [limitations.md](limitations.md), which separates the submitted findings from later implementation-audit observations.

## Main U-MIDAS comparison

Metrics are averaged over the 15 country test sets.

| Model | RMSE | MAE | RMSE vs. RW | RMSE vs. AR(1) |
|---|---:|---:|---:|---:|
| Random walk | 0.8717 | 0.5754 | 0.00% | -1.13% |
| AR(1) | 0.8817 | 0.5784 | 1.14% | 0.00% |
| LASSO | 0.8597 | 0.5688 | -1.38% | -2.50% |
| XGBoost | 1.4792 | 0.9836 | 69.69% | 67.77% |
| LSTM | **0.8006** | **0.5415** | **-8.16%** | **-9.20%** |

As submitted, the U-MIDAS LSTM has the lowest average RMSE and MAE. It reduces average country RMSE by 8.16% relative to the random walk and 9.20% relative to AR(1).

The result is heterogeneous:

- the LSTM has a lower RMSE than both univariate benchmarks in 9 of 15 countries;
- its strongest relative improvements over AR(1) occur in the United States, Turkey, and Brazil;
- it underperforms AR(1) most clearly in Japan, where its relative RMSE is 1.23;
- excluding Turkey, average RMSE is 0.5620 for LSTM, 0.5650 for LASSO, 0.5760 for XGBoost, 0.5804 for random walk, and 0.5848 for AR(1);
- Turkey materially affects the full-sample averages: its RMSE is 4.1411 for LSTM and 14.1239 for XGBoost.

## Reported U-MIDAS forecast tests

Appendix Table B.8 reports:

| Model 1 | Model 2 | DM statistic | p-value | Thesis interpretation |
|---|---|---:|---:|---|
| LSTM | AR(1) | -2.244 | 0.025 | LSTM lower loss |
| LSTM | Random walk | -2.243 | 0.025 | LSTM lower loss |
| LSTM | LASSO | -2.111 | 0.035 | LSTM lower loss |
| LSTM | XGBoost | -3.666 | <0.001 | LSTM lower loss |

These compare forecasts from the thesis evaluation period; they do not show that one model class is universally superior.

## Performance across regimes

The thesis reports the number of countries in which LSTM beats AR(1):

| Period | Countries beating AR(1) |
|---|---:|
| Pandemic shock, 2020 | 7 of 15 |
| Inflation surge, 2021-2023 | **10 of 15** |
| Normalization, 2024-Sep. 2025 | 3 of 15 |

The reported advantage is concentrated in the inflation-surge period. During normalization, the persistent AR(1) benchmark becomes harder to beat.

## Pooling versus country-specific models

The pooled U-MIDAS LSTM beats separately estimated country models in 10 of 15 countries. The thesis reports an average RMSE reduction of 9.43% from pooling.

Germany, India, Japan, South Korea, and the United Kingdom perform better with country-specific models. Among subgroup models, Europe is the only regional grouping that beats the corresponding estimates from the full 15-country panel.

The submitted interpretation is that the additional observations created by pooling generally matter more than strict cross-country homogeneity, while country indicators absorb part of the remaining differences.

## Weekly tracking

| Model | Week 1 | Week 2 | Week 3 | Week 4 | Week 1 to 4 |
|---|---:|---:|---:|---:|---:|
| One-Model-Fits-All | 0.7857 | 0.7877 | 0.7820 | **0.7768** | -1.13% |
| Week-Specific | 0.7994 | 0.7983 | 0.8138 | 0.8006 | 0.15% |

The One-Model-Fits-All approach has lower reported RMSE at every week position and improves modestly between weeks 1 and 4. Its average across all four positions is 0.7830, compared with 0.8030 for the Week-Specific design.

At country level:

- both designs improve from week 1 to week 4 in 8 of 15 countries;
- One-Model-Fits-All week 4 beats Week-Specific week 4 in 12 of 15 countries;
- the largest One-Model-Fits-All improvement is South Africa at 8.16%;
- Week-Specific changes range from a 19.11% improvement in South Africa to an 11.02% deterioration in Japan.

The thesis C-versus-D DM table is not treated as verified here because the current implementation does not align forecasts by `week_position`. Those p-values must be recomputed before reuse.

## Interpretability findings

After excluding lagged inflation from the SHAP ranking, the leading exogenous drivers reported by the thesis are:

1. oil;
2. the Finance search category;
3. exchange rate;
4. the Interest Rate search topic;
5. the Export category;
6. U.S. 10-year Treasury yield.

Group rankings change across regimes:

- **Pandemic shock:** Tech and Media, Real Estate, and Travel and Tourism rank relatively highly.
- **Inflation surge:** Macro and Finance ranks first, followed by Labour and Income and Consumer and Lifestyle.
- **Normalization:** Business and Industry ranks first; Macro and Finance and Labour and Income remain important.

These patterns describe how the fitted model used the signals. They are not causal effects.

## Submitted conclusion

The thesis concludes that high-frequency search and financial variables can add predictive information beyond persistent inflation benchmarks, with the strongest gains in volatile, nonlinear periods. It also finds that pooling countries and weekly information sets can help a data-hungry LSTM learn more stable representations.

The conclusion is conditional rather than universal: simpler models remain competitive in several countries and in more stable inflation periods.
