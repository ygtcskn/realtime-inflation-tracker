# Limitations and production considerations

This repository preserves the analytical pipeline developed for a completed master's thesis. It is a research artifact, not a live forecast service.

The values in [results.md](results.md) are thesis-reported. A later review of the surviving code identified items that should be resolved and followed by a clean rerun before the outputs are described as independently reproduced or production-ready.

## Limitations stated in the thesis

### Google Trends normalization

Google Trends supplies an index from 0 to 100, not raw search volumes. Each query and window is normalized independently, so separately downloaded weekly chunks are not directly comparable.

The pipeline rescales overlapping windows against a monthly backbone. This corrects broad level differences but can leave smaller shape distortions.

### Sampling variation

Queries use changing samples of searches. Repeating the same request can therefore produce different values, especially for low-volume topics or smaller regions. The thesis did not average repeated draws because of the acquisition cost, so an exact rebuild may differ even with identical definitions.

### Limited history and changing providers

Usable high-frequency search histories begin only in the mid-2000s, limiting the number of inflation cycles. API behavior, source tickers, and historical values can also change.

### Country coverage

The panel includes 15 G20 economies. China and Saudi Arabia were excluded because of Google Trends coverage concerns, Argentina because of concerns about its official inflation series, and Australia because the target available for the study design was quarterly.

The findings should not be generalized automatically to omitted economies.

## Research-design limitations

### Crisis-heavy test period

January 2020-September 2025 contains the pandemic, reopening, energy shock, and global inflation surge. This provides a demanding stress test but is not representative of every macroeconomic regime. The LSTM advantage is concentrated in 2021-2023 and weakens during normalization.

### Sensitivity to Turkey

Turkey has a much larger inflation scale and volatility than most countries. Although the headline score equally averages country RMSEs, Turkey materially affects model comparisons, especially XGBoost. Results with and without Turkey should be read together.

### Pseudo-real-time rather than vintage-real-time

The experiment advances through time and withholds the current inflation target, but it does not reconstruct every historical source vintage or publication timestamp. A true real-time study needs versioned data vintages and an explicit release calendar.

### Interpretation is not causation

SHAP and permutation importance explain how the fitted model uses inputs. They do not establish that searches or market movements cause inflation. Correlated features can also exchange importance.

## Current implementation audit

| Item | Consequence | Production-oriented remediation |
|---|---|---|
| Google Trends detrending uses full-history two-sided HP filtering, scaling, PCA, and rescaling | Future observations can affect earlier features despite training-only scaling inside the model scripts | Refit adaptive transformations inside every expanding window or use explicitly one-sided features |
| Break adjustment uses post-break averages, including the January 2022 test-period break | An early post-break feature can depend on later observations | Estimate corrections only from information available at each origin and test truncation invariance |
| Financial Fridays remain unshifted in `6_panel.py`, while the thesis describes a +2-day Sunday alignment | Thesis and current snapshot use different stated alignment conventions | Choose one time-zone/calendar convention, encode it once, and add boundary-date tests |
| In months with more than four weekly observations, all observations after week 3 are averaged into `_w4` | A feature labelled week 4 can include fifth-week information | Use timestamp cutoffs; retain week 5 separately or exclude it from the week-4 estimate |
| C/D DM joins omit `week_position` | Four rows for the same country-month can form a many-to-many merge, invalidating C/D p-values | Join on date, country, horizon, and week position; assert one-to-one alignment; rerun the table |
| LSTM sequence loops use `range(len(group) - sequence_length)` | The final eligible sequence is omitted; a September 2025 panel ends with an August prediction | Include the endpoint and test first/last prediction dates |
| The inspected financial artifact has no Brazil or Turkey rows and only partial Russian history | Missing financial features can be converted to zero and treated as observations | Fail fast on coverage gaps; add explicit missingness handling and training-only imputation |
| Download helpers broadly catch failures and may return empty results | Provider failures can pass downstream without an actionable error | Log source/ticker/range/exception, retry transient failures, and stop on missing required coverage |
| Financial values are forward-filled without a maximum gap; missing South African benchmark data fall back to the EZA ETF | A long outage can create stale features, and a non-equivalent proxy can silently change the target series definition | Bound fill lengths, emit missingness indicators, and require the private benchmark input for thesis-equivalent runs |

Several items can affect forecast errors. Existing numerical outputs should therefore keep the **thesis-reported** label until the pipeline is corrected and rerun from source inputs.

## Portfolio and production boundary

The code-only release protects third-party data rights and makes the source reviewable, but a clone cannot regenerate thesis tables without reacquiring data.

Operational use would additionally require:

- schema and coverage checks at every source boundary;
- versioned raw snapshots and feature artifacts;
- retry, alerting, and provider-fallback policies;
- unit tests for time alignment and sequence boundaries;
- data-drift and forecast-error monitoring;
- a deterministic, minimal environment lock;
- model and data lineage for every published estimate.

These form a production-hardening roadmap; they are not capabilities claimed by the submitted thesis.
