# Local data contract

This directory is a local workspace for restricted raw data and generated analytical files. Only this README should be committed to Git.

The root [`.gitignore`](../../.gitignore) excludes everything below `analysis/data/` except this file.

## Directory layout

~~~text
analysis/data/
├── README.md
├── raw/                         # privately acquired source files
│   ├── raw_monthly/             # monthly Google Trends exports
│   ├── raw_weekly/              # overlapping weekly Google Trends exports
│   ├── ZAF.csv                  # South African index export
│   └── russia_inflation.csv     # post-OECD Russian inflation supplement
├── temp/                        # reproducible intermediate files
│   ├── preprocessing/
│   │   └── d_detrend/
│   └── ...
└── final/                       # model-ready targets, predictors, and panels
~~~

`raw/`, `temp/`, and `final/` are generated or populated locally. Git tracks only the `.gitkeep` placeholder in each directory; all actual contents remain ignored.

## Required source inputs

### Google Trends exports

Place monthly exports in `raw/raw_monthly/` and weekly exports in `raw/raw_weekly/`.

Filename patterns:

~~~text
<series_id>_<country>_m_<period>.csv
<series_id>_<country>_w_<period>.csv
~~~

Examples:

~~~text
10001_US_m_04_25.csv
10001_US_w_04_09.csv
10001_US_w_08_12.csv
~~~

The weekly windows used by the thesis files are `04_09`, `08_12`, `11_16`, `15_20`, `19_24`, and `23_25`.

- `<series_id>` is a Google category ID or a repository topic ID.
- `<country>` is a selected two-letter country code.
- topic IDs map to feature names beginning with `t_`.
- the complete ID-to-feature mapping is `CAT_MAP` in [`1_splice.py`](../prog/prep/gt/1_splice.py).

The parser expects two metadata rows followed by a two-column table:

| Field | Monthly | Weekly |
|---|---|---|
| Period | `YYYY-MM` | `YYYY-MM-DD` |
| Value | Google Trends index from 0 to 100 | Google Trends index from 0 to 100 |

The displayed header can be localized because the parser skips the first two rows and assigns the internal names `Date` and `value`. It converts `<1` to `0.5`.

The study's selected country codes are:

~~~text
BR CA DE FR GB ID IN IT JP KR MX RU TR US ZA
~~~

The mapping file defines 237 candidate Google Trends variables: 183 categories and 54 topics. The thesis analysis files contain 236 because no raw export exists for `10006` / `t_cheap`.

### South African stock index

`raw/ZAF.csv` is the manual Investing.com input read by [`financial.py`](../prog/prep/financial/financial.py). It is required for the thesis-equivalent South African benchmark. If it is absent or empty, the current script falls back to Yahoo's EZA exchange-traded fund, which is a non-equivalent proxy and should be flagged in any rerun.

| Column | Type | Use |
|---|---|---|
| `Date` | parseable date | Observation date |
| `Price` | numeric or comma-formatted text | Closing index level |

Other exported columns are ignored.

### Russian inflation supplement

`raw/russia_inflation.csv` supplies the Russian observations that are unavailable from the OECD endpoint after early 2022. It is a private source input and must not be committed.

Required schema:

| Column | Type | Rule |
|---|---|---|
| `country` | string | Must be `RUS` |
| `date` | string | Unique calendar month in `YYYY-MM` format |
| `infl_yoy` | numeric | Non-missing year-on-year inflation rate, in percent |

[`inflation.py`](../prog/prep/financial/inflation.py) fails with a clear error when this file is absent or invalid. The merge fills months not supplied by OECD; it does not expose the observations in public source code.

### Inflation and other financial data

No local raw file is required for OECD inflation or the Yahoo Finance and Stooq series. Their scripts request data remotely:

- [`inflation.py`](../prog/prep/financial/inflation.py)
- [`financial.py`](../prog/prep/financial/financial.py)

A rebuild therefore requires internet access and remains dependent on current provider availability.

## Generated schemas

### Google Trends intermediates

Stages 1-5 are wide matrices. The first column is `Date`; every other column follows:

~~~text
<two-letter-country>_<feature>
~~~

For example:

~~~text
US_t_inflation
DE_real_estate
JP_jobs
~~~

| Output | Meaning |
|---|---|
| `temp/1_monthly_raw.csv` | Monthly normalization backbones |
| `temp/1_weekly_spliced_raw.csv` | Weekly windows rescaled and merged |
| `temp/2_monthly_breakadj.csv` | Monthly break-adjusted series |
| `temp/2_weekly_breakadj.csv` | Weekly break-adjusted series |
| `temp/3_monthly_clean.csv` | Selectively smoothed monthly series |
| `temp/3_weekly_clean.csv` | Selectively smoothed weekly series |
| `temp/4_monthly_detrended_final.csv` | Monthly country-level common trend removed |
| `temp/4_weekly_detrended_final.csv` | Weekly country-level common trend removed |
| `temp/preprocessing/d_detrend/<CC>_monthly_pc1_trend.csv` | Monthly country trend diagnostic |
| `temp/preprocessing/d_detrend/<CC>_weekly_pc1_trend.csv` | Weekly country trend diagnostic |
| `temp/5_monthly_logdiff.csv` | Monthly transformed Google Trends inputs |
| `temp/5_weekly_logdiff.csv` | Weekly transformed Google Trends inputs |

Topic columns (`<CC>_t_*`) receive a plain log transform. Category columns receive a 12-month or 52-week log difference.

### Inflation

`final/inflation.csv`:

| Column | Type | Meaning |
|---|---|---|
| `country` | string | ISO alpha-3 code |
| `infl_yoy` | float | Year-on-year CPI inflation, percent |
| `date` | string | Month in `YYYY-MM` format |

The panel builder maps these codes to two-letter panel codes and converts dates to month-end timestamps.

### Financial data

`temp/financials_weekly_raw.csv`:

~~~text
date,v_oil,v_copper,v_gold,v_vix,v_us10y,v_fxrate,v_stock,country
~~~

`final/financials_weekly_52w_logdiff.csv`:

~~~text
Date,country,v_oil,v_copper,v_gold,v_vix,v_us10y,v_fxrate,v_stock
~~~

`Date` is normally a Friday. `country` uses ISO alpha-3 codes. All seven predictor columns must be numeric.

The model sample expects complete usable coverage for all 15 selected countries. Remote-download success must be checked explicitly; an output file can have the correct columns while still omitting a country.

### Final panels

`final/B_panel.csv` and `final/D_panel.csv` use this column pattern:

~~~text
date
<gt_feature>_w1 ... <gt_feature>_w4
country
infl_yoy
v_oil_w1 ... v_stock_w4
infl_yoy_lag1
C_CA ... C_ZA
~~~

They contain 236 Google Trends variables at four within-month positions, seven financial variables at four positions, the target, one inflation lag, and 14 country dummies.

`final/C_panel.csv` has the same fields plus `week_position`. Its unique key is `date`, `country`, `week_position`.

`final/B_monthly_panel.csv` contains one monthly value per Google Trends feature, the final weekly financial observation associated with the month, the inflation target and lag, and the country dummies.

## Thesis-snapshot validation targets

These checks describe the model sample used in the thesis, not arbitrary future downloads.

| File | Rows | Columns | Date range |
|---|---:|---:|---|
| `B_panel.csv` | 3,735 | 990 | 2005-01-31 to 2025-09-30 |
| `D_panel.csv` | 3,735 | 990 | 2005-01-31 to 2025-09-30 |
| `C_panel.csv` | 14,940 | 991 | 2005-01-31 to 2025-09-30 |
| `B_monthly_panel.csv` | 3,735 | 261 | 2005-01-31 to 2025-09-30 |

Before running models, verify:

1. all 15 selected countries are present;
2. every B/D country-month key is unique;
3. every C country-month-week-position key is unique;
4. each country has 249 monthly observations;
5. `week_position` is exactly 1, 2, 3, or 4;
6. all seven financial variables have usable coverage by country and period;
7. missingness is reviewed before any model-side imputation;
8. no raw or generated data file is staged for Git.

The current panel builder groups the Friday-dated financial observations by their existing calendar month. The submitted thesis describes shifting these observations by two days to align them with Sunday-dated Google Trends data. Choose and document one convention before claiming an exact replication.

For months containing more than four weekly observations, the current builder averages observations four and five into `_w4`. Months with fewer than four Google Trends weeks are not added to the canonical panel.

## Generation order

~~~text
financial/financial.py
financial/inflation.py
gt/1_splice.py
gt/2_breakadj.py
gt/3_denoise.py
gt/4_detrend.py
gt/5_logdiff.py
gt/6_panel.py
~~~

Intermediate files are reproducible from the raw inputs and may be deleted after a successful run. Keep privately acquired raw files, provider metadata, retrieval timestamps, and checksums in a separate backed-up research archive.
