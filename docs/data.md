# Data

This is a code-only research repository. It documents and implements the data pipeline developed for the master's thesis *Tracking Inflation in the G20 at High Frequency and in Real Time: A Machine Learning Approach*, but it does not redistribute the underlying third-party datasets.

A fresh clone therefore shows how the data are acquired, transformed, aligned, and modelled, but it cannot reproduce the thesis results until the required source data have been obtained from their original providers.

## Study scope

The analysis combines a monthly inflation target with weekly Google Trends and financial-market predictors.

| Data group | Measures | Original frequency | Model frequency | Thesis period |
|---|---|---:|---:|---:|
| Inflation | Consumer-price inflation, year-on-year percentage change | Monthly | Monthly | 2004-2025 |
| Google Trends categories | 183 search categories per country | Monthly and weekly exports | Weekly | 2004-2025 |
| Google Trends topics | 54 topic definitions per country | Monthly and weekly exports | Weekly | 2004-2025 |
| Country financial indicators | Bilateral exchange rate and benchmark stock index | Daily | Weekly | 2004-2025 |
| Global financial indicators | Oil, copper, gold, VIX, and U.S. 10-year Treasury yield | Daily | Weekly | 2004-2025 |

The final modelling panel runs from January 2005 through September 2025: 249 months for 15 countries, or 3,735 country-month observations. The out-of-sample evaluation period is January 2020 through September 2025.

The analyzed input contains 236 Google Trends variables per country. The catalogue in [`1_splice.py`](../analysis/prog/prep/gt/1_splice.py) defines 183 categories and 54 topics, but no raw export is present for topic ID `10006` (`t_cheap`).

## Country sample

| Country | Panel code | Source code |
|---|---:|---:|
| Brazil | `BR` | `BRA` |
| Canada | `CA` | `CAN` |
| France | `FR` | `FRA` |
| Germany | `DE` | `DEU` |
| India | `IN` | `IND` |
| Indonesia | `ID` | `IDN` |
| Italy | `IT` | `ITA` |
| Japan | `JP` | `JPN` |
| Mexico | `MX` | `MEX` |
| Russia | `RU` | `RUS` |
| South Africa | `ZA` | `ZAF` |
| South Korea | `KR` | `KOR` |
| Turkey | `TR` | `TUR` |
| United Kingdom | `GB` | `GBR` |
| United States | `US` | `USA` |

Four G20 countries were excluded in the submitted study because the required inputs were not considered sufficiently comparable: China and Saudi Arabia because of Google Trends coverage concerns, Argentina because of concerns about the official inflation series, and Australia because inflation data were available quarterly rather than monthly for the study design.

## Sources and variables

### Inflation

Monthly year-on-year CPI inflation is requested through the OECD SDMX API. The downloader combines OECD price datasets based on the earlier and COICOP 2018 classifications.

The resulting schema is:

| Column | Meaning |
|---|---|
| `country` | ISO alpha-3 country code |
| `date` | Calendar month in `YYYY-MM` format |
| `infl_yoy` | Year-on-year CPI inflation rate, in percent |

OECD coverage for Russia ends in early 2022 in the thesis data. Later observations, attributed in the thesis to Rosstat and accessed through TradingView, are read from the private `analysis/data/raw/russia_inflation.csv` supplement. The values are deliberately not embedded in the public source. [`inflation.py`](../analysis/prog/prep/financial/inflation.py) validates that file before merging it. The script also contains logic to interpolate a missing U.S. October 2025 value; this lies after the September 2025 modelling cutoff.

### Google Trends

Google Trends reports relative search intensity on a normalized 0-100 scale, not raw query counts. The pipeline uses topics and categories rather than language-specific keywords to improve comparability across countries.

Examples include:

- categories such as Finance, Real Estate, Jobs, Shopping, Energy and Utilities, and Travel;
- topics such as Inflation, Cost of Living, Consumer Price Index, Interest Rate, Exchange Rate, Rent, Electricity, Wages, Debt, Recession, and Financial Crisis.

Monthly exports provide a long-run normalization backbone. Weekly data are downloaded in overlapping windows because Google Trends does not expose a full long-history weekly series in one export. The pipeline interpolates the monthly backbone to weekly frequency, rescales each weekly chunk over its overlap, and averages duplicate dates.

The remaining preprocessing stages are:

1. adjust documented Google Trends breaks;
2. selectively smooth noisy series with cubic smoothing splines;
3. remove the common country-level trend using HP-filtered trends and the first principal component;
4. apply 12-month or 52-week log differences to category variables;
5. apply a plain log transform to topic variables.

Raw values reported by Google as `<1` are represented as `0.5` by the parser.

### Financial indicators

| Column | Scope | Measure |
|---|---|---|
| `v_fxrate` | Country-specific | Bilateral exchange rate against the U.S. dollar; DXY for the United States |
| `v_stock` | Country-specific | National benchmark equity index |
| `v_oil` | Global | Crude-oil futures price |
| `v_copper` | Global | Copper futures price |
| `v_gold` | Global | Gold futures price |
| `v_vix` | Global | CBOE Volatility Index |
| `v_us10y` | Global | U.S. 10-year Treasury yield |

Most series are downloaded from Yahoo Finance. The pipeline uses Stooq overrides for the Turkish exchange rate, Russian stock index, and Brazilian exchange rate. Thesis-equivalent construction requires a private Investing.com `ZAF.csv` for the South African benchmark index because the Yahoo series was considered unreliable. If that file is missing or empty, the current code falls back to Yahoo's EZA exchange-traded fund; that proxy is not the same benchmark and should not be used for an exact replication.

Daily observations are sampled at the final observation of each Friday, then missing values are forward-filled within country without a maximum gap in the archived implementation. This was intended to bridge market holidays, but an extended provider gap can therefore become a stale value; a new run should impose and validate a bounded fill policy. Each series is then transformed to a 52-week log difference. The downloader begins in December 2003 so that transformed values are available when the panel starts in January 2005.

Remote ticker availability can change. A rebuild should verify that all seven financial variables cover all 15 study countries before the panels are used for modelling. The locally generated file inspected while preparing these docs contains only 13 countries—Brazil and Turkey are absent—so the current local acquisition output does not pass that coverage check.

## Data lineage

~~~text
Google Trends monthly exports ─┐
                               ├─ splice weekly windows
Google Trends weekly exports ──┘
        -> break adjustment
        -> selective spline smoothing
        -> country-level HP/PCA detrending
        -> log or year-on-year log-difference transform ─┐
                                                         │
Daily financial series -> Friday sampling                │
        -> forward fill -> 52-week log differences ──────┤
                                                         ├─ monthly panel builder
Monthly CPI inflation --------------------------------───┘
        -> U-MIDAS, weekly-tracking, and monthly comparison panels
~~~

The current repository implementation keeps financial observations on their Friday dates and groups them by calendar month. The submitted thesis text describes an additional two-day shift to align Fridays with the Sunday dates used by Google Trends. Reproduction work should state explicitly which convention is used.

When a month has more than four weekly observations, the panel builder keeps the first three and averages the remaining observations into week four. Months with fewer than four Google Trends observations are skipped.

## Analytical panels

| File | Purpose | Key | Thesis-snapshot shape |
|---|---|---|---:|
| `B_panel.csv` | Full four-week U-MIDAS panel | `date`, `country` | 3,735 x 990 |
| `C_panel.csv` | One-model-fits-all weekly tracker | `date`, `country`, `week_position` | 14,940 x 991 |
| `D_panel.csv` | Canonical input for four week-specific models | `date`, `country` | 3,735 x 990 |
| `B_monthly_panel.csv` | Monthly-frequency comparison panel | `date`, `country` | 3,735 x 261 |

`B_panel.csv` and `D_panel.csv` contain the same canonical data. They remain separate so the existing model entry points can use distinct filenames.

The weekly B/D schema contains:

- 944 Google Trends columns: 236 variables multiplied by four within-month positions;
- 28 financial columns: seven variables multiplied by four positions;
- `infl_yoy` and `infl_yoy_lag1`;
- `country` and month-end `date`;
- 14 one-hot country columns, with Brazil as the omitted reference country.

`C_panel.csv` adds `week_position` with values 1-4 and repeats each country-month four times. Later, not-yet-observed weekly values are filled with the last value available at that week position so that one model can use a fixed input shape.

## Why the data are not in Git

The data are intentionally excluded because:

- the repository is intended to demonstrate pipeline and modelling work, not redistribute provider datasets;
- Google Trends, Yahoo Finance, Stooq, Investing.com, TradingView, and OECD data remain subject to their providers' current terms;
- Google Trends values are normalized and can vary with the query window and retrieval date;
- financial ticker histories and remote availability can change;
- the raw and generated files are large and would obscure the source code in the repository.

The repository is currently all rights reserved. Its [copyright notice](../LICENSE) covers repository-owned code and documentation only; it does not grant rights to any third-party data, brand, or index.

To reproduce the analysis, obtain the inputs directly from their providers, review the applicable terms, and retain a private provenance manifest containing the source, query or ticker, geography, retrieval timestamp, requested period, filename, and checksum.
