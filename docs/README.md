# Documentation

This directory explains the research and code without requiring access to the private datasets or the full thesis. It separates three things that are easy to conflate:

1. the methodology described in the submitted thesis;
2. the behavior of the current archival code snapshot; and
3. the work needed to turn the prototype into a production system.

## Suggested reading paths

For a brief overview, read the [root README](../README.md), then [architecture](architecture.md) and [results](results.md).

For a technical review, continue with [pipeline](pipeline.md), [methodology](methodology.md), [models](models.md), and [evaluation](evaluation.md).

For reproduction or reuse, read [data](data.md), [reproducibility](reproducibility.md), [limitations](limitations.md), and the [analysis guide](../analysis/README.md).

## Document map

| Document | Purpose |
|---|---|
| [Architecture](architecture.md) | System boundaries, components, and data flow |
| [Pipeline](pipeline.md) | Ordered transformation and modeling stages mapped to scripts |
| [Data](data.md) | Sources, country coverage, schemas, private-data policy, and validation expectations |
| [Methodology](methodology.md) | Research design and mixed-frequency formulations |
| [Models](models.md) | Benchmarks, LASSO, XGBoost, and LSTM designs |
| [Evaluation](evaluation.md) | Expanding-window protocol, metrics, tests, and interpretation |
| [Results](results.md) | Main numerical findings reported in the submitted thesis |
| [Limitations](limitations.md) | Google Trends constraints, archival-code caveats, and productionization work |
| [Reproducibility](reproducibility.md) | Reproduction boundary, environment, and current snapshot status |
| [Publishing](publishing.md) | Checks for publishing code and documentation without data |

## Evidence convention

The words **thesis-reported** mean that a value or conclusion was transcribed from the submitted paper. **Current code** refers to behavior visible in this repository snapshot. Unless explicitly stated, the public documentation does not claim that the thesis tables were independently regenerated after publication.
