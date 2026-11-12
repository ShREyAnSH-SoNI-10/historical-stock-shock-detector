# Historical Stock Shock Detector

A basic research project asking: **When historical trading conditions resembled today, how often did a sharp decline follow?**

Implemented scope: **APP only**, not the S&P 500. A shock is a close at least **10% below the forecast-day adjusted close at any of the next 10 observed daily closes**. Forecasts use data through the current daily close.

This is an educational prototype. At the same 50% warning threshold, updated accuracy was **54.9%**, precision **43.5%**, and recall **16.9%** on 275 forecast days. Always predicting no shock achieved **57.1% accuracy**. Reliable predictive skill has **not** been established.

## Setup

Python 3.10 or newer; recorded run used Python 3.12, NumPy 2.5.4 and pandas 3.0.1.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Place your own APP.csv in data/ or pass its path. Raw source data is not distributed. See [schema](data/README.md).

## Train and test

```bash
python -m shock_model.baseline --data data/APP.csv --out artifacts/baseline
python -m shock_model.model --data data/APP.csv --out artifacts/v2
python -m unittest discover -s tests -v
```

Baseline is the original 30-neighbor model. Updated training is self-contained and reconstructs baseline comparisons; running baseline first is optional. The fixed v2 validation design needs at least 1,250 eligible labeled rows. A shorter input raises a clear error.

## Predict without retraining

```bash
python -m shock_model.model --data data/APP_updated.csv --predict artifacts/v2/model_v2.npz --out artifacts/latest
```

Retain enough earlier rows for rolling features. A supported score is a **weighted historical frequency**, not a calibrated probability. When support is insufficient, the output says **insufficient historical similarity**, not zero risk.

## Method

1. Calculate recent returns, volatility, drawdown, relative volume and gaps.
2. Standardize using fitting history only.
3. Find nearest historical origins at least 11 observed sessions apart.
4. Weight closer matches by `1 / (distance + 0.1)^2`.
5. Reject matches beyond a training-only radius; require five matches and three effective weighted matches.
6. Compare 10/20/30 neighbors, all/recent-504 history, full/reduced features and two support-radius quantiles on earlier validation periods.
7. Select warning thresholds using a separate pre-test policy window.
8. Evaluate later dates, counting abstentions explicitly.

Selected: 10 neighbors, all fitting history, five reduced features. A 50% **warning threshold** is a cutoff on the historical-match score, not a 50% decline. The shock target remains 10%. Validation-selected balanced/detection cutoffs were 25%, precision cutoff 67.5%.

## Recorded results

Same 50% warning threshold, same 275 forecast origins from 2025-09-24 through 2026-10-27:

| Metric | Original | Updated |
|---|---:|---:|
| Alert-policy accuracy | 48.7% | 54.9% |
| Warning-day precision | 29.1% | 43.5% |
| Positive-day recall | 13.6% | 16.9% |
| False flagged days | 39 | 26 |
| AUC | 0.395 | 0.441 on supported days |
| Supported forecast days | 275 | 271 |

Four updated test days were unsupported. Policy accuracy treats abstention as no issued alert, not as a confident no-shock prediction. Covered-only metrics are retained in [results](results/APP_experiment.json).

Balanced mode warned about all seven heuristic breach clusters, with 25 notifications and 14 false alarms. The original model retuned for detection also caught seven clusters, with 13 false alarms from 25 notifications. This is **not 100% prediction accuracy** or proof of superiority. Notification counts use a ten-session cooldown and differ from daily flags.

Latest recorded input, 2026-11-10: **insufficient historical similarity**, with two accepted matches, fewer than five required.

## Files

- `shock_model/baseline.py`: original model and data preparation.
- `shock_model/model.py`: updated training, validation and saved-model scoring.
- `shock_model/finbert_wording.py`: optional text classifier, excluded from forecasts.
- `tests/test_model.py`: synthetic unit tests with no private CSV dependency.
- `results/APP_experiment.json`: aggregate results, configuration and split dates.
- [Project explanation](docs/PROJECT.md): method, design, evaluation and limitations.
- `docs/PROJECT_REPORT.docx`: Word report for academic presentation.

## Optional FinBERT

```bash
python -m pip install -r requirements-optional.txt
python -m shock_model.finbert_wording --text statement.txt
```

This classifies supplied text only. Classifying the generated forecast does **not** measure public sentiment or independently confirm a prediction. FinBERT is not used in model accuracy or alert selection. Review its separate [model license](https://huggingface.co/ProsusAI/finbert).

## Limitations and publishing

Ten-day labels overlap. Spacing prevents overlapping neighbor target windows but does not prove independent economic events. Breach clusters are heuristic groups, not independently annotated crises. This dataset was repeatedly inspected during development; results are retrospective, not fresh independent validation. Data provenance and exchange-session completeness were not independently verified.

The repository does not collect live data, send external notifications, run a scheduler or place trades. Raw CSVs, credentials, virtual environments and learned vector files are ignored. Do not redistribute data without permission. No open-source license has been assigned; choose one intentionally if desired.
