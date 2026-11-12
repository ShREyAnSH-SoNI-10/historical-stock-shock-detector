# Historical Stock Shock Detection

A historical pattern matching research prototype

Academic project report   |   12 November 2026

Source code: https://github.com/ShREyAnSH-SoNI-10/historical-stock-shock-detector

## Project overview

This project investigates whether repeating trading conditions can provide advance warning of a sharp stock decline. The prototype compares the current state of APP with similar historical states and measures how often a defined decline followed those matches. It produces a historical risk score, an alert condition, or an explicit statement that the evidence is insufficient.

The updated model improved several metrics relative to the original nearest-neighbor baseline. At the same 50% warning threshold, accuracy increased from 48.7% to 54.9%, precision from 29.1% to 43.5%, and recall from 13.6% to 16.9%. However, always predicting no shock achieved 57.1% accuracy. The project therefore demonstrates an interpretable, reproducible method, but does not establish reliable shock prediction.

## Problem statement

Given data available after today's close, identify whether historical conditions resembling today were followed by APP closing at least 10% lower within the next ten observed trading sessions. The warning is issued before the potential future decline, rather than when a decline has already crossed a threshold.

| Design choice | Implemented scope |
| --- | --- |
| Asset | APP stock only |
| Prediction horizon | Next 10 observed trading sessions |
| Shock definition | Adjusted close at least 10% below today's adjusted close |
| Method | Distance-weighted historical neighbors with support checks |
| Primary output | Historical frequency or insufficient evidence |

The broader motivation is market early warning. This implementation is a single-stock experiment and should not be presented as an S&P 500 model or an emergency-intervention predictor.

## Data and prediction target

The input contains 1,401 APP observations from 15 April 2021 through 10 November 2026. It includes the ticker, index membership label, date, open, high, low, close, adjusted close, volume, dividends and stock splits. The index label is metadata; it does not contain S&P 500 prices or establish historical membership on every date.

Initial checks found no missing values, duplicate ticker-date pairs, invalid high-low relationships or negative volume. These checks verify structure, not the accuracy of the provider's prices. Exchange-calendar completeness and data provenance were not independently verified.

### Defining the outcome

For every eligible forecast origin, the label is positive if any of the next ten adjusted closes is at most 90% of the origin's adjusted close. A day with a $100 adjusted close therefore has a positive label if a subsequent adjusted close reaches $90 or less within the next ten sessions. Intraday lows are not used.

The last ten rows cannot receive complete historical outcome labels. Rolling features also require earlier observations. All feature calculations use the current and earlier rows only. Forecasting therefore occurs after the current close; it is not a pre-open or intraday system.

### Features selected by validation

| Feature | Calculation and interpretation |
| --- | --- |
| Five-session return | Adjusted close divided by its value five sessions earlier, minus one |
| Twenty-session volatility | Standard deviation of daily adjusted-close returns over twenty sessions |
| Twenty-session drawdown | Current adjusted close divided by the maximum of the last twenty closes, minus one |
| Relative volume | Current volume divided by its twenty-session mean |
| Overnight gap | Current open divided by the previous unadjusted close, minus one |

The full candidate set also included twenty-session returns, ten-session volatility and daily range. Validation preferred the five-feature set, reducing repeated influence from related price and volatility measures. This is a selection result, not evidence that those features are universally optimal.

### Shock size and warning cutoff

The 10% decline defines the event we are trying to predict. A warning cutoff, such as 50%, is the historical-match score needed to flag a warning. Changing that cutoff changes sensitivity and false alarms; it does not change the event definition or improve the information in the data.

## Historical pattern matching method

The pipeline is: validate the CSV, calculate features, standardize using fitting data, find spaced historical neighbors, reject weak matches, weight surviving matches, and compare their outcome frequency with an alert cutoff.

### Standardization and distance

Each feature is centered by its training mean and divided by its training standard deviation. This prevents large numerical units, such as volume, from dominating similarity. Distance is the root mean square difference between the current and historical standardized feature vectors.

### Distinct historical matches

Neighbors are selected in order of increasing distance. Accepted historical forecast origins must be at least eleven observed sessions apart. This prevents their ten-session target windows from overlapping. Such spacing reduces duplicated evidence, but separate windows can still belong to the same economic selloff.

### Distance weights and frequency

A match receives an unnormalized weight equal to one divided by the square of its distance plus 0.1. We normalize the weights so they sum to one. The reported score is the sum of the normalized weights for matches with positive future-decline labels. Closer matches therefore contribute more than less similar matches.

An illustrative score of 40% means that positive outcomes account for 40% of the accepted historical match weight. It is not automatically a calibrated 40% chance of a future decline.

### Minimum evidence requirement

The similarity radius is estimated only from fitting data. For sampled training observations, the method finds the fifth separated neighbor after excluding nearby origins, then takes a candidate 90th or 95th percentile of those distances. Current matches beyond the chosen radius are rejected.

A forecast requires at least five accepted matches and at least three effective weighted matches. Effective match count is one divided by the sum of squared normalized weights. It detects cases where one match dominates the score. If either condition fails, the model returns insufficient historical similarity.

### Candidate configurations

Twenty-four configurations compare 10, 20 or 30 neighbors; all training history or the latest 504 eligible origins; eight or five features; and two support-radius quantiles. Earlier chronological validation selected ten neighbors, all fitting history, five features and the 95th-percentile support radius.

## Training and evaluation design

Time order is preserved throughout the experiment. Random row splitting is inappropriate because future information and overlapping outcome windows can contaminate an earlier forecast. Every split boundary purges origins whose ten-session outcomes could reach the next partition.

| Stage | Period | Purpose |
| --- | --- | --- |
| Configuration validation one | 12 Jan to 6 Jul 2023 | Compare candidate settings on forecasts from earlier-fitted models |
| Configuration validation two | 27 Dec 2023 to 18 Jun 2024 | Repeat selection under another historical period |
| Alert policy selection | 18 Dec 2024 to 9 Sep 2025 | Choose cutoffs using only earlier forecasts and outcomes |
| Final fitting history | 13 May 2021 to 9 Sep 2025 | Fit 1,086 eligible historical origins |
| Evaluation | 24 Sep 2025 to 27 Oct 2026 | Evaluate 275 later forecast origins |

Candidate selection minimizes validation Brier loss with an explicit fitting-frequency fallback for unsupported days. Candidates with at least 50% validation coverage are preferred when available. The fallback is used only for complete probability-loss accounting; it does not turn an abstention into a supported forecast.

### Alert operating settings

| Setting | Selected cutoff | Policy objective |
| --- | --- | --- |
| Precision | 67.5% | Favor precision using F0.5 |
| Balanced | 25.0% | Balance notification precision and cluster recall using F1 |
| Detection | 25.0% | Favor cluster recall using F2 |

The policy search also constrains false notifications and the fraction of above-threshold days. An enabled cutoff must have at least three notifications in the policy period. Evaluation applies a ten-session cooldown to avoid counting every repeated daily flag as a separate notification. Budgets constrain earlier policy data, not future outcomes.

### Fair comparisons

The main before-and-after comparison retains the original 50% cutoff, same target and same 275 origins. Additional comparisons retune the original model on the same policy window. This distinguishes improvements from the matching method from improvements obtained simply by lowering the cutoff.

The dataset was repeatedly inspected during project development. Its chronological evaluation is useful retrospective evidence, but it is no longer a fresh independent confirmation of model performance.

## Results and interpretation

| Metric at the same 50% cutoff | Original | Updated |
| --- | --- | --- |
| Alert-policy accuracy | 48.7% | 54.9% |
| Warning-day precision | 29.1% | 43.5% |
| Positive-forecast-day recall | 13.6% | 16.9% |
| Flagged forecast days | 55 | 46 |
| False flagged days | 39 | 26 |
| Supported forecast days | 275 | 271 |
| Brier loss with explicit prior fallback | 0.3152 | 0.3008 |

Precision asks how many flagged days were followed by the defined decline. Recall asks how many positive forecast days were flagged. Accuracy measures correct flag/no-flag decisions. A false flagged day is an origin with a warning but no qualifying decline during its next ten sessions.

At the 50% cutoff, the updated model produced 20 true positive days, 26 false positive days, 98 false negative days and 131 true negative days. Its four unsupported days are counted as no issued alert in policy accuracy, not as confident no-shock predictions. Covered-only accuracy is 54.2%.

The always-no-shock policy achieved 57.1% accuracy, and the constant training-frequency baseline achieved a Brier loss of 0.2678. Both comparisons remain unfavorable for the updated model. The modest before-and-after gains do not establish a reliable predictor.

### Notifications and decline clusters

With its selected 25% cutoff, balanced mode emitted 25 notifications, including 14 false alarms, and warned about all seven heuristic breach clusters. The original model retuned for detection also warned about all seven, with 13 false alarms from 25 notifications. Therefore seven out of seven clusters is not 100% prediction accuracy or proof of superior forecasting.

Clusters join target-dependent first-breach dates separated by no more than ten sessions. They are not independently annotated shocks. Day-level scores and cooldown-adjusted notification statistics must not be mixed.

### Latest recorded observation

For 10 November 2026, only two matches passed the similarity radius, below the required five. The model returned insufficient historical similarity and no numerical forecast. This means the evidence requirement was not met; it does not mean the risk was zero.

## Software implementation and demonstration

The GitHub repository contains a portable Python package, configuration files, synthetic unit tests, aggregate results and this report. It uses NumPy and pandas. Raw source CSVs, credentials and saved historical training vectors are excluded from publishing.

| Component | Role |
| --- | --- |
| shock_model/baseline.py | Input validation, features, forward labels and original model |
| shock_model/model.py | Spaced weighted matching, abstention, validation, evaluation and frozen scoring |
| tests/test_model.py | Ten synthetic checks for labels, spacing, weighting, rejection and cooldowns |
| results/APP_experiment.json | Recorded configuration, split dates, metrics and latest result |
| shock_model/finbert_wording.py | Optional wording classifier excluded from forecasting |

### Reproduce the experiment

```bash
python -m pip install -r requirements.txt
python -m shock_model.model --data data/APP.csv
python -m unittest discover -s tests -v
```

Training writes the frozen model, validation candidate table, test forecasts, latest forecast and similar-period records under artifacts/v2. The fixed v2 validation schedule needs at least 1,250 eligible labeled origins. The original 1,401-row input satisfies this requirement.

### Score updated data without retraining

```bash
python -m shock_model.model --data data/APP_updated.csv
  --predict artifacts/v2/model_v2.npz --out artifacts/latest
```

Enter the command on one line. The updated CSV must retain enough prior rows for rolling features. The scorer loads historical vectors and scaling from the saved file. It does not fetch market prices, schedule executions, deliver external notifications or execute trades.

### Role of FinBERT

An optional module can classify supplied text as positive, negative or neutral. Applying it to a generated forecast measures the wording of that forecast. It does not measure public sentiment or independently confirm risk. FinBERT is excluded from the model results and alert decisions. Its model weights have separate licensing terms that must be checked before reuse.

## Limitations and future work

The principal limitation is the amount of independent predictive information. APP prices and volume describe one stock. They do not directly identify earnings surprises, geopolitical developments, credit stress or unscheduled policy interventions. A similar price pattern need not have the same cause or future outcome.

- Overlapping labels reduce effective sample size; spaced neighbors are not necessarily independent events.
- The alert threshold changes sensitivity and false alarms, but cannot create predictive information.
- Historical support rejection improves honesty of the output, but can also abstain before a real shock.
- Regime changes and a small number of policy notifications make cutoffs unstable.
- Repeated development on the same evaluation history can overstate apparent gains.
- Input prices, adjustment conventions and exchange-session completeness need independent verification.

Future work should first freeze the present method and evaluate genuinely new observations. Later additions can include broad-market and sector returns, volatility and credit indicators, and independently timestamped news. For an S&P 500 project, the index itself must become the target. Each addition should be tested against simple baselines under the same alert budget.

## Presentation guide

Suggested opening: We built a historical-pattern early-warning prototype for APP. It compares today's trading conditions with separated historical periods and asks how often a 10% decline followed within ten sessions. We added distance weights and an insufficient-evidence option. Performance improved relative to the original model at the same threshold, but remained below a simple no-shock accuracy baseline. The contribution is an interpretable, reproducible experiment and a clear account of its limits.

### Questions to be ready to answer

Why use chronological testing? Because a real forecast cannot use future observations. Why separate neighbors? To reduce repeated evidence from overlapping ten-session outcomes. Why not claim a calibrated probability? Because historical match frequency has not demonstrated reliable probability calibration. Why can seven clusters be caught while the model remains weak? Because frequent warnings and cluster grouping can produce high cluster recall alongside many false alarms.

## References and project evidence

1. Project source and reproducible experiment: https://github.com/ShREyAnSH-SoNI-10/historical-stock-shock-detector

2. NumPy documentation: https://numpy.org/doc/
3. pandas documentation: https://pandas.pydata.org/docs/
4. FinBERT model card and terms: https://huggingface.co/ProsusAI/finbert
5. Original FinBERT paper: https://arxiv.org/abs/1908.10063

Experiment evidence is stored in results/APP_experiment.json. The optional FinBERT module is not part of the forecasting results.
