# Input data

Put your own APP.csv here or pass --data with its path. Raw source data and learned historical feature vectors are not distributed in this repository.

Required columns: Ticker, Date, Open, High, Low, Close, Adj Close, Volume.
Date format: YYYY-MM-DD. One row per observed trading session, APP only. Positive OHLC/adjusted prices, nonnegative volume, valid high/low ranges and unique dates are required. Extra columns are ignored. Keep full history for rolling features. The original experiment used 1,401 rows from 2021-04-15 through 2026-11-10.

The v2 fixed validation schedule needs at least 1,250 eligible labeled rows (approximately 1,280 total rows with the default rolling features and horizon). It is intentionally not a generic short-dataset auto-tuner. Baseline mode supports shorter histories.

Public market-data redistribution rights depend on your source. Add data only when you have permission.
