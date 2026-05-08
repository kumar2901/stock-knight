# stock-knight

NSE stock **ranking** system that outputs **Top 10/20/30** candidates for multiple horizons (1D, 7D, 1M, 6M, 12M), with walk-forward training and cost-aware backtests.

## Strategy used (how the algorithm “makes a list”)

This project is intentionally a **ranking model**, not a “predict exact price” model.

- **Daily workflow (EOD)**:
  - Build the **universe** (Top 100 snapshot; currently via a liquidity rule).
  - Compute **features** from market data (no future leakage).
  - For each horizon (1D, 7D, 1M, 6M, 12M), score each stock with a **horizon-specific model**.
  - Sort scores to produce **Top 10/20/30 longs** (and optional bottom lists for short research).

- **Why ranking?**
  - In practice, you act on a **relative bet**: “these names should outperform peers”.
  - It aligns better with evaluation metrics like **IC (rank correlation)** and **Top‑k spread**.

- **Training objective (current implementation)**:
  - Predict **forward returns** at each horizon using supervised ML.
  - Baseline model: **Ridge regression** on cross‑sectionally normalized features.
  - Upgrade path already scaffolded: **LightGBM** and learning-to-rank.

- **Backtest philosophy**
  - Focus on **net-of-cost** results by modeling turnover and per‑side costs.
  - Long/short execution in India is assumed via **futures/hedges** (full SSF/index hedge logic is a planned upgrade; current backtest is MVP).

## Data points analyzed (current)

## Universe construction (how “Top 100” is chosen)

Today the code builds the daily universe using a **liquidity-based rule** (because NIFTY100 membership-by-date is not wired yet).

- **Data points used**
  - **close price** (minimum price filter)
  - **volume**
  - **traded value** = `close × volume`

- **Rule (daily snapshot)**
  - Look back **60 trading days** (configurable)
  - Compute **median(traded value)** per stock over the lookback window
  - Filter:
    - `close >= min_price` (default 20)
    - `median(traded_value) >= min_traded_value_inr` (default 2 crore)
  - Rank by median traded value, take **top 120**, then keep **top 100**

This keeps the universe **tradable** and stable enough for EOD strategies.

## Coverage of your data categories

You asked whether we use:
1) Price (OHLC) 2) Volume 3) Technical indicators 4) Chart patterns 5) Fundamentals

Current status:
- **1) Price data (OHLC)**: **Yes (ingested)**; **partially used** (close-based returns + MA distances). Candlestick/ATR/gap features are not added yet.
- **2) Volume data**: **Yes** (used for traded value features + universe construction).
- **3) Technical indicators**: **Partially** (momentum/volatility/MA). RSI/MACD/Bollinger/ADX/ATR/OBV are not added yet.
- **4) Chart patterns**: **Not yet** (pattern recognition layer is not implemented).
- **5) Fundamental data**: **Not yet** (planned for 6M/12M horizons; requires a fundamentals source + reporting-lag handling).

### Raw inputs (EOD)
From OHLCV candles per symbol:
- **open, high, low, close**
- **volume**

Derived:
- **traded_value = close × volume**

### Features currently computed
All features are computed using only data available up to date \(t\), then normalized cross‑sectionally per day (z-scores).

- **Returns / momentum**
  - 1D return
  - trailing momentum: 5D, 21D, 63D, 126D, 252D

- **Volatility**
  - realized volatility (std of 1D returns): 10D, 21D, 63D

- **Trend**
  - moving average distance: 20D, 50D, 200D

- **Liquidity**
  - median traded value: 21D, 63D

### Labels (targets) used for training
Forward returns computed from close prices:
- **1D** (\(~1\) trading day)
- **7D** (\(~5\) trading days)
- **1M** (\(~21\) trading days)
- **6M** (\(~126\) trading days)
- **12M** (\(~252\) trading days)

## What’s planned next (data + strategy upgrades)
- **Corporate actions adjustment**: splits/bonus/dividends factors per symbol/date (scaffold exists in `src/stock_knight/data/corporate_actions.py`).
- **Benchmark-relative targets**: excess/residual returns vs NIFTY (to reduce “market direction” leakage into picks).
- **Sector/beta neutralization**: avoid accidental sector concentration.
- **Long/short execution realism**: SSF liquidity filtering + index futures hedge sizing.
- **Learning-to-rank objective**: better Top‑k precision than plain regression.

## Quickstart (dev)

## Run locally (step-by-step)

### Prerequisites
- **Python**: use `python3` (recommended 3.11+).
- **macOS/Linux**: commands below assume a POSIX shell.

### 1) Create venv + install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Verify it runs:

```bash
python -m stock_knight.cli --help
```

### 2) Run the full local pipeline (using mock data)

Ingest EOD OHLCV and build a universe snapshot:

```bash
python -m stock_knight.cli ingest --broker mock --start 2019-01-01 --end 2025-12-31
```

Build features + labels:

```bash
python -m stock_knight.cli features-labels --ohlcv-parquet data/ohlcv/ohlcv_2019-01-01_2025-12-31.parquet
```

Optional — enrich with **Yahoo Finance** snapshot ratios (TTM P/E, P/B, dividend yield, ROE, debt/equity, and balance-sheet–derived current ratio / D/E). Same values are repeated on every historical row per symbol (good for live ranking; less ideal for strict historical backtests). **Retrain** models after adding this, so `*_z` fundamental columns are in `feature_cols`:

```bash
python -m stock_knight.cli features-labels \
  --ohlcv-parquet data/ohlcv/ohlcv_2019-01-01_2025-12-31.parquet \
  --yfinance-fundamentals \
  --yfinance-delay 0.2
```

Use `data/sectors.csv` (via `--sectors-csv`) so **sector-relative** spreads (`yf_pe_vs_sector_z`, `yf_pb_vs_sector_z`, `yf_div_yield_vs_sector_z`) are populated.

Train models (baseline ridge):

```bash
python -m stock_knight.cli train \
  --features-parquet data/features/features_ohlcv_2019-01-01_2025-12-31.parquet \
  --labels-parquet data/labels/labels_ohlcv_2019-01-01_2025-12-31.parquet \
  --model-kind ridge
```

Generate the Top‑K list (example: 1D horizon for a date):

```bash
python -m stock_knight.cli rank \
  --features-parquet data/features/features_ohlcv_2019-01-01_2025-12-31.parquet \
  --model-path artifacts/models/ridge_1d.joblib \
  --asof 2025-12-30 --horizon 1d \
  --out reports/ranks_1d_2025-12-30.json
```

To attach **Yahoo fundamentals** to the rank JSON / Stock Detail **without** rebuilding the features parquet, add `--yfinance-fundamentals` (and optionally `--sectors-csv data/sectors.csv` for sector-relative columns). Expect ~0.2s per symbol.

Optional: run the MVP cost-aware backtest (1D PnL accounting):

```bash
python -m stock_knight.cli backtest \
  --features-parquet data/features/features_ohlcv_2019-01-01_2025-12-31.parquet \
  --labels-parquet data/labels/labels_ohlcv_2019-01-01_2025-12-31.parquet \
  --model-path artifacts/models/ridge_1d.joblib \
  --horizon 1d --k 20
```

Optional: compute a drift report (rolling IC summary):

```bash
python -m stock_knight.cli drift \
  --features-parquet data/features/features_ohlcv_2019-01-01_2025-12-31.parquet \
  --labels-parquet data/labels/labels_ohlcv_2019-01-01_2025-12-31.parquet \
  --model-path artifacts/models/ridge_1d.joblib \
  --horizon 1d --window-days 252 \
  --out reports/drift_1d.json
```

### 3) Running with real broker data (next step)
Install broker + UI dependencies:

```bash
pip install -e '.[brokers,ui]'
```

Then select a broker via `--broker`:
- `mock` (synthetic dev data)
- `yahoo` (Yahoo Finance via `yfinance`; provide symbols like `RELIANCE.NS`)
- `zerodha` (Kite Connect; provide NSE tradingsymbols like `RELIANCE`)
- `upstox` (provide instrument_keys; see note below)
- `angelone` (provide symboltokens; see note below)

Because most real broker APIs require **instrument identifiers** (tokens/keys), the recommended workflow is to pass a symbol list file:

```bash
python -m stock_knight.cli ingest --broker yahoo --symbols-file symbols.txt --start 2023-01-01 --end 2025-12-31
```

#### Credentials (environment variables)
- **Zerodha**: `ZERODHA_API_KEY`, `ZERODHA_ACCESS_TOKEN`
- **Upstox**: `UPSTOX_ACCESS_TOKEN`
- **Angel One**: `ANGELONE_API_KEY`, `ANGELONE_CLIENT_CODE`, `ANGELONE_PASSWORD`, `ANGELONE_TOTP`

#### Credentials (external file, recommended)
1. Copy the template:

```bash
cp credentials.example.yaml credentials.yaml
```

2. Fill in your keys in `credentials.yaml` (this file is **gitignored**).

3. (Zerodha) Generate a fresh access token (interactive login, once per day).

Recommended (works with any redirect URL): manual mode (paste `request_token`):

```bash
python -m stock_knight.auth.zerodha_login --creds-file credentials.yaml --mode manual
```

Alternative: server mode (requires setting Zerodha app redirect URL to `http://127.0.0.1:8787/`):

```bash
python -m stock_knight.auth.zerodha_login --creds-file credentials.yaml --mode server --host 127.0.0.1 --port 8787
```

3. Run ingest with:

```bash
python -m stock_knight.cli ingest --broker zerodha --creds-file credentials.yaml --symbols-file symbols.txt --start 2023-01-01 --end 2025-12-31
```

#### Streamlit UI

```bash
streamlit run streamlit_app.py
```

Create a virtualenv and install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run an end-to-end *ingestion + universe snapshot* using synthetic data:

```bash
python -m stock_knight.cli ingest --broker mock --start 2024-01-01 --end 2024-06-30
```

This writes data into `data/` (parquet + json).

Build features + labels:

```bash
python -m stock_knight.cli features-labels --ohlcv-parquet data/ohlcv/ohlcv_2024-01-01_2024-06-30.parquet
```

Train models (baseline ridge):

```bash
python -m stock_knight.cli train \
  --features-parquet data/features/features_ohlcv_2024-01-01_2024-06-30.parquet \
  --labels-parquet data/labels/labels_ohlcv_2024-01-01_2024-06-30.parquet \
  --model-kind ridge
```

Generate Top‑K lists for a date:

```bash
python -m stock_knight.cli rank \
  --features-parquet data/features/features_ohlcv_2024-01-01_2024-06-30.parquet \
  --model-path artifacts/models/ridge_1d.joblib \
  --asof 2024-06-28 --horizon 1d \
  --out reports/ranks_1d_2024-06-28.json
```