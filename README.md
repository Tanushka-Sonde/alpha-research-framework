# Alpha Research & Backtesting Framework

A research pipeline that takes raw OHLCV market data → generates ML-based
short-horizon return signals → validates them with **purged walk-forward
cross-validation** (not a random split) → backtests them with realistic
transaction costs, slippage, and volatility-targeted position sizing →
reports risk-adjusted performance (Sharpe, Sortino, max drawdown, Calmar,
turnover) in an interactive dashboard.

The point of this repo isn't "a model that predicts stocks." It's a
demonstration that you understand **why most backtests lie** (lookahead
bias, survivorship-style leakage via overlapping labels, ignoring costs)
and how to build a pipeline that doesn't.

## Why this is built the way it is

| Design choice | Why |
|---|---|
| Purged walk-forward CV, not random K-fold | A label at time *t* is a forward return over `[t, t+H]`. Random splitting lets training rows "see" price information that overlaps a test window — the model trains on the future relative to some of what it's evaluated on. Sharpe from that kind of split is fiction. |
| Embargo period after each test fold | Return/vol serial correlation means information right after a test window can still leak backward into a later fold's training data if you don't buffer it out. |
| GBM (LightGBM) as the primary model | Tabular, low-dimensional, non-stationary financial features are exactly where gradient boosting outperforms deep nets in practice — it's what quant shops actually reach for first. |
| LSTM as a stretch option | Shows range on sequence modeling, and gives you a real "why A over B" tradeoff to discuss (LSTMs need more data, overfit small daily-bar datasets more easily). |
| Costs + slippage charged on every turnover event | "Did the sign match" isn't a strategy. A signal that flips a lot at 5-10bps round-trip cost can have its whole edge eaten by trading frictions — the report shows you this directly (`total_costs_paid`, `avg_daily_turnover`). |
| Volatility targeting in the backtest | This is how real books size positions — scale exposure to hit a target annualized vol rather than betting a fixed dollar amount regardless of regime. |
| Regime breakdown in the dashboard | A strategy's Sharpe over the whole sample can hide the fact that it only works in one volatility regime — the dashboard splits performance by high-vol vs low-vol periods explicitly. |

## What you'll actually see when you run it

Out-of-sample Sharpe on synthetic/short-history data will usually be
**low, sometimes barely positive** — that's expected and correct, not a
bug. A "95% accurate" backtest is almost always leaking information. Talk
about *why* the strategy probably wouldn't survive live (overfitting to
a regime, cost drag, turnover) — that self-awareness is the actual signal
in an interview, not the Sharpe number itself.

## Project structure

```
alpha-research-framework/
├── config.yaml                  # every knob lives here
├── src/
│   ├── data/
│   │   ├── loader.py             # yfinance OHLCV + local parquet cache + synthetic fallback
│   │   └── features.py           # point-in-time features + forward-return labels
│   ├── validation/
│   │   └── walk_forward.py       # purged walk-forward CV with embargo
│   ├── models/
│   │   ├── gbm_model.py          # LightGBM signal model (primary)
│   │   └── lstm_model.py         # PyTorch LSTM (stretch, optional dep)
│   ├── backtest/
│   │   └── engine.py             # signal -> positions -> P&L, costs, vol targeting
│   ├── reporting/
│   │   └── metrics.py            # Sharpe, Sortino, max DD, Calmar, turnover
│   └── pipeline.py               # orchestrates the whole thing end to end
├── dashboard/app.py               # Streamlit dashboard (reads artifacts/)
├── scripts/
│   ├── run_pipeline.py            # CLI entrypoint
│   └── download_data.py           # just fetch/cache data
├── tests/                          # pytest unit tests (incl. leakage checks)
├── Dockerfile / docker-compose.yml
└── requirements.txt / requirements-optional.txt (torch, for the LSTM)
```

## How to run it

### Option A — plain Python / VS Code (recommended for you)

```bash
# from the project root
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
# optional, only if you want the LSTM variant:
# pip install -r requirements-optional.txt

python scripts/run_pipeline.py
```

This will:
1. Download OHLCV for the tickers in `config.yaml` (cached to `data_cache/` so re-runs are instant)
2. Build features + forward-return labels per ticker
3. Run purged walk-forward CV, training a fresh model per fold
4. Stitch all out-of-sample fold predictions into one signal panel
5. Backtest the signal with costs/slippage/vol-targeting
6. Print a performance summary and write everything to `artifacts/`

Then open the dashboard:

```bash
streamlit run dashboard/app.py
```
→ http://localhost:8501

### Option B — Docker

```bash
docker compose up pipeline     # runs the pipeline once, writes artifacts/
docker compose up dashboard    # serves the dashboard at http://localhost:8501
```

Both containers mount `./data_cache`, `./artifacts`, and `./config.yaml` from
the host, so you can edit the config and re-run without rebuilding the image.

### Running tests

```bash
pytest tests/ -v
```

Includes explicit tests that the walk-forward splitter never lets a
training fold's label window overlap into its test fold, and that the
embargo zone is respected across folds — these are the tests you want to
be able to point to and say "I verified this doesn't leak."

## Configuring it

Everything is in `config.yaml` — universe of tickers, date range, feature
windows, label horizon, walk-forward fold sizes/embargo, model
hyperparameters, backtest cost assumptions and position sizing rules. No
need to touch code to run a different experiment; change the config and
re-run `scripts/run_pipeline.py`.

Key ones to know when you're asked about it in an interview:
- `label.horizon`: how many days forward you're predicting returns over
- `validation.embargo_days`: the leakage buffer between folds
- `backtest.transaction_cost_bps` / `slippage_bps`: your cost assumptions
- `backtest.vol_target_annual`: what annualized vol the book is sized to

## A note on the synthetic data fallback

`src/data/loader.py` tries `yfinance` first and caches results locally. If
network access to Yahoo Finance isn't available (e.g. a sandboxed dev
environment, CI, or a flaky connection), it transparently falls back to a
synthetic multi-regime OHLCV generator so the **entire pipeline still runs
end to end** — useful for testing the code itself, but you should always
train/evaluate on real data before drawing any actual research conclusions.
You'll see a `WARNING: Falling back to SYNTHETIC data` log line whenever
this happens; on a normal internet connection this won't trigger at all.

## Talking points for the interview

- **Why walk-forward, not random split**: explained above and in
  `src/validation/walk_forward.py`'s module docstring — be ready to draw
  the picture of a label window overlapping a test fold.
- **Why the Sharpe is modest**: real, cost-aware, leakage-free backtests
  on short-history single-signal strategies usually *are* modest. Point to
  `total_costs_paid` and `avg_daily_turnover` in the report — a chunk of
  gross edge is turnover cost, which is exactly the kind of thing that
  kills strategies in production.
- **Why GBM over deep nets**: data efficiency on tabular, non-stationary
  features; LSTM is included specifically to show you understand the
  tradeoff rather than defaulting to "deep learning is better."
- **What would break this in production**: regime shift (see the
  dashboard's high-vol vs low-vol Sharpe split — most single-signal
  strategies aren't regime-stable), capacity/market impact not modeled
  here (this uses a fixed bps slippage, not a market-impact model), and
  survivorship bias in the ticker universe itself if you don't
  deliberately include delisted names.
