# alpha-research-framework

## Project 1 (Finance/Trading): Alpha Research & Backtesting Framework

This is the actual "bhaiya" project — it maps directly to the **Quant Researcher** track from those notes (ML + probability + market analysis), which fits your ML background way better than the C++ quant dev track.

**What you build:** a research pipeline that takes raw market data → generates ML-based trading signals ("alpha") → backtests them honestly → reports risk-adjusted performance.

**Why "honestly" matters (this is the whole point):** 90% of student trading projects get destroyed in interviews because of lookahead bias or survivorship bias. If you show you *understand and prevented* these, you instantly look more serious than someone with a "95% accurate stock predictor."

Structure:
1. **Data layer** — pull OHLCV data (yfinance/Alpha Vantage free tier is fine), build proper point-in-time features (rolling returns, volatility, RSI/MACD-style indicators, order flow imbalance if you can get L2 data)
2. **Signal generation** — train a model (start with gradient boosting, it's what quant shops actually use more than deep nets for tabular alpha; add an LSTM/transformer variant as a stretch to show range) to predict short-horizon returns
3. **Walk-forward validation, not random train/test split** — this is the single biggest "you get it" signal. Explain in your README *why* random splitting leaks future information into training
4. **Backtest engine** — simulate actual trading with transaction costs, slippage, and position sizing constraints (not just "did the sign match")
5. **Risk reporting** — Sharpe ratio, max drawdown, turnover — the metrics a PM actually cares about, not accuracy
6. **Dashboard** — a simple Streamlit app showing signal performance over time, feature importance, regime behavior (does it work in high vol vs low vol periods?)

What to say in the interview: talk about *why* your model probably wouldn't survive in production (overfitting to a regime, transaction costs eating the edge) — self-awareness about a strategy's limits reads as far more credible than claiming it's profitable. Quant interviewers specifically probe for this.
