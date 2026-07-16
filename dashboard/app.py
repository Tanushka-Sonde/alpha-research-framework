"""
Streamlit dashboard for the alpha research framework.

Reads artifacts written by `scripts/run_pipeline.py` (default: ./artifacts).
Run the pipeline first, then:
    streamlit run dashboard/app.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.reporting import metrics  # noqa: E402

st.set_page_config(page_title="Alpha Research Dashboard", layout="wide")

ARTIFACT_DIR = Path("artifacts")


@st.cache_data
def load_artifacts(artifact_dir: Path):
    def _p(name):
        path = artifact_dir / name
        return pd.read_parquet(path) if path.exists() else None

    data = {
        "equity_curve": _p("equity_curve.parquet"),
        "daily_returns": _p("daily_returns.parquet"),
        "weights_history": _p("weights_history.parquet"),
        "turnover": _p("turnover.parquet"),
        "costs": _p("costs.parquet"),
        "signals_panel": _p("signals_panel.parquet"),
        "returns_panel": _p("returns_panel.parquet"),
        "feature_importance": _p("feature_importance.parquet"),
    }
    for name, key in [("report.json", "report"), ("fold_metrics.json", "fold_metrics"),
                       ("regime_report.json", "regime_report")]:
        path = artifact_dir / name
        data[key] = json.loads(path.read_text()) if path.exists() else None
    return data


st.title("📈 Alpha Research & Backtesting Dashboard")

if not ARTIFACT_DIR.exists() or not (ARTIFACT_DIR / "report.json").exists():
    st.warning(
        "No artifacts found yet. Run the pipeline first:\n\n"
        "```\npython scripts/run_pipeline.py\n```"
    )
    st.stop()

data = load_artifacts(ARTIFACT_DIR)
report = data["report"]

# --- Top-line metrics ---
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Sharpe Ratio", f"{report['sharpe_ratio']:.2f}")
c2.metric("Annualized Return", f"{report['annualized_return']:.1%}")
c3.metric("Max Drawdown", f"{report['max_drawdown']:.1%}")
c4.metric("Sortino Ratio", f"{report['sortino_ratio']:.2f}")
c5.metric("Avg Daily Turnover", f"{report['avg_daily_turnover']:.1%}")

st.divider()

# --- Equity curve + drawdown ---
left, right = st.columns([2, 1])
with left:
    st.subheader("Equity Curve (out-of-sample)")
    eq = data["equity_curve"]["equity"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name="Equity"))
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Drawdown")
    dd = metrics.drawdown_series(eq)
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", fill="tozeroy", name="Drawdown"))
    fig_dd.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_dd, use_container_width=True)

with right:
    st.subheader("Regime Performance")
    regime = data["regime_report"]
    if regime:
        for label, key in [("High Vol Regime", "high_vol_regime"), ("Low Vol Regime", "low_vol_regime")]:
            seg = regime.get(key, {})
            st.markdown(f"**{label}**")
            if seg.get("sharpe") is not None:
                st.write(f"Sharpe: {seg['sharpe']:.2f} | Ann. Return: {seg['ann_return']:.1%} | Days: {seg['n_days']}")
            else:
                st.write("Not enough data in this regime.")
        st.caption(
            f"Regime split by median trailing cross-sectional vol "
            f"({regime.get('median_annualized_vol_threshold', 0):.1%} annualized threshold)."
        )

    st.subheader("Cost Drag")
    total_costs = report.get("total_costs_paid", 0)
    st.write(f"Cumulative cost drag on returns: **{total_costs:.2%}**")
    st.caption("This is why turnover-heavy signals with a thin theoretical edge often don't survive contact with a real book.")

st.divider()

# --- Feature importance ---
if data["feature_importance"] is not None:
    st.subheader("Feature Importance (avg across walk-forward folds, per ticker)")
    fi = data["feature_importance"]
    ticker_choice = st.selectbox("Ticker", fi.columns.tolist())
    top_n = fi[ticker_choice].dropna().sort_values(ascending=False).head(15)
    fig_fi = go.Figure(go.Bar(x=top_n.values[::-1], y=top_n.index[::-1], orientation="h"))
    fig_fi.update_layout(height=450, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_fi, use_container_width=True)

st.divider()

# --- Walk-forward fold windows ---
st.subheader("Walk-Forward Validation Windows")
fold_metrics = data["fold_metrics"] or {}
if fold_metrics:
    ticker_for_folds = st.selectbox("Ticker (folds)", list(fold_metrics.keys()), key="fold_ticker")
    folds_df = pd.DataFrame(fold_metrics[ticker_for_folds])
    st.dataframe(folds_df, use_container_width=True)
    st.caption(
        "Each fold trains only on data strictly before its test window, with the "
        "label horizon purged from the train/test boundary and an embargo period "
        "afterward — see src/validation/walk_forward.py for why this matters."
    )

st.divider()

# --- Position weights over time ---
st.subheader("Position Weights Over Time")
weights = data["weights_history"]
if weights is not None and not weights.empty:
    tickers_to_plot = st.multiselect(
        "Tickers", weights.columns.tolist(), default=weights.columns.tolist()[:3]
    )
    if tickers_to_plot:
        fig_w = go.Figure()
        for t in tickers_to_plot:
            fig_w.add_trace(go.Scatter(x=weights.index, y=weights[t], mode="lines", name=t))
        fig_w.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_w, use_container_width=True)
