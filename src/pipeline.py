"""
End-to-end orchestration.

load OHLCV -> per-ticker features/labels -> purged walk-forward loop
(train a model per fold, predict OOS fold only) -> stitch OOS predictions
into a single out-of-sample signal panel -> backtest with costs & vol
targeting -> risk report + persisted artifacts for the dashboard.

Everything the dashboard shows is read from these persisted artifacts, so
`scripts/run_pipeline.py` must be run at least once before the dashboard
has anything to display.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.backtest.engine import BacktestConfig, BacktestEngine
from src.data.features import FeatureConfig, assemble_dataset
from src.data.loader import DataLoader, DataLoaderConfig
from src.models.gbm_model import GBMSignalModel, LGBMConfig
from src.reporting import metrics
from src.validation.walk_forward import PurgedWalkForward, WalkForwardConfig

logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict):
    model_cfg = cfg["model"]
    if model_cfg["type"] == "lightgbm":
        return GBMSignalModel(LGBMConfig(**model_cfg["lightgbm"]))
    elif model_cfg["type"] == "lstm":
        from src.models.lstm_model import LSTMConfig, LSTMSignalModel

        return LSTMSignalModel(LSTMConfig(**model_cfg["lstm"]))
    else:
        raise ValueError(f"Unknown model type: {model_cfg['type']}")


def run_walk_forward_for_ticker(dataset: pd.DataFrame, cfg: dict) -> dict:
    """
    dataset: feature+label DataFrame for ONE ticker, indexed by date.
    Returns dict with:
      - oos_predictions: Series of out-of-sample predicted labels, indexed by date
      - fold_metrics: list of per-fold info (train/test size, dates)
      - feature_importance: averaged importance across folds (GBM only)
    """
    feature_cols = [c for c in dataset.columns if c != "label"]
    X, y = dataset[feature_cols], dataset["label"]

    wf_cfg = WalkForwardConfig(
        n_splits=cfg["validation"]["n_splits"],
        min_train_size=cfg["validation"]["min_train_size"],
        test_size=cfg["validation"]["test_size"],
        embargo_days=cfg["validation"]["embargo_days"],
        label_horizon=cfg["label"]["horizon"],
    )
    splitter = PurgedWalkForward(wf_cfg)

    oos_pred = pd.Series(np.nan, index=dataset.index)
    fold_infos = []
    importances = []

    for fold_i, (train_idx, test_idx) in enumerate(splitter.split(dataset.index)):
        # carve a small validation slice off the END of train for early stopping
        # (still strictly before test, so no leakage)
        val_frac = 0.15
        n_val = max(1, int(len(train_idx) * val_frac))
        inner_train_idx, inner_val_idx = train_idx[:-n_val], train_idx[-n_val:]

        X_tr, y_tr = X.iloc[inner_train_idx], y.iloc[inner_train_idx]
        X_val, y_val = X.iloc[inner_val_idx], y.iloc[inner_val_idx]
        X_te = X.iloc[test_idx]

        model = build_model(cfg)
        model.fit(X_tr, y_tr, X_val, y_val)
        preds = model.predict(X_te)
        oos_pred.iloc[test_idx] = preds

        fold_infos.append(
            {
                "fold": fold_i,
                "train_start": str(dataset.index[train_idx[0]].date()),
                "train_end": str(dataset.index[train_idx[-1]].date()),
                "test_start": str(dataset.index[test_idx[0]].date()),
                "test_end": str(dataset.index[test_idx[-1]].date()),
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            }
        )
        if hasattr(model, "feature_importance"):
            importances.append(model.feature_importance())

    avg_importance = None
    if importances:
        avg_importance = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)

    return {
        "oos_predictions": oos_pred,
        "fold_metrics": fold_infos,
        "feature_importance": avg_importance,
    }


def run_full_pipeline(config_path: str = "config.yaml", output_dir: str | None = None) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config(config_path)
    output_dir = output_dir or cfg["reporting"]["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- 1. Load data ---
    loader = DataLoader(
        DataLoaderConfig(
            tickers=cfg["data"]["tickers"],
            start_date=cfg["data"]["start_date"],
            end_date=cfg["data"]["end_date"],
            interval=cfg["data"]["interval"],
            cache_dir=cfg["data"]["cache_dir"],
            use_synthetic_if_offline=cfg["data"]["use_synthetic_if_offline"],
        )
    )
    universe = loader.load_universe()
    logger.info("Loaded %d tickers", len(universe))

    feat_cfg = FeatureConfig(**cfg["features"])

    all_oos_signals = {}
    all_realized_1d_returns = {}
    all_fold_metrics = {}
    all_importances = {}

    for ticker, ohlcv in universe.items():
        logger.info("=== Processing %s ===", ticker)
        dataset = assemble_dataset(
            ohlcv, feat_cfg, cfg["label"]["horizon"], cfg["label"]["target_type"]
        )
        if len(dataset) < cfg["validation"]["min_train_size"] + cfg["validation"]["test_size"]:
            logger.warning("%s: not enough data after feature/label warmup, skipping", ticker)
            continue

        result = run_walk_forward_for_ticker(dataset, cfg)
        all_oos_signals[ticker] = result["oos_predictions"]
        all_fold_metrics[ticker] = result["fold_metrics"]
        if result["feature_importance"] is not None:
            all_importances[ticker] = result["feature_importance"]

        # realized 1-day close-to-close returns, needed by the backtest engine
        # for daily P&L accounting (separate from the H-day forward label)
        realized_1d = ohlcv["close"].pct_change().reindex(dataset.index)
        all_realized_1d_returns[ticker] = realized_1d

    signals_panel = pd.DataFrame(all_oos_signals).dropna(how="all")
    returns_panel = pd.DataFrame(all_realized_1d_returns).reindex(signals_panel.index)

    # Only keep dates where at least 2 tickers have a signal (need a cross-section
    # for the quantile long/short bucketing to mean anything)
    valid_dates = signals_panel.notna().sum(axis=1) >= min(2, signals_panel.shape[1])
    signals_panel = signals_panel[valid_dates]
    returns_panel = returns_panel.reindex(signals_panel.index)

    logger.info(
        "Out-of-sample signal panel: %d dates x %d tickers", *signals_panel.shape
    )

    # --- 3. Backtest ---
    bt_cfg = BacktestConfig(**cfg["backtest"])
    engine = BacktestEngine(bt_cfg)
    bt_result = engine.run(signals_panel, returns_panel)

    # --- 4. Risk report ---
    report = metrics.full_report(
        bt_result["daily_returns"],
        bt_result["equity_curve"],
        bt_result["turnover"],
        bt_result["costs"],
        cfg["reporting"]["risk_free_rate_annual"],
    )
    logger.info("Performance report: %s", json.dumps(report, indent=2, default=str))

    # --- 5. Regime split (vol regime) for the dashboard ---
    regime_report = _regime_breakdown(bt_result["daily_returns"], returns_panel)

    # --- 6. Persist everything the dashboard needs ---
    _persist_artifacts(
        output_dir, signals_panel, returns_panel, bt_result, report,
        all_fold_metrics, all_importances, regime_report,
    )

    return {
        "signals_panel": signals_panel,
        "returns_panel": returns_panel,
        "backtest_result": bt_result,
        "report": report,
        "fold_metrics": all_fold_metrics,
        "feature_importance": all_importances,
        "regime_report": regime_report,
    }


def _regime_breakdown(daily_returns: pd.Series, returns_panel: pd.DataFrame, window: int = 21) -> dict:
    """Splits strategy performance by realized market-vol regime (high vs low vol),
    using the cross-sectional average asset volatility as the regime proxy."""
    avg_asset_vol = returns_panel.rolling(window).std().mean(axis=1) * np.sqrt(252)
    avg_asset_vol = avg_asset_vol.reindex(daily_returns.index)
    median_vol = avg_asset_vol.median()
    high_vol_mask = avg_asset_vol >= median_vol
    low_vol_mask = ~high_vol_mask

    def _seg_report(mask):
        seg = daily_returns[mask.fillna(False)]
        if len(seg) < 10:
            return {"n_days": len(seg), "sharpe": None, "ann_return": None}
        return {
            "n_days": int(len(seg)),
            "sharpe": metrics.sharpe_ratio(seg),
            "ann_return": metrics.annualized_return(seg),
        }

    return {
        "high_vol_regime": _seg_report(high_vol_mask),
        "low_vol_regime": _seg_report(low_vol_mask),
        "median_annualized_vol_threshold": float(median_vol) if not np.isnan(median_vol) else None,
    }


def _persist_artifacts(output_dir, signals_panel, returns_panel, bt_result, report,
                        fold_metrics, importances, regime_report):
    out = Path(output_dir)

    signals_panel.to_parquet(out / "signals_panel.parquet")
    returns_panel.to_parquet(out / "returns_panel.parquet")
    bt_result["equity_curve"].to_frame("equity").to_parquet(out / "equity_curve.parquet")
    bt_result["daily_returns"].to_frame("daily_return").to_parquet(out / "daily_returns.parquet")
    bt_result["weights_history"].to_parquet(out / "weights_history.parquet")
    bt_result["turnover"].to_frame("turnover").to_parquet(out / "turnover.parquet")
    bt_result["costs"].to_frame("costs").to_parquet(out / "costs.parquet")

    with open(out / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(out / "fold_metrics.json", "w") as f:
        json.dump(fold_metrics, f, indent=2, default=str)
    with open(out / "regime_report.json", "w") as f:
        json.dump(regime_report, f, indent=2, default=str)

    if importances:
        imp_df = pd.DataFrame(importances)
        imp_df.to_parquet(out / "feature_importance.parquet")

    logger.info("Artifacts written to %s", out.resolve())
