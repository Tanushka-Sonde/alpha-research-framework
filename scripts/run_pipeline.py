#!/usr/bin/env python
"""
Run the full alpha research pipeline end to end.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --config config.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import run_full_pipeline  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    result = run_full_pipeline(config_path=args.config)
    report = result["report"]

    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY (out-of-sample, walk-forward)")
    print("=" * 60)
    print(f"Annualized return     : {report['annualized_return']:.2%}")
    print(f"Annualized volatility : {report['annualized_vol']:.2%}")
    print(f"Sharpe ratio          : {report['sharpe_ratio']:.2f}")
    print(f"Sortino ratio         : {report['sortino_ratio']:.2f}")
    print(f"Max drawdown          : {report['max_drawdown']:.2%}")
    print(f"Calmar ratio          : {report['calmar_ratio']:.2f}")
    print(f"Hit rate              : {report['hit_rate']:.2%}")
    print(f"Avg daily turnover    : {report['avg_daily_turnover']:.2%}")
    print(f"Final equity          : {report['final_equity']:,.0f}")
    print("=" * 60)
    print("\nRun `streamlit run dashboard/app.py` to explore interactively.\n")


if __name__ == "__main__":
    main()
