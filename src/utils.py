"""Shared utilities: config loading, logging setup."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path = None) -> dict:
    """Load the YAML config. Defaults to <project_root>/config.yaml."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def resolve_path(relative_path: str) -> Path:
    """Resolve a config path relative to the project root, creating it if needed."""
    p = PROJECT_ROOT / relative_path
    p.mkdir(parents=True, exist_ok=True)
    return p
