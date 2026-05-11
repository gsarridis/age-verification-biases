"""Argument helpers shared across CLI scripts."""
from __future__ import annotations

import argparse


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="configs/default.yaml",
                   help="Path to YAML config file.")
    p.add_argument("--override", "-o", action="append", default=[],
                   help="Override a config key, e.g. -o experiment.seed=7. Repeatable.")
