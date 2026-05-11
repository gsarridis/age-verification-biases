"""Compute metrics from per-model predictions."""
from __future__ import annotations

import argparse
import sys

from evaluation.metrics import evaluate_all
from scripts._args import add_common_args
from utils import apply_overrides, load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute metrics from predictions.")
    add_common_args(parser)
    args = parser.parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args.override)
    evaluate_all(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
