"""Build Set A and Set B manifests from the configured dataset."""
from __future__ import annotations

import argparse
import sys

from data.loader import load_dataset
from data.splits import build_and_save
from scripts._args import add_common_args
from utils import apply_overrides, load_config, seed_everything


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build test set manifests.")
    add_common_args(parser)
    args = parser.parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args.override)
    seed_everything(cfg["experiment"]["seed"])

    df = load_dataset(cfg)
    build_and_save(df, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
