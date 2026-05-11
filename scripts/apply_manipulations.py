"""Apply all configured manipulations to Set B images."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from manipulations.pipeline import apply_to_manifest
from scripts._args import add_common_args
from utils import Paths, apply_overrides, get_logger, load_config, seed_everything

LOG = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply manipulations to Set B.")
    add_common_args(parser)
    args = parser.parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args.override)
    seed_everything(cfg["experiment"]["seed"])

    paths = Paths.from_config(cfg)
    set_b = paths.manifests_dir / "set_b_minors.csv"
    if not set_b.exists():
        LOG.error("Set B manifest not found at %s. Run scripts.build_test_sets first.", set_b)
        return 1

    apply_to_manifest(set_b, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
