"""Generate the HTML report from metrics."""
from __future__ import annotations

import argparse
import sys

from reports.generate import generate
from scripts._args import add_common_args
from utils import apply_overrides, load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the HTML report.")
    add_common_args(parser)
    args = parser.parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args.override)
    generate(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
