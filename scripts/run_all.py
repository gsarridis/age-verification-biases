"""End-to-end runner: build sets -> manipulate -> run models -> evaluate -> report -> showcase."""

from __future__ import annotations

import argparse
import sys

from data.loader import load_dataset
from data.splits import build_and_save
from evaluation.metrics import evaluate_all
from evaluation.runner import run_all
from manipulations.pipeline import apply_to_manifest
from reports.generate import generate
from scripts.make_showcase import make_showcase
from scripts._args import add_common_args
from utils import Paths, apply_overrides, get_logger, load_config, seed_everything

LOG = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the full age-bias pipeline.")
    add_common_args(parser)
    parser.add_argument(
        "--skip-manipulations",
        action="store_true",
        help="Skip the manipulation step (assumes it's already been run).",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip running models (assumes predictions exist).",
    )
    parser.add_argument(
        "--showcase-n",
        type=int,
        default=10,
        help="Number of subjects to include in the showcase figure.",
    )
    args = parser.parse_args(argv)

    cfg = apply_overrides(load_config(args.config), args.override)
    seed_everything(cfg["experiment"]["seed"])
    paths = Paths.from_config(cfg)

    LOG.info("=== Step 1/6: building manifests ===")
    df = load_dataset(cfg)
    build_and_save(df, cfg)

    if not args.skip_manipulations:
        LOG.info("=== Step 2/6: applying manipulations ===")
        set_b = paths.manifests_dir / "set_b_minors.csv"
        apply_to_manifest(set_b, cfg)

    if not args.skip_models:
        LOG.info("=== Step 3/6: running models ===")
        run_all(cfg)

    LOG.info("=== Step 4/6: computing metrics ===")
    evaluate_all(cfg)

    LOG.info("=== Step 5/6: generating report ===")
    out = generate(cfg)

    LOG.info("=== Step 6/6: generating showcase figure ===")
    try:
        make_showcase(cfg, n_samples=args.showcase_n)
    except Exception as e:
        LOG.warning("Showcase generation failed (non-fatal): %s", e)

    LOG.info("Done. Open %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
