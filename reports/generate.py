"""Generate plots and an HTML report from the metrics CSVs.

Plots produced (under <report_dir>/):
  * set_a_mae.png            — MAE per model (overall, minors, adults) with 95% CIs.
  * binary_accuracy.png      — accuracy at 13 on original vs each manipulation, per model.
  * delta_predictions.png    — mean Δ predicted age (manipulation − original) per model × manipulation.
  * flip_rates.png           — % of correctly-classified minors flipped to adult by each manipulation.

Report:
  * report.html              — narrative walk-through, leading with the 13-year story.

The showcase figure (showcase.png) is produced separately by ``scripts.make_showcase``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import Paths, get_logger

LOG = get_logger(__name__)

THRESHOLD = 13

# Consistent palette across plots: one color per model.
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


# ---------------------------------------------------------------------------
# Plot 1: Set A MAE per model (overall + minor / adult)
# ---------------------------------------------------------------------------


def _plot_set_a_mae(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path).sort_values("model").reset_index(drop=True)
    models = df["model"].tolist()

    fig, ax = plt.subplots(
        figsize=(max(8, 1.6 * len(models)), 4.8), constrained_layout=True
    )
    x = np.arange(len(models))
    w = 0.27

    groups = [
        ("Overall", "mae_overall", "mae_overall_lo", "mae_overall_hi", "#444444"),
        (
            "Age range: [6,12]",
            "mae_minors",
            "mae_minors_lo",
            "mae_minors_hi",
            "#1f77b4",
        ),
        (
            "Age range: [13,20]",
            "mae_adults",
            "mae_adults_lo",
            "mae_adults_hi",
            "#ff7f0e",
        ),
    ]
    for k, (label, col, lo, hi, color) in enumerate(groups):
        means = df[col].to_numpy()
        yerr_lo = means - df[lo].to_numpy()
        yerr_hi = df[hi].to_numpy() - means
        ax.bar(
            x + (k - 1) * w,
            means,
            w,
            label=label,
            yerr=[yerr_lo, yerr_hi],
            capsize=3,
            color=color,
            alpha=0.9,
        )
        # for xi, m in zip(x + (k - 1) * w, means):
        #     ax.text(xi, m + 0.4, f"{m:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=10, ha="right")
    ax.set_ylabel("Mean Absolute Error (years)")
    # ax.set_title("Set A — Overall age estimation accuracy (lower = better)")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    LOG.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
# Plot 2: Binary accuracy at 13 — original vs each manipulation, per model.
# ---------------------------------------------------------------------------


def _plot_binary_accuracy(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    models = sorted(df["model"].unique())
    manips = sorted(df["manipulation"].unique())

    fig, ax = plt.subplots(
        figsize=(max(8, 1.4 * (1 + len(manips)) * len(models) / 3), 5),
        constrained_layout=True,
    )

    # Grouped bars: x = manipulations, hue = model.
    n_models = len(models)
    bar_w = 0.8 / n_models

    # We want both 'original' baseline and each manipulation. The CSV has one row per
    # (model, manipulation), and 'acc_original' is repeated across manipulations
    # (it's the same model on the same set of originals). Pull a single per-model
    # baseline.
    baselines = df.groupby("model")["acc_original"].mean().reindex(models).to_dict()

    # Plot original baselines as horizontal lines.
    x = np.arange(len(manips))
    for k, model in enumerate(models):
        sub = df[df["model"] == model].set_index("manipulation").reindex(manips)
        means = sub["acc_manipulated"].to_numpy()
        yerr_lo = means - sub["acc_manipulated_lo"].to_numpy()
        yerr_hi = sub["acc_manipulated_hi"].to_numpy() - means
        offset = (k - (n_models - 1) / 2) * bar_w
        ax.bar(
            x + offset,
            means,
            bar_w,
            yerr=[yerr_lo, yerr_hi],
            capsize=2,
            color=PALETTE[k % len(PALETTE)],
            label=model,
        )
        # Print value on top of each bar.
        for xi, m in zip(x + offset, means):
            ax.text(xi, m + 0.03, f"{m:.0%}", ha="center", va="bottom", fontsize=7)

    # Add baseline lines (one per model, dashed in the matching color).
    for k, model in enumerate(models):
        ax.axhline(
            baselines[model],
            color=PALETTE[k % len(PALETTE)],
            linestyle="--",
            alpha=0.5,
            lw=1,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(manips, rotation=15, ha="right")
    ax.set_ylabel(f"Accuracy")
    ax.set_ylim(0, 1.05)
    # ax.set_title(
    #     f"Set B — Binary accuracy at age {THRESHOLD} threshold\n"
    #     "Dashed lines = baseline (original images). Bars = accuracy after manipulation."
    # )
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    LOG.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
# Plot 3: Mean Δ predicted age per model × manipulation.
# ---------------------------------------------------------------------------


def _plot_delta_predictions(csv_path: Path, out_path: Path) -> None:
    """Two-panel plot: prediction-vs-original (left) and prediction-vs-truth (right).

    Left panel: how much does the manipulation shift the model's *own* prediction?
                (model robustness)
    Right panel: how far is the prediction from ground truth before vs after manipulation?
                 (real-world impact)
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    models = sorted(df["model"].unique())
    manips = sorted(df["manipulation"].unique())

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(max(12, 2.0 * len(manips) * len(models) / 3), 5),
        constrained_layout=True,
    )
    x = np.arange(len(manips))
    n_models = len(models)
    bar_w = 0.8 / n_models

    # ---------- Left: Δ vs original prediction ----------
    for k, model in enumerate(models):
        sub = df[df["model"] == model].set_index("manipulation").reindex(manips)
        means = sub["mean_delta_years"].to_numpy()
        yerr_lo = means - sub["mean_delta_years_lo"].to_numpy()
        yerr_hi = sub["mean_delta_years_hi"].to_numpy() - means
        offset = (k - (n_models - 1) / 2) * bar_w
        ax_left.bar(
            x + offset,
            means,
            bar_w,
            yerr=[yerr_lo, yerr_hi],
            capsize=2,
            color=PALETTE[k % len(PALETTE)],
            label=model,
        )
        for xi, m in zip(x + offset, means):
            ax_left.text(
                xi,
                m + (0.5 if m >= 0 else -0.3),
                f"{m:+.1f}",
                ha="center",
                va="bottom" if m >= 0 else "top",
                fontsize=7,
            )

    ax_left.axhline(0, color="black", lw=0.8)
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(manips, rotation=15, ha="right")
    ax_left.set_ylabel("Δ predicted age (years)")
    # ax_left.set_title(
    #     "Baseline = model's own prediction on the original\n(measures: robustness)"
    # )
    ax_left.legend(loc="upper right")
    ax_left.grid(axis="y", alpha=0.3)

    # ---------- Right: signed error vs ground truth, paired bars ----------
    # Two bars per (model × manipulation): "original" and "manipulated" signed error.
    # We collapse this into one panel by showing original (hatched) next to manipulated (solid).
    bar_w2 = 0.8 / (n_models * 2)
    for k, model in enumerate(models):
        sub = df[df["model"] == model].set_index("manipulation").reindex(manips)
        orig = sub["mean_signed_error_original"].to_numpy()
        manp = sub["mean_signed_error_manipulated"].to_numpy()
        # Place originals slightly left, manipulated slightly right of each model's slot.
        center_offset = (k - (n_models - 1) / 2) * (bar_w2 * 2)
        color = PALETTE[k % len(PALETTE)]
        ax_right.bar(
            x + center_offset - bar_w2 / 2,
            orig,
            bar_w2,
            color=color,
            alpha=0.55,
            hatch="//",
            edgecolor="white",
            label=f"{model} (original)" if k == 0 else None,
        )
        ax_right.bar(
            x + center_offset + bar_w2 / 2,
            manp,
            bar_w2,
            color=color,
            alpha=1.0,
            label=f"{model} (manipulated)" if k == 0 else None,
        )

    ax_right.axhline(0, color="black", lw=0.8)
    ax_right.set_xticks(x)
    ax_right.set_xticklabels(manips, rotation=15, ha="right")
    ax_right.set_ylabel("Signed error (years)\n(predicted − true age)")
    # ax_right.set_title("Baseline = ground-truth age\n(measures: real-world impact)")

    # Build a custom legend: one entry per model + a hatch/solid legend.
    from matplotlib.patches import Patch

    handles = []
    for k, model in enumerate(models):
        handles.append(Patch(color=PALETTE[k % len(PALETTE)], label=model))
    handles.append(
        Patch(
            facecolor="gray",
            alpha=0.55,
            hatch="//",
            edgecolor="white",
            label="original",
        )
    )
    handles.append(Patch(facecolor="gray", alpha=1.0, label="manipulated"))
    ax_right.legend(handles=handles, loc="upper right", fontsize=8, ncol=2)
    ax_right.grid(axis="y", alpha=0.3)

    # fig.suptitle("Set B — Two views of the same manipulation effect", fontsize=12)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    LOG.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
# Plot 4: Flip rates — % of correctly classified minors flipped to "adult".
# ---------------------------------------------------------------------------


def _plot_flip_rates(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    models = sorted(df["model"].unique())
    manips = sorted(df["manipulation"].unique())

    fig, ax = plt.subplots(
        figsize=(max(8, 1.4 * len(manips) * len(models) / 3), 5),
        constrained_layout=True,
    )
    x = np.arange(len(manips))
    n_models = len(models)
    bar_w = 0.8 / n_models

    for k, model in enumerate(models):
        sub = df[df["model"] == model].set_index("manipulation").reindex(manips)
        rates = sub["flip_rate"].to_numpy()
        offset = (k - (n_models - 1) / 2) * bar_w
        ax.bar(x + offset, rates, bar_w, color=PALETTE[k % len(PALETTE)], label=model)
        for xi, r, n in zip(x + offset, rates, sub["n_correct_to_wrong"].to_numpy()):
            if not np.isnan(r):
                ax.text(
                    xi,
                    r + 0.01,
                    f"{r:.0%}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(manips, rotation=15, ha="right")
    ax.set_ylabel("Flip rate")
    ax.set_ylim(0, max(0.5, np.nanmax(df["flip_rate"]) * 1.3))
    # ax.set_title(
    #     "Set B — Flip rate per manipulation\n"
    #     "(of minors correctly classified on the original, the % flipped to ‘adult’ after manipulation)"
    # )
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    LOG.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8">
<title>Can a fake mustache fool age verification?</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          max-width: 1100px; margin: 2em auto; padding: 0 1em; color: #222;
          line-height: 1.5; }}
  h1 {{ font-size: 1.9em; margin-bottom: 0.2em; }}
  h2 {{ border-bottom: 2px solid #ddd; padding-bottom: 0.3em; margin-top: 2em; }}
  h3 {{ color: #444; margin-top: 1.5em; }}
  .subtitle {{ color: #666; font-size: 1.1em; margin-bottom: 2em; }}
  .lede {{ font-size: 1.1em; padding: 1em 1.4em;
           background: #f8f9fc; border-left: 4px solid #4a72c2;
           border-radius: 3px; margin: 1em 0 2em; }}
  table {{ border-collapse: collapse; margin: 1em 0; font-size: 0.92em;
           font-variant-numeric: tabular-nums; }}
  th, td {{ border: 1px solid #ccc; padding: 5px 9px; text-align: right; }}
  th {{ background: #f4f4f4; }}
  td.left, th.left {{ text-align: left; }}
  .bad {{ color: #cc1f1f; font-weight: 600; }}
  .good {{ color: #1f7a1f; }}
  img {{ max-width: 100%; height: auto; margin: 1em 0; border: 1px solid #eee; }}
  .caption {{ font-size: 0.88em; color: #666; margin-top: -0.5em; }}
  .takeaway {{ background: #fff8db; border-left: 4px solid #f0c000;
               padding: 0.8em 1.2em; margin: 1em 0; }}
  .callout {{ background: #fdf3f3; border-left: 4px solid #cc1f1f;
              padding: 0.8em 1.2em; margin: 1em 0; }}
  .callout strong {{ color: #cc1f1f; }}
  .warn {{ background: #fff3cd; border-left: 4px solid #ffc107;
           padding: 0.5em 1em; }}
  .method {{ background: #f4f4f4; border-left: 4px solid #888;
             padding: 0.5em 1em; font-size: 0.92em; }}
  code {{ background: #f4f4f4; padding: 2px 5px; border-radius: 3px; font-size: 0.9em; }}
  pre {{ background: #f4f4f4; padding: 0.8em; border-radius: 3px; overflow-x: auto; }}
</style></head>
<body>

<h1>Can a fake mustache fool age verification?</h1>
<p class="subtitle">
Testing how three open-source age-estimation models respond to simple, freely-available
image manipulations on faces of children.
</p>

<div class="lede">
{lede_summary}
</div>

<h2>The setup</h2>
<p>
  We sample {n_set_a} faces from UTKFace covering ages 2–60, balanced 50/50 between
  minors (under 13) and adults. We also sample {n_set_b} faces of children aged 6–12,
  apply three simple manipulations to each (a thin painted mustache, beard stubble, and
  heavy eye makeup), and measure how the predictions of three open-source models change
  between the original image and the manipulated version of the same subject.
</p>
<p>The models we test cover three different architectures and training regimes:</p>
<ul>
  <li><strong>FairFace</strong> — ResNet-34 trained on a demographically balanced face
      dataset, designed for fairness across race and gender. 9-bucket classifier.</li>
  <li><strong>MiVOLO</strong> — multi-input transformer that combines face and body
      cues. Regression head; widely cited as state-of-the-art on standard benchmarks.</li>
  <li><strong>ViT age classifier</strong> — Vision Transformer fine-tuned for age
      estimation, distributed via HuggingFace.</li>
</ul>
<p>
  All three are open-source and easy to use — exactly the kind of model a hobbyist or
  small product team would actually plug into a "verify the user is over 13" pipeline.
  The manipulations we apply are equally accessible: programmatic OpenCV overlays driven
  by face landmarks. Anyone with a few hours of Python could produce them.
</p>

<div class="method">
  <strong>Methodology note.</strong> All metrics are computed on identical UTKFace
  samples across all three models. For Set B, we evaluate each model on the
  <em>original</em> image and the <em>manipulated</em> version of the same subject, so
  every Δ is a paired comparison — the manipulation effect is isolated from
  per-subject difficulty. Confidence intervals are 95% percentile bootstrap with
  1,000 resamples.
</div>

<h2>Question 1: How accurate are these models, in general?</h2>
<p>
  Mean Absolute Error (MAE) on Set A — the balanced minors-vs-adults set:
</p>
{set_a_table}
<img src="set_a_mae.png" alt="Set A MAE chart">
<p class="caption">
  Lower is better. Error bars are bootstrap 95% CIs. Notice that MAE on minors is
  systematically higher than MAE on adults across all three models — these models are
  better at estimating adult ages than children's, which is consistent with the
  age distributions of the datasets they were trained on.
</p>

<h2>Question 2: How much does each manipulation push predicted age up?</h2>
<p>
  We measure this two ways. <strong>Left:</strong> how much does the manipulation
  shift the model's <em>own</em> prediction on the same subject? (model robustness).
  <strong>Right:</strong> how far is the prediction from the ground-truth age before
  vs. after manipulation? (real-world impact).
</p>
<img src="delta_predictions.png" alt="Delta predictions chart">
<p class="caption">
  In the right panel, hatched bars are the model's signed error on the original image
  (predicted − true age). Solid bars are signed error after manipulation. The gap
  between hatched and solid <em>is</em> the manipulation effect. Bars above zero mean
  the model is over-predicting age (treating children as older than they are).
</p>
{set_b_delta_table}

<h2>Question 3: Does the manipulation push predictions across the 13-year line?</h2>
<p>
  This is the crux. Age verification is fundamentally a binary decision: is this user
  under 13 or not? The chart below shows what fraction of children in Set B each model
  correctly classifies as a minor — first on the original photo, then on each manipulated
  version. The dashed line is the baseline: how the model does without any tampering.
</p>
<img src="binary_accuracy.png" alt="Binary accuracy at 13">

<h3>The flip analysis</h3>
<p>
  For each (model × manipulation), we look at the children that the model
  <em>did</em> correctly classify on the original photo, and ask: what fraction got
  flipped to "adult" after the manipulation was applied? This isolates the
  manipulation's effect from the model's underlying error rate.
</p>
{set_b_binary_table}
<img src="flip_rates.png" alt="Flip rates by manipulation">

<h2>Examples</h2>
<p>
  Concrete cases. Each row below is the same child; columns are the original photo
  and each manipulation. Below each panel, every model's predicted age is shown in
  <span class="good">green</span> if the model still correctly predicts under 13 and
  in <span class="bad">red</span> if the manipulation pushed the prediction over 13 —
  i.e., the model would now incorrectly grant adult access.
</p>
<img src="showcase.png" alt="Showcase of example subjects">

{example_callouts}

<h2>Takeaways</h2>
<div class="takeaway">
{takeaway_text}
</div>

<h2>Caveats</h2>
<div class="warn">
<ul>
  <li>The manipulations here are deliberately simple — overlays drawn programmatically
      with OpenCV, not photorealistic. A motivated adversary could do much better; this
      is a lower bound, not an upper bound.</li>
  <li>Our test set is drawn from UTKFace; results on other distributions
      (different ethnicities, lighting conditions, image qualities) may differ.</li>
  <li>The 13-year threshold is a regulatory line, not a perceptual one. Even a
      well-calibrated age estimator should have a confidence interval that crosses 13
      for any 12-year-old, simply because the difference between a 12-year-old and a
      13-year-old isn't reliably visible from a single photo. So we should expect
      <em>some</em> error here even without any adversarial manipulation.</li>
  <li>Bootstrap 95% CIs assume i.i.d. samples and ignore demographic clustering.</li>
</ul>
</div>

<h2>Reproducing this</h2>
<p>
  Code: <code>github.com/your-org/age-bias-test</code>. After downloading UTKFace and
  the model weights:
</p>
<pre><code>python -m scripts.run_all --config configs/default.yaml
python -m scripts.make_showcase --config configs/default.yaml --n 5</code></pre>

</body></html>"""


def _df_to_html_table(
    df: pd.DataFrame, drop_cols: list[str] | None = None, float_format: str = "{:.2f}"
) -> str:
    if df.empty:
        return "<p><em>No data.</em></p>"
    cols = [c for c in df.columns if not drop_cols or c not in drop_cols]
    fmt = df[cols].copy()
    for c in fmt.columns:
        if pd.api.types.is_float_dtype(fmt[c]):
            fmt[c] = fmt[c].map(
                lambda v: float_format.format(v) if pd.notna(v) else "—"
            )
    head = "<tr>" + "".join(f"<th>{c}</th>" for c in fmt.columns) + "</tr>"
    body = ""
    for _, r in fmt.iterrows():
        body += "<tr>" + "".join(f"<td>{v}</td>" for v in r) + "</tr>"
    return f"<table>{head}{body}</table>"


def _build_lede_summary(set_a_df: pd.DataFrame, binary_df: pd.DataFrame) -> str:
    """One-paragraph lede that fronts the most striking finding."""
    if binary_df.empty:
        return "<p>Run the pipeline to populate this report with results.</p>"

    # Worst flip rate across all (model, manipulation) pairs.
    valid = binary_df.dropna(subset=["flip_rate"])
    if valid.empty:
        return "<p>No paired predictions available; manipulations may not have been applied.</p>"
    worst = valid.loc[valid["flip_rate"].idxmax()]
    n_total = int(worst["n_correct_to_correct"]) + int(worst["n_correct_to_wrong"])

    # Best-case (lowest flip rate) for contrast — same model or different.
    best = valid.loc[valid["flip_rate"].idxmin()]

    return f"""
<p>
  We applied three simple, programmatically-generated manipulations — a thin painted
  mustache, beard stubble, and heavy eye makeup — to photographs of children aged 6–12,
  and tested whether three popular open-source age-estimation models still classified
  the children as minors. The headline finding: when <strong>{worst['model']}</strong>
  saw children with an added <strong>{worst['manipulation']}</strong>, it
  flipped <strong class="bad">{worst['flip_rate']:.0%}</strong>
  ({int(worst['n_correct_to_wrong'])} of {n_total}) of the children it had originally
  classified correctly into the "adult" category. The smallest effect we observed
  ({best['model']} on {best['manipulation']}) still flipped
  <strong>{best['flip_rate']:.0%}</strong>
  ({int(best['n_correct_to_wrong'])} of {int(best['n_correct_to_correct']) + int(best['n_correct_to_wrong'])}).
  None of these manipulations is hard to apply.
</p>
"""


def _build_example_callouts(
    predictions_dir: Path,
    manifest_path: Path,
    n_callouts: int = 3,
    manipulation_filter: list[str] | None = None,
) -> str:
    """Pick a handful of specific (subject, model, manipulation) tuples that illustrate
    the failure mode, and write narrative callouts naming the numbers."""
    # Concatenate all set_b prediction CSVs to find dramatic individual flips.
    rows = []
    for p in sorted(predictions_dir.glob("set_b_manipulated__*.csv")):
        model = p.stem.split("__", 1)[1]
        df = pd.read_csv(p)
        df["model"] = model
        rows.append(df)
    if not rows:
        return ""
    preds = pd.concat(rows, ignore_index=True).dropna(subset=["predicted_age"])
    if manipulation_filter:
        keep = set(manipulation_filter) | {"original"}
        preds = preds[preds["manipulation"].isin(keep)]

    # Pair originals with manipulations.
    originals = preds[preds["manipulation"] == "original"][
        ["sample_id", "model", "predicted_age", "true_age"]
    ].rename(columns={"predicted_age": "pred_original"})
    others = preds[preds["manipulation"] != "original"]
    paired = others.merge(
        originals[["sample_id", "model", "pred_original", "true_age"]],
        on=["sample_id", "model"],
        suffixes=("", "_orig"),
        how="inner",
    )
    paired["delta"] = paired["predicted_age"] - paired["pred_original"]
    # We only want cases where: classified correctly on original AND wrongly on manipulated.
    paired["correct_orig"] = paired["pred_original"] < THRESHOLD
    paired["correct_manip"] = paired["predicted_age"] < THRESHOLD
    flipped = paired[paired["correct_orig"] & ~paired["correct_manip"]].copy()
    if flipped.empty:
        return (
            "<p><em>(No example callouts: no children flipped from correct→wrong "
            "in this run.)</em></p>"
        )

    # Select a diverse set: try to get one per (manipulation) and span true ages.
    flipped["score"] = flipped["delta"]  # Bigger jump = more dramatic.
    selected: list[pd.Series] = []
    seen_manips: set[str] = set()
    seen_models: set[str] = set()
    for _, row in flipped.sort_values("score", ascending=False).iterrows():
        if len(selected) >= n_callouts:
            break
        # Spread across manipulations first.
        if (
            row["manipulation"] in seen_manips
            and len(seen_manips) < flipped["manipulation"].nunique()
        ):
            continue
        selected.append(row)
        seen_manips.add(row["manipulation"])
        seen_models.add(row["model"])

    bullets = []
    for row in selected:
        bullets.append(
            f"<li>A <strong>{int(row['true_age'])}-year-old</strong> child (subject "
            f"<code>{row['sample_id']}</code>) was correctly classified by "
            f"<strong>{row['model']}</strong> on the original photo "
            f"(predicted age = {row['pred_original']:.1f}). After we added "
            f"<strong>{row['manipulation']}</strong>, the same model predicted "
            f"<strong class='bad'>{row['predicted_age']:.1f}</strong> — "
            f"a jump of <strong>{row['delta']:+.1f} years</strong>, "
            f"placing them above the 13-year threshold.</li>"
        )

    return f"""
<div class="callout">
<strong>A few specific cases worth pointing out:</strong>
<ul>
{''.join(bullets)}
</ul>
</div>
"""


def _build_takeaway_text(
    set_a_df: pd.DataFrame, delta_df: pd.DataFrame, binary_df: pd.DataFrame
) -> str:
    """Auto-generate the bullet-point takeaways from the data."""
    bullets: list[str] = []

    if not set_a_df.empty:
        worst = set_a_df.loc[set_a_df["mae_minors"].idxmax()]
        bullets.append(
            f"Even on clean images, the worst-performing model on minors "
            f"({worst['model']}) is off by {worst['mae_minors']:.1f} years on average."
        )

    if not delta_df.empty:
        # Most age-inflating (model, manipulation) pair.
        idx = delta_df["mean_delta_years"].idxmax()
        worst = delta_df.loc[idx]
        bullets.append(
            f"The largest mean age-inflation we observed was "
            f"+{worst['mean_delta_years']:.1f} years, from "
            f"<strong>{worst['manipulation']}</strong> on <strong>{worst['model']}</strong>."
        )

    if not binary_df.empty:
        # Worst flip rate.
        idx = binary_df["flip_rate"].idxmax()
        worst = binary_df.loc[idx]
        if not np.isnan(worst["flip_rate"]):
            bullets.append(
                f"For <strong>{worst['model']}</strong>, the "
                f"<strong>{worst['manipulation']}</strong> manipulation flipped "
                f"<strong>{worst['flip_rate']:.0%}</strong> "
                f"({int(worst['n_correct_to_wrong'])}/{int(worst['n_correct_to_correct']) + int(worst['n_correct_to_wrong'])}) "
                f"of correctly-classified minors over the 13-year line."
            )
        # Largest accuracy drop.
        idx2 = binary_df["delta_accuracy"].idxmax()
        worst2 = binary_df.loc[idx2]
        bullets.append(
            f"The biggest single drop in age-13 accuracy was "
            f"<strong>{worst2['acc_original']:.0%} → {worst2['acc_manipulated']:.0%}</strong> "
            f"(–{worst2['delta_accuracy']:.0%}), from "
            f"<strong>{worst2['manipulation']}</strong> on <strong>{worst2['model']}</strong>."
        )

    if not bullets:
        return "<p>(Insufficient data to auto-generate takeaways.)</p>"
    return "<ul>" + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate(cfg: dict) -> Path:
    paths = Paths.from_config(cfg)
    metrics_dir = paths.metrics_dir
    out_dir = paths.report_dir

    set_a_csv = metrics_dir / "set_a_mae.csv"
    delta_csv = metrics_dir / "set_b_delta_predictions.csv"
    binary_csv = metrics_dir / "set_b_binary_threshold.csv"

    if set_a_csv.exists():
        _plot_set_a_mae(set_a_csv, out_dir / "set_a_mae.png")
    if delta_csv.exists():
        _plot_delta_predictions(delta_csv, out_dir / "delta_predictions.png")
    if binary_csv.exists():
        _plot_binary_accuracy(binary_csv, out_dir / "binary_accuracy.png")
        _plot_flip_rates(binary_csv, out_dir / "flip_rates.png")

    set_a_df = pd.read_csv(set_a_csv) if set_a_csv.exists() else pd.DataFrame()
    delta_df = pd.read_csv(delta_csv) if delta_csv.exists() else pd.DataFrame()
    binary_df = pd.read_csv(binary_csv) if binary_csv.exists() else pd.DataFrame()

    set_a_table = _df_to_html_table(
        set_a_df,
        drop_cols=["mean_predicted_age_minors", "mean_predicted_age_adults"],
    )
    set_b_delta_table = _df_to_html_table(
        delta_df,
        drop_cols=["mean_pred_original", "mean_pred_manipulated", "mean_true_age"],
    )
    set_b_binary_table = _df_to_html_table(
        binary_df,
        drop_cols=[
            "acc_original_lo",
            "acc_original_hi",
            "acc_manipulated_lo",
            "acc_manipulated_hi",
        ],
    )

    n_set_a = int(set_a_df["n"].max()) if not set_a_df.empty else "?"
    # Set B size = unique sample_id count, which we can read from the manifest.
    set_b_man = paths.manifests_dir / "set_b_minors.csv"
    n_set_b = (
        len(pd.read_csv(set_b_man))
        if set_b_man.exists()
        else (binary_df["n_paired"].max() if not binary_df.empty else "?")
    )

    takeaway_text = _build_takeaway_text(set_a_df, delta_df, binary_df)
    lede_summary = _build_lede_summary(set_a_df, binary_df)
    manip_filter = cfg.get("evaluation", {}).get("manipulation_filter") or None
    example_callouts = _build_example_callouts(
        paths.predictions_dir,
        paths.manifests_dir / "set_b_manipulated.csv",
        manipulation_filter=manip_filter,
    )

    html = HTML_TEMPLATE.format(
        n_set_a=n_set_a,
        n_set_b=n_set_b,
        set_a_table=set_a_table,
        set_b_delta_table=set_b_delta_table,
        set_b_binary_table=set_b_binary_table,
        takeaway_text=takeaway_text,
        lede_summary=lede_summary,
        example_callouts=example_callouts,
    )
    out_html = out_dir / "report.html"
    out_html.write_text(html)
    LOG.info("Wrote report: %s", out_html)
    return out_html
