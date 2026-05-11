# Age Verification Model Bias Testing Framework

A research framework for evaluating how open-source age estimation / age verification models are influenced by visual manipulations such as added facial hair, heavy makeup, glasses, and other appearance modifications. The framework focuses in particular on **failure modes that affect minors** — i.e., manipulations that cause a model to misclassify a child (<13) as an adult.

## What this framework does

1. **Loads** a face dataset with age annotations (UTKFace by default).
2. **Builds two test sets**:
   - **Set A — Balanced overall**: equal numbers of <13 and ≥13 samples for general performance evaluation.
   - **Set B — Minors only, original vs. manipulated**: only <13 samples, each present in an original version and in N manipulated versions (e.g., added mustache, added makeup). This isolates the effect of the manipulation.
3. **Applies manipulations** using two pipelines:
   - **Classical CV** (OpenCV/dlib landmark-based overlays — fast, deterministic, reproducible).
   - **GenAI inpainting** (Stable Diffusion inpainting — slower, more realistic).
4. **Runs multiple open-source age models** on both sets and records predictions.
5. **Evaluates** with metrics tailored to age verification:
   - MAE (mean absolute error in years).
   - Threshold-crossing rate at 13, 18, 21.
   - **Adult misclassification rate for minors** (the headline number).
   - Per-manipulation performance drop.
6. **Reports** results as CSVs and plots.

## Models tested

| Model | Source | License | Notes |
|---|---|---|---|
| DeepFace (Age model) | `deepface` PyPI | MIT | Wraps a VGG-Face age model. |
| MiVOLO | github.com/WildChlamydia/MiVOLO | MIT | SoTA, uses both face + body. |
| FairFace | github.com/joojs/fairface | Custom (research) | Trained for fairness across demographics. |
| InsightFace (genderage) | `insightface` PyPI | Apache-2.0 | Lightweight, ONNX. |
| ViT age classifier | HuggingFace `nateraw/vit-age-classifier` | Apache-2.0 | 9-bucket classifier. |

> Each model wrapper inherits from a common `AgeModel` interface so adding new models is straightforward.

## Datasets

UTKFace is the recommended primary dataset because it contains a wide age range (0–116) including many young children, with filenames encoding age/gender/race. Download instructions are in `data/README.md`.

> **Ethical note**: This framework intentionally does not bundle any face images. You must download UTKFace yourself under its own license, and you are responsible for ensuring your use complies with applicable laws, your institution's ethics policies, and the dataset license. Consider IRB approval if this is academic work involving images of minors.

## Quickstart

```bash
# Install
pip install -r requirements.txt

# (Optional, for the inpainting pipeline)
pip install -r requirements-genai.txt

# Configure dataset path
export UTKFACE_DIR=/path/to/UTKFace

# Build the test sets (writes manifests to results/)
python -m scripts.build_test_sets --config configs/default.yaml

# Apply manipulations to Set B
python -m scripts.apply_manipulations --config configs/default.yaml

# Run all models on both sets
python -m scripts.run_models --config configs/default.yaml

# Compute metrics and generate the report
python -m scripts.evaluate --config configs/default.yaml
python -m scripts.report --config configs/default.yaml
```

## Project layout

```
age_bias_test/
├── data/             # Dataset loaders + test-set construction
├── models/           # Age-model wrappers (one file per model family)
│   └── README.md     # Model setup guide (weight downloads, troubleshooting)
├── manipulations/    # Image manipulation pipelines (classical + GenAI)
├── evaluation/       # Metrics + experiment runner
├── reports/          # Plotting / report generation
├── scripts/          # CLI entry points
├── configs/          # YAML configs
├── assets/overlays/  # PNG overlays (mustache, glasses, etc.)
├── outputs/          # Reference preview + final report (preview_astronaut.jpg)
└── results/          # Manifests + per-model predictions (generated)
```

## Before running on real data: verification scripts

Two scripts help you sanity-check the framework on your machine before launching a full
experiment:

```bash
# 1. Visualize every manipulation on one image. Reference: outputs/preview_astronaut.jpg
python -m scripts.preview_on_face --image /path/to/any/face.jpg

# 2. Verify each enabled model loads & predicts on one image. Useful after first install.
python -m scripts.smoke_test_models --image /path/to/any/face.jpg
```

The first script catches problems with the manipulation pipeline (missing landmarks,
ill-fitting overlays). The second catches problems with model setup (missing weights,
incompatible TensorFlow versions, etc.) — see `models/README.md` for per-model setup
instructions.

## Tests

```bash
pytest tests/ -q
```

There are 31 tests covering: config loading, UTKFace filename parsing, metric math
(MAE, threshold-crossing, bootstrap CIs), manipulation registry, overlay generation,
the synthetic end-to-end pipeline (manifest → manipulation → metrics → report), and
model-wrapper logic that doesn't require weight downloads.

## Citing / referencing

If you publish results obtained with this framework, please cite the underlying datasets and models — see `CITATIONS.md`.
