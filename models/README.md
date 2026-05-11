# Model setup guide

The framework supports five age-estimation models. They differ in how their weights are
distributed: some are downloaded automatically by their library, others require you to
fetch a checkpoint by hand. This file documents both.

> Run `python -m scripts.smoke_test_models` to verify every enabled model loads and
> produces a prediction on a single image. The summary at the end tells you which models
> are working and which need attention.

## 1. DeepFace (`backend: deepface`) — automatic download

DeepFace downloads its weights from `github.com/serengil/deepface_models/releases` on
first use. Default cache: `~/.deepface/`. If your network blocks GitHub releases:

```bash
# Manual download:
mkdir -p ~/.deepface/weights
wget -O ~/.deepface/weights/age_model_weights.h5 \
  https://github.com/serengil/deepface_models/releases/download/v1.0/age_model_weights.h5
```

DeepFace also pulls down its detector backend (default: `opencv`). Other detectors
(`retinaface`, `mtcnn`, etc.) trigger additional downloads.

## 2. InsightFace (`backend: insightface`) — automatic download

InsightFace downloads the `buffalo_l` analysis pack on first use. Default cache:
`~/.insightface/models/`. Manual fallback:

```bash
mkdir -p ~/.insightface/models
wget -O /tmp/buffalo_l.zip \
  https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
unzip /tmp/buffalo_l.zip -d ~/.insightface/models/buffalo_l/
```

For GPU inference, install `onnxruntime-gpu` (already in `requirements.txt`); CPU users
should swap it for `onnxruntime`.

## 3. ViT age classifier (`backend: hf_transformers`) — HuggingFace Hub

Pulls `nateraw/vit-age-classifier` from HuggingFace Hub on first use. Cached under
`~/.cache/huggingface/`. No manual setup required — the weights are ~340 MB.

You can swap to any other HF age classifier by changing `model_id` in the config; as long
as it returns `[batch, n_buckets]` logits with bucket labels in its `id2label`, the
wrapper will compute the expected age automatically.

## 4. MiVOLO (`backend: mivolo`) — manual install

MiVOLO is not pip-installable. Install it as an editable package from source:

```bash
git clone https://github.com/WildChlamydia/MiVOLO.git
cd MiVOLO
pip install -e .
```

Then download a checkpoint from the project's releases page and a YOLOv8 face detector
checkpoint. Point your config to the local file paths:

```yaml
models:
  - name: mivolo
    enabled: true
    backend: mivolo
    weights: /path/to/model_imdb_cross_person_4.22_99.46.pth.tar
    detector_weights: /path/to/yolov8x_person_face.pt
    device: cuda
    with_persons: false        # Set true if you also want to use body cues.
```

## 5. FairFace (`backend: fairface`) — manual download

```bash
mkdir -p assets/weights
# 4-race variant (recommended for age):
wget -O assets/weights/fairface_4race.pt \
  https://drive.google.com/uc?id=1pcxc4i75pqLb1Iqcnbpz2UeJoIwc5Lqe
# (Drive links are flaky — see the FairFace README for alternatives.)
```

Then point your config:

```yaml
- name: fairface
  enabled: true
  backend: fairface
  weights: assets/weights/fairface_4race.pt
```

> **Note on output layout**: the FairFace 4-race ResNet34 outputs 18 logits in the order
> `[race(7), gender(2), age(9)]`. Our wrapper takes the **last 9** logits as the age head
> by default, which is correct for the standard checkpoint. If you use a custom
> checkpoint with a different layout, override `age_slice`.

## Running multiple models

To run only a subset of models for a quick test, edit the `models` list in
`configs/default.yaml` (set `enabled: false` for the ones you want to skip). Or build a
new config that inherits the rest and override:

```bash
python -m scripts.run_models --config configs/default.yaml \
    -o models.0.enabled=false -o models.2.enabled=false
```

## Troubleshooting

* **OOM on a 24 GB GPU when running multiple models**: the runner explicitly tears down
  each model and calls `torch.cuda.empty_cache()` between models. If you still see OOM,
  it's likely from MiVOLO / Stable Diffusion competing for memory. Run them in separate
  invocations (the runner skips already-completed prediction CSVs, so resuming is cheap).
* **DeepFace says "tf-keras required"**: `pip install tf-keras`. Recent TensorFlow
  versions split Keras into a separate package and DeepFace hasn't fully adapted.
* **MediaPipe protobuf conflict** (when also installing tf-keras): the warning is benign;
  the framework only uses MediaPipe for landmark detection, which still works.
