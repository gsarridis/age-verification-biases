# Model weights

This folder is for any model weights / checkpoints the framework needs at runtime.
Most are auto-downloaded on first use; a few must be fetched manually.

## Auto-downloaded

| File | Size | Source | Used by |
|---|---|---|---|
| `face_landmarker.task` | ~3 MB | https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task | MediaPipe Tasks fallback for landmark detection |

## Optional manual downloads

### dlib 68-point facial landmark predictor
Used as a last-resort fallback if neither legacy nor Tasks MediaPipe APIs work.

- File: `shape_predictor_68_face_landmarks.dat`
- Source: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
- Steps:
  ```bash
  cd assets/weights
  wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
  bunzip2 shape_predictor_68_face_landmarks.dat.bz2
  ```

### MiVOLO checkpoint (optional, only if `models.mivolo.enabled: true`)
- Repo: https://github.com/WildChlamydia/MiVOLO
- Download a checkpoint (e.g. `mivolo_d1.pth`) following the repo's README and place
  it at the path you set under `models[].weights` in your config (default:
  `assets/weights/mivolo_d1.pth`). MiVOLO also requires a YOLO person/face detector;
  see the MiVOLO README for details.

### FairFace checkpoint (optional, only if `models.fairface.enabled: true`)
- Repo: https://github.com/joojs/fairface
- Default checkpoint: `res34_fair_align_multi_4_20190809.pt` — place at
  `assets/weights/fairface_age_4race.pt` or wherever you point your config.
