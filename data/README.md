# UTKFace setup

UTKFace is not bundled with this repository.

## Where to get it

- Project page: https://susanqq.github.io/UTKFace/
- The dataset has been re-distributed in many forms; the canonical layout is one folder of JPEGs
  with filenames following:

      [age]_[gender]_[race]_[date&time].jpg.chip.jpg

  e.g. `7_0_3_20170109150557335.jpg.chip.jpg` => age 7, gender 0 (male), race 3.

## Variants

- **In-the-wild** (`UTKFace/Faces in the wild/`): full-frame photos.
- **Aligned & cropped** (`UTKFace/Aligned & Cropped/`): 200×200 pre-cropped faces.

The loader prefers the aligned & cropped variant if `prefer_aligned: true` in your config.

## Expected directory structure

```
$UTKFACE_DIR/
├── 1_0_0_20161219140622307.jpg.chip.jpg
├── 1_0_0_20161219140623097.jpg.chip.jpg
├── ...
```

Or, if you keep multiple variants:

```
$UTKFACE_DIR/
├── aligned/
│   └── *.jpg.chip.jpg
└── inthewild/
    └── *.jpg
```

The loader will glob recursively for `*.jpg` and `*.jpg.chip.jpg`.

## License & ethics reminder

UTKFace is released for **non-commercial research only**. Do not redistribute. If you intend
to publish results, check that your use complies with the dataset license and your
institution's ethics policies — particularly because UTKFace contains images of children.
