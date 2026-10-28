# AvocadoVision AI

An avocado-only computer-vision application that compares a classical image
classifier with a compact CNN and detects avocado boxes in full-scene images.
Supported ripeness labels are `unripe`, `ripe`, and `overripe`; a generic
avocado box without a stage is shown as `avocado (stage unknown)`.

## Dataset

The app uses [Avocado Ripeness YOLO Format](https://www.kaggle.com/datasets/deivdf/avocado-ripeness-yolo-format),
Kaggle dataset version 1, listed under the MIT license. Its source schema has
four avocado-only labels: `ripe`, `overripe`, `unripe`, and generic `avocado`.
The generic label has no ripeness stage; it is omitted from crop
classification, but retained as a distinct class for full-scene detection so
these annotated fruit are not incorrectly treated as background.

Run dataset preparation:

```powershell
python prepare_avocado_dataset.py
python prepare_detector_dataset.py
```

The crop-classification data and provenance manifest are written to
`data/avocado_pure/`. The full-scene YOLO dataset, containing images and
bounding boxes for only the source's avocado categories, is written to
`data/avocado_detection/`. Both scripts verify the public archive's exact
size, SHA-256, ZIP integrity, source label schema, and publisher splits.
Dataset inspection requires the prepared `train/` and `test/` folders. If
they are missing, use the app's **Download and prepare dataset now** action,
or run `python prepare_avocado_dataset.py` in the project root before
analyzing or training.

## Train and run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python prepare_avocado_dataset.py
python train.py
python prepare_detector_dataset.py
python train_detector.py
streamlit run app.py
```

Crop models and metrics are stored in `models/avocado_pure/`. YOLO11n detector
weights and held-out test metrics are stored in `models/avocado_detection/`.
To keep CPU training practical, it uses a deterministic, class-balanced subset
of 100 training scenes per source label; validation and test use the complete
publisher splits. The training report records the subset and class
composition. Increase epochs, image size, and subset size on a faster or
GPU-equipped machine. The UI supports full-scene detection and single-avocado
crop classification.

## Models and evaluation

- **Classical crop classifier:** HSV histograms and cell-based gradient
  descriptors with a standardized, class-balanced RBF SVM.
- **Deep-learning crop classifier:** compact CNN trained with class-weighted
  loss, light augmentation, and validation macro-F1 early stopping.
- **Full-scene detector:** YOLO11n fine-tuned on the source's avocado bounding
  boxes and all four avocado-only annotations.

Classifier results include accuracy, macro/weighted F1, recall, precision,
per-class scores, confusion matrices, and an example. The detector is
evaluated with precision, recall, mAP@0.5, and mAP@0.5:0.95 on the publisher's
test split. See [REPORT.md](./REPORT.md) for measured results, dataset audit,
limitations, and all required assignment answers.

An optional cross-source crop-classification evaluation is available:

```powershell
python validate_external_avocado.py
```

It downloads only single-object avocado annotations from a separately
published dataset, saves provenance under the git-ignored
`data/external_avocado_test/`, and evaluates the existing SVM and CNN without
training or tuning on those images. Metrics are written to
`models/avocado_pure/external_test_metrics.json`. The source exposes numeric
labels, so the crosswalk is approximate; this is one-source evidence, not a
multi-farm benchmark. See the report before interpreting its scores.

## Public deployment

The application is deployed on **Streamlit Community Cloud**:
[open AvocadoVision AI](https://avocado-vision-ai-shivateja832.streamlit.app/).
Its source repository is [Shivateja832/Avocado-Vision-AI](https://github.com/Shivateja832/Avocado-Vision-AI).
The public deployment has been smoke-tested for the crop-classification and
full-scene detection workflows. This confirms that the hosted application
loads and can run inference; it does not establish production accuracy or
generalization to new farms, cameras, or collection conditions.

To deploy a fork, connect its GitHub repository in the Streamlit Community
Cloud dashboard and select `app.py` as the entry point. The prepared datasets
under `data/` are excluded from Git; inference uses the saved model artifacts
under `models/`, and data can be regenerated with the preparation scripts.
The root `packages.txt` installs the `libgl1` and `libglib2.0-0` system
libraries required by OpenCV/Ultralytics inference on the Linux host.

The deployed app is a demonstration, not a production service with a verified
uptime or accuracy guarantee. Do not commit account secrets or Kaggle tokens.

## Verification

Install the development/test dependencies (the file also installs the
application dependencies) before running tests:

```powershell
python -m pip install -r requirements-dev.txt
```

For a deployment/runtime-only environment, install `requirements.txt`;
pytest is kept out of the application runtime dependencies.

```powershell
python -m pytest -q tests
python -m compileall -q app.py train.py train_detector.py prepare_avocado_dataset.py prepare_detector_dataset.py validate_external_avocado.py avocado_vision tests
```

For local use, run `streamlit run app.py`.
