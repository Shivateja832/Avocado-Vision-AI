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

## Train and run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
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

## Public deployment

The app is prepared for **Streamlit Community Cloud**. Connect a GitHub
repository containing `app.py`, `requirements.txt`, and the trained weights
under `models/`, then deploy `app.py` from the Streamlit Community Cloud
dashboard. The prepared datasets under `data/` are excluded from Git; dataset
inspection can use the saved evaluation results, and data can be regenerated
with the preparation scripts.

A public URL is verified only after the repository is connected and the
deployed app is smoke-tested. This workspace does not include GitHub or
Streamlit account credentials. Do not commit account secrets or Kaggle tokens.

## Verification

```powershell
python -m pytest -q tests
python -m compileall -q app.py train.py train_detector.py prepare_avocado_dataset.py prepare_detector_dataset.py avocado_vision tests
```

For local use, run `streamlit run app.py`.
