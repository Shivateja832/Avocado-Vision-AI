# Avocado Detection and Ripeness Classification Report

## Dataset and task scope

This version uses [Avocado Ripeness YOLO Format](https://www.kaggle.com/datasets/deivdf/avocado-ripeness-yolo-format),
Kaggle dataset version 1, uploaded by David Fernando Hernandez Cuero and listed
under the MIT license. The archive's exact class list is `ripe`, `overripe`,
`unripe`, and generic `avocado`; it contains no other fruit classes. The model
uses the first three ripeness labels. Generic avocado-object boxes without a
stage label are excluded from ripeness training.

The source archive contains orchard images with avocado annotations. The
preparation process crops only boxes explicitly labeled `ripe`, `overripe`, or
`unripe`; these **3,996 prepared images are individual avocado crops**. No
other fruit class or fruit image is used as a training example. Because the
crops come from orchard photos, incidental background pixels can remain inside
an avocado box. The archive's generic avocado boxes are also avocado, but do
not identify a ripeness stage, so they are not included as training examples.
The crop classifiers are for one-fruit crops; a separate YOLO model performs
whole-scene localization and stage prediction.

The source data YAML has four object classes, but only three indicate a stage.
Consequently this app does not claim to classify `partially ripe` or `rotten`:
the chosen avocado-only source does not provide those labels. Crop classifiers
exclude generic `avocado`; the full-scene detector retains it as
`avocado (stage unknown)` so annotated avocados are not mislabeled as
background.

The preparation downloads the public version-1 archive, verifies its expected
size and ZIP CRC, records a local SHA-256, validates the source class schema,
and preserves the publisher's train/validation/test partitions. The crop
manifest records source images, label counts, boxes, crop coordinates, and
crop files; the detector manifest records full-scene image paths and
source-class box counts.
Source archive SHA-256:
`16d59b39fa0fc0de8773a770d672540eb6d426037ced056add4b4ebab75567d2`.

## Dataset inspection

| Stage | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| overripe | 1,079 | 119 | 134 | 1,332 |
| ripe | 1,089 | 123 | 120 | 1,332 |
| unripe | 1,028 | 157 | 147 | 1,332 |
| **Total crops** | **3,196** | **399** | **401** | **3,996** |

- Training majority/minority ratio is 1,089 / 1,028 = **1.06:1**, and the
  three stages have exactly 1,332 crops each overall.
- The source provides 4,423 train, 522 validation, and 453 test images.
  Its 7,743 generic `avocado` boxes do not specify stage and are excluded.
  Three training images have empty labels and do not yield stage crops.
- All 3,996 prepared crops were readable. Crop dimensions range from 237x236 to
  512x512, with median 315x445. The area-IQR rule flagged 5 resolution
  outliers, the aspect-ratio-IQR rule flagged 994, and 29 crops are below 256
  pixels on at least one side. These screening signals are not proof of
  incorrect labels; no samples were removed based on them.
- Mean sampled brightness is 107.78/255; the brightness-IQR rule flagged 25
  crops.
- There were zero exact duplicate crop files and zero source-image checksum
  duplicates crossing the published splits. This does not rule out visually
  similar photos or correlated orchard/camera conditions.
- Dataset bias remains possible: collection, background, illumination, camera,
  cultivar, stage-label policy, and the publisher's split strategy may affect
  results. The publisher's image partitions are not documented as orchard- or
  collection-session independent.

## Model choices

### Classical baseline: HSV/gradient descriptor + RBF SVM

The model combines HSV color histograms with cell-based gradient-orientation
histograms, then applies feature standardization and a class-balanced RBF
SVM. Visible color/texture changes are useful signals for avocado stage, and
the SVM is an efficient, interpretable baseline for comparison. Its
hand-crafted descriptors can miss spatial patterns and are sensitive to
illumination and backgrounds.

### Deep-learning approach: compact CNN

The compact CNN learns image features directly from 64x64 RGB crops. It uses
class-weighted loss, light flip/brightness/contrast augmentation, and
validation macro-F1 early stopping. The held-out test set is not used for
training or epoch selection. The CNN provides an end-to-end comparison, but
its small architecture and downsampling can lose useful detail.

### Full-scene detector: YOLO11n

Because the source includes bounding boxes, a small pretrained YOLO11n model
is fine-tuned on full orchard scenes using the publisher's four avocado-only
classes. This supports actual object localization and stage classification
for the three ripeness labels; the generic avocado class is surfaced as stage
unknown. To keep CPU training practical, the completed run used a deterministic,
balanced subset of 100 training scenes per source class (400 scenes total),
192-pixel inputs, batch size 8, 10 frozen backbone layers, and 5 epochs.
Validation and test used the complete publisher splits. Detection is evaluated
with precision, recall, mAP@0.5, and mAP@0.5:0.95, separately from
crop-classification accuracy and F1.

## Evaluation and measured results

Accuracy gives the overall correct fraction. Macro precision, recall, and F1
weight all three stages equally; macro recall exposes missed stages, while
macro F1 balances recall and precision. Weighted F1 gives more influence to
classes with more examples. Per-class values and confusion matrices reveal
errors that aggregate scores hide. Inference time is a local CPU measurement
per image, including image loading/preprocessing during the evaluation batch;
it is not a deployed service latency guarantee.

Both models were fitted using the train split; CNN early stopping used
validation. The 401-image test split was used for the final comparison.

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | ms/image |
|---|---:|---:|---:|---:|---:|---:|
| HSV/gradient + SVM | **87.03%** | **86.87%** | **87.01%** | **86.78%** | **87.08%** | 37.50 |
| Compact CNN | 79.30% | 79.13% | 78.75% | 78.89% | 79.57% | **16.03** |

The **SVM is the stronger accuracy/F1 choice** on this held-out test split:
it leads on accuracy and all F1/recall metrics. The CNN is faster in this local
measurement but gives up 7.89 percentage points of macro F1. These scores are
dataset-specific and do not establish real-world production accuracy.

**Full-scene YOLO11n detector** — evaluated on all 453 held-out test scenes:

| Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---:|---:|---:|---:|
| 53.32% | 74.46% | 53.58% | 48.53% |

Per-class mAP@0.5 is 50.07% for ripe, 48.52% for overripe, 88.40% for
unripe, and 7.14% for generic `avocado` (stage unknown). In particular, the
generic class result is poor; this detector is an evaluated assignment
baseline, not production-ready. The reduced training subset and five epochs
limit the strength of this result.

## Deployment and production-readiness evidence

The application is publicly deployed on
[Streamlit Community Cloud](https://avocado-vision-ai-shivateja832.streamlit.app/)
from [its GitHub repository](https://github.com/Shivateja832/Avocado-Vision-AI).
The hosted page responds successfully, and live smoke tests have exercised
both crop classification and full-scene detection. This verifies that the
packaged app and saved weights can load and run inference in the hosted
environment. It is operational evidence only—not evidence of real-world
accuracy, service-level availability, or generalization.

The publisher-held-out test split is not documented as farm-, camera-, or
collection-session-independent. To add cross-source evidence without
training or tuning on it, the saved crop models were also evaluated on the
avocado-only object crops from a separately published Mendeley dataset. The
hosted snapshot is pinned to commit
`321578b982118c7395b6cc3c8093396886a2fa99`; it has a single publisher split
and exposes numeric categories rather than semantic names. Only one-object
rows labeled with avocado IDs 4-7 were retained. This produced 457 unique
crops: 129 unripe, 112 ripe (source IDs 5 and 6 merged), and 216 overripe.
Strawberry-only/non-avocado rows and 119 ambiguous or invalid boxes were
excluded; no exact duplicate crops remained.

| Model | External accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| HSV/gradient + SVM | 34.14% | 54.19% | 40.68% | 33.56% | 31.47% |
| Compact CNN | 51.42% | 52.70% | 52.38% | 49.69% | 51.68% |

These are substantially below the publisher-held-out accuracy (87.03% SVM,
79.30% CNN). The result is evidence of poor cross-source transfer, not proof
of production readiness. In particular, ripe-stage precision/recall are
24.86%/79.46% for the SVM and 16.88%/23.21% for the CNN. Neither crop
classifier is suitable for unattended production decisions on this evidence.
The detector was not evaluated on this source; its 7.14% mAP@0.5 for generic
`avocado` on the publisher test set remains a material weakness.

The external snapshot covers Maharashtra, India and is described as farm and
market imagery, but it is one source rather than a multi-farm/camera
benchmark. Its card provides numeric class IDs; mapping IDs 4-7 to the
project's stages was inferred from the source scope and visually checked
samples, not confirmed by an explicit semantic label map. The source's
`ripening` category was merged with `ripe`, and its ID 7 wording may mean
rotten rather than the project's overripe definition. Portrait annotations
can also be mis-scaled or ambiguous; boxes were corrected only when exactly
one interpretation fit and otherwise excluded. Some backgrounds are
studio-like. Therefore this is a useful stress test, but not a definitive
field-accuracy estimate. See the
[Mendeley dataset record](https://data.mendeley.com/datasets/zysvgmxcyz/1),
[Hugging Face dataset card](https://huggingface.co/datasets/Project-AgML/strawberry_avocado_ripeness_detection),
and [public technical audit](https://github.com/CreAI-mx/agrizar_CV_datasets/blob/chore/initial-import/GROWTH_DATASET_FINDINGS.md).
The reproducible preparation and evaluation command is
`python validate_external_avocado.py`; external images remain in the
git-ignored `data/external_avocado_test/`, and metrics are written to
`models/avocado_pure/external_test_metrics.json`.

To establish stronger field generalization, obtain expert-labeled avocado
images from multiple farms, cameras, and collection sessions absent from
training. Freeze models and thresholds before evaluation, then report
per-stage precision, recall, macro-F1, confusion matrices, detector mAP, and
performance by acquisition domain. The current external result is not a
substitute for that multi-domain study.

The detector's precision/recall and mAP measure localization as well as class
assignment; they must not be ranked directly against the crop classifiers'
accuracy or macro F1. The crop SVM is the selected crop-classification model,
while YOLO is the only approach here that localizes fruit in a full scene.

Confusion matrices use rows = actual class and columns = predicted class; order
is `overripe`, `ripe`, `unripe`.

**HSV/gradient + SVM**

|  | Overripe | Ripe | Unripe |
|---|---:|---:|---:|
| Overripe | 105 | 25 | 4 |
| Ripe | 11 | 109 | 0 |
| Unripe | 10 | 2 | 135 |

**Compact CNN**

|  | Overripe | Ripe | Unripe |
|---|---:|---:|---:|
| Overripe | 98 | 32 | 4 |
| Ripe | 31 | 88 | 1 |
| Unripe | 12 | 3 | 132 |

The largest confusion is between the neighboring ripe and overripe stages.
Unripe examples have the strongest recall for both models.

## Example run

The saved test example is an **overripe** crop. The SVM predicted `ripe`;
the compact CNN predicted `overripe`. The same example and model outputs are
stored in `models/avocado_pure/metrics.json` and shown in the app's
**Train and compare** tab. On this example, the SVM's top-class confidence was
0.521 and the CNN's was 0.510. Uploaded images are run through both saved
models. Displayed confidence values are not calibrated probabilities.

The full-scene detector was also run on the saved test image
`colored_T10_d01_089_b_1_jpg_jpg.rf.f28e54e10af14d97849f20e7a50bf07e.jpg`.
Its one ground-truth unripe avocado was detected as `unripe` at confidence
0.808. The model returned a bounding box on the 800x800 image, and the app
renders the annotated image and detection table. The example input, ground
truth, and measured prediction are stored in
`models/avocado_detection/detection_metrics.json`.

## Required questions

### 1. In production, how would your model perform?

The best measured result is 87.03% accuracy and 86.78% macro F1 for the SVM
on the 401-image test split. Its evaluation latency was approximately 37.50
ms/image on this machine. On the separate 457-crop source, SVM accuracy and
macro F1 fell to 34.14% and 33.56%; CNN accuracy and macro F1 were 51.42% and
49.69%. The full-scene detector achieved 53.32% precision, 74.46% recall, and
53.58% mAP@0.5 on the publisher's held-out scenes, but was not tested on the
external source. These results do not support unattended production use.
The app is publicly deployed, and both hosted prediction workflows have
passed smoke tests. Those checks prove operational inference only, not
production accuracy.

### 2. What are the drawbacks?

- This avocado-only dataset labels three stages: unripe, ripe, and overripe.
  It cannot support partially-ripe or rotten predictions.
- The source's published split may share collection conditions across
  partitions; no independent farm/session holdout is documented.
- The cross-source crop test shows a substantial accuracy drop. Its single
  source and inferred/coarsened labels also limit the strength of the
  generalization conclusion.
- The SVM relies on color/texture descriptors and the CNN is less accurate on
  this split. Both may fail under lighting or background shifts.
- The detector was trained for only five epochs on 400 balanced scenes rather
  than the full training split. Its generic `avocado` class has just 7.14%
  mAP@0.5, so unknown-stage fruit detection is especially unreliable.
- The crop classifier expects a crop centered on one avocado; full-scene
  localization instead relies on the separately trained detector.
- The Kaggle dataset provides no detailed collection-method description on its
  page. Its annotations and source generalization warrant independent review.

### 3. What would you improve with more time?

Collect expert-verified avocado images across farms, cultivars, cameras, and
lighting, with a held-out farm/session split and authoritative stage
definitions. Use a confirmed semantic map and expert review for the external
source labels before drawing cross-source conclusions. Investigate the large
cross-source performance drop using training-only diagnostics, without
repeatedly tuning against the external test set. Test a pretrained lightweight
image model and calibrate its confidence on independent validation data. For
full-scene performance, train the detector on the complete training split for
more epochs at a larger image size, investigate the weak generic-avocado
results, and evaluate at multiple confidence thresholds on independent
orchard scenes.

### 4. Where did you spend the most time?

Most effort went into confirming that the primary source has only avocado
classes, verifying its labels and published splits, extracting and auditing
the stage-specific crops and full-scene annotations, then training and
evaluating the SVM, CNN, and YOLO detector. I also prepared an avocado-only
cross-source holdout, reviewed its annotation limitations, and measured the
large performance drop. The remaining effort integrated dataset
preparation, measured examples, and both prediction workflows into the app,
then deploying and smoke-testing the public version. Multi-farm/camera
generalization remains unverified.
