from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import joblib
import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_SIZE = 64
SEED = 42


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    split: str
    label: str


def load_dataset_splits(data_dir: str | Path) -> dict[str, list[ImageRecord]]:
    """Read class-folder train, optional validation, and test splits."""
    root = Path(data_dir)
    records: dict[str, list[ImageRecord]] = {}
    split_dirs = [("train", root / "train"), ("test", root / "test")]
    validation_dirs = [
        root / name for name in ("validation", "val") if (root / name).is_dir()
    ]
    if len(validation_dirs) > 1:
        raise ValueError("Use only one validation folder: `validation/` or `val/`.")
    if validation_dirs:
        split_dirs.insert(1, ("validation", validation_dirs[0]))
    for split, split_dir in split_dirs:
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Expected dataset split folder: {split_dir}")
        split_records = []
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            split_records.extend(
                ImageRecord(path, split, class_dir.name)
                for path in sorted(class_dir.rglob("*"))
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
        if not split_records:
            raise ValueError(f"No supported images found under {split_dir}")
        records[split] = split_records

    train_classes = {record.label for record in records["train"]}
    for split, split_records in records.items():
        split_classes = {record.label for record in split_records}
        if train_classes != split_classes:
            missing_train = sorted(split_classes - train_classes)
            missing_split = sorted(train_classes - split_classes)
            raise ValueError(
                f"Train/{split} class folders must match. "
                f"Missing from train: {missing_train}; missing from {split}: {missing_split}"
            )
    return records


def load_records(data_dir: str | Path) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Backward-compatible reader for the train and test portions."""
    records = load_dataset_splits(data_dir)
    return records["train"], records["test"]


def analyze_dataset(data_dir: str | Path) -> dict[str, Any]:
    """Summarize class balance, image integrity, duplicates, and visual outliers."""
    split_records = load_dataset_splits(data_dir)
    records = [record for values in split_records.values() for record in values]
    split_names = list(split_records)
    class_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {split: 0 for split in split_names}
    )
    sizes: list[tuple[int, int]] = []
    brightness: list[float] = []
    hashes: dict[str, list[str]] = defaultdict(list)
    unreadable: list[str] = []

    for record in records:
        class_counts[record.label][record.split] += 1
        try:
            digest = hashlib.sha256(record.path.read_bytes()).hexdigest()
            hashes[digest].append(str(record.path))
            with Image.open(record.path) as source:
                source.verify()
            with Image.open(record.path) as source:
                image = source.convert("RGB")
                sizes.append(image.size)
                sample = image.resize((16, 16), Image.Resampling.BILINEAR)
                brightness.append(float(np.asarray(sample, dtype=np.float32).mean()))
        except (OSError, ValueError) as exc:
            unreadable.append(f"{record.path}: {exc}")

    counts = {label: dict(splits) for label, splits in sorted(class_counts.items())}
    train_counts = [value["train"] for value in counts.values()]
    duplicates = [paths for paths in hashes.values() if len(paths) > 1]
    cross_split_duplicates = [
        paths
        for paths in duplicates
        if len({Path(path).parts[-3] for path in paths}) > 1
    ]

    if sizes:
        width = np.asarray([size[0] for size in sizes], dtype=np.float64)
        height = np.asarray([size[1] for size in sizes], dtype=np.float64)
        area = width * height
        q1, q3 = np.percentile(area, [25, 75])
        iqr = q3 - q1
        area_outliers = int(np.sum((area < q1 - 1.5 * iqr) | (area > q3 + 1.5 * iqr)))
        aspect = width / height
        aspect_q1, aspect_q3 = np.percentile(aspect, [25, 75])
        aspect_iqr = aspect_q3 - aspect_q1
        aspect_outliers = int(
            np.sum(
                (aspect < aspect_q1 - 1.5 * aspect_iqr)
                | (aspect > aspect_q3 + 1.5 * aspect_iqr)
            )
        )
        size_summary = {
            "count": len(sizes),
            "width_min": int(width.min()),
            "width_median": int(np.median(width)),
            "width_max": int(width.max()),
            "height_min": int(height.min()),
            "height_median": int(np.median(height)),
            "height_max": int(height.max()),
            "resolution_iqr_outliers": area_outliers,
            "aspect_ratio_iqr_outliers": aspect_outliers,
            "below_256px_images": int(np.sum((width < 256) | (height < 256))),
        }
    else:
        size_summary = {"count": 0}

    train_total = len(split_records["train"])
    minority = min(train_counts)
    brightness_q1, brightness_q3 = (
        np.percentile(brightness, [25, 75]) if brightness else (0.0, 0.0)
    )
    brightness_iqr = brightness_q3 - brightness_q1
    brightness_outliers = int(
        sum(
            value < brightness_q1 - 1.5 * brightness_iqr
            or value > brightness_q3 + 1.5 * brightness_iqr
            for value in brightness
        )
    )
    return {
        "data_dir": str(Path(data_dir).resolve()),
        "total_images": len(records),
        "valid_images": len(sizes),
        "unreadable_count": len(unreadable),
        "unreadable_examples": unreadable[:10],
        "class_counts": counts,
        "train_images": train_total,
        "validation_images": len(split_records.get("validation", [])),
        "test_images": len(split_records["test"]),
        "train_imbalance_ratio": round(max(train_counts) / minority, 3),
        "exact_duplicate_groups": len(duplicates),
        "cross_split_duplicate_groups": len(cross_split_duplicates),
        "image_size": size_summary,
        "mean_sample_brightness_0_255": (
            round(float(np.mean(brightness)), 2) if brightness else None
        ),
        "brightness_iqr_outliers": brightness_outliers,
        "limitations": [
            "Image-level train/test splitting may not measure generalization to new farms, cameras, or backgrounds.",
            "The publisher's random image partitions may contain related collection conditions across splits.",
        ],
    }


def _hog_features(rgb: np.ndarray) -> np.ndarray:
    gray = rgb.mean(axis=2)
    gy, gx = np.gradient(gray)
    magnitude = np.hypot(gx, gy)
    orientation = np.mod(np.arctan2(gy, gx), np.pi)
    bins = np.floor(orientation * (8 / np.pi)).astype(np.int32) % 8
    cells: list[np.ndarray] = []
    for y in range(0, IMAGE_SIZE, 8):
        for x in range(0, IMAGE_SIZE, 8):
            hist = np.bincount(
                bins[y : y + 8, x : x + 8].ravel(),
                weights=magnitude[y : y + 8, x : x + 8].ravel(),
                minlength=8,
            ).astype(np.float32)
            norm = np.linalg.norm(hist)
            cells.extend([hist / (norm + 1e-6)])
    return np.concatenate(cells)


def extract_features(image: Image.Image) -> np.ndarray:
    """Create a compact color-histogram plus HOG descriptor for the classical model."""
    rgb_image = image.convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
    )
    rgb = np.asarray(rgb_image, dtype=np.float32) / 255.0
    hsv = np.asarray(rgb_image.convert("HSV"), dtype=np.uint8)
    color = np.concatenate(
        [
            np.histogram(hsv[:, :, channel], bins=16, range=(0, 256))[0]
            for channel in range(3)
        ]
    ).astype(np.float32)
    color /= color.sum() + 1e-6
    return np.concatenate([color, _hog_features(rgb)]).astype(np.float32)


def _features_for_paths(paths: list[Path]) -> np.ndarray:
    vectors = []
    for path in paths:
        with Image.open(path) as image:
            vectors.append(extract_features(image))
    return np.stack(vectors)


def _classification_metrics(
    truth: list[str], predicted: list[str], classes: list[str]
) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_precision": float(
            precision_score(truth, predicted, labels=classes, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(truth, predicted, labels=classes, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(truth, predicted, labels=classes, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(truth, predicted, labels=classes, average="weighted", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=classes).tolist(),
        "per_class": classification_report(
            truth, predicted, labels=classes, output_dict=True, zero_division=0
        ),
    }


def train_and_evaluate(
    data_dir: str | Path,
    model_dir: str | Path = "models",
    max_epochs: int = 16,
    dataset_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit the classical SVM and compact CNN; evaluate each once on the test split."""
    split_records = load_dataset_splits(data_dir)
    train_records = split_records["train"]
    validation_records = split_records.get("validation")
    test_records = split_records["test"]
    class_names = sorted({record.label for record in train_records})
    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    truth = [record.label for record in test_records]

    start = time.perf_counter()
    svm = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=SEED),
    )
    svm.fit(
        _features_for_paths([record.path for record in train_records]),
        [record.label for record in train_records],
    )
    svm_eval_start = time.perf_counter()
    svm_predictions = svm.predict(
        _features_for_paths([record.path for record in test_records])
    ).tolist()
    svm_latency = (time.perf_counter() - svm_eval_start) * 1000 / len(test_records)
    joblib.dump({"model": svm, "class_names": class_names}, output_dir / "classical.joblib")
    svm_metrics = _classification_metrics(truth, svm_predictions, class_names)
    svm_metrics["inference_ms_per_image"] = svm_latency
    svm_metrics["training_seconds"] = time.perf_counter() - start

    from avocado_vision.cnn import predict_cnn, train_cnn

    cnn_path = output_dir / "compact_cnn.pt"
    cnn_training = train_cnn(
        train_records,
        class_names,
        cnn_path,
        validation_records=validation_records,
        max_epochs=max_epochs,
    )
    cnn_predictions, cnn_latency = predict_cnn(cnn_path, test_records)
    cnn_metrics = _classification_metrics(truth, cnn_predictions, class_names)
    cnn_metrics.update(cnn_training)
    cnn_metrics["inference_ms_per_image"] = cnn_latency

    sample = test_records[0]
    sample_index = test_records.index(sample)
    if (
        dataset_analysis is None
        or Path(dataset_analysis.get("data_dir", "")).resolve()
        != Path(data_dir).resolve()
    ):
        dataset_analysis = analyze_dataset(data_dir)
    results = {
        "dataset": dataset_analysis,
        "classes": class_names,
        "metrics": {"classical_svm": svm_metrics, "compact_cnn": cnn_metrics},
        "example_run": {
            "image": str(sample.path),
            "actual_label": sample.label,
            "classical_svm_prediction": svm_predictions[sample_index],
            "compact_cnn_prediction": cnn_predictions[sample_index],
        },
        "model_selection": (
            "Select by macro F1 and macro recall, with accuracy and the confusion matrix "
            "as supporting measures; compare only on the untouched test split."
        ),
        "selection_note": (
            f"Best macro F1 on this test split: "
            f"{max(cnn_metrics['macro_f1'], svm_metrics['macro_f1']):.3f}. "
            "Scores describe the supplied avocado image subset, not guaranteed production performance."
        ),
    }
    results_path = output_dir / "metrics.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def predict_image(
    image: Image.Image | str | Path | BinaryIO, model_dir: str | Path = "models"
) -> dict[str, dict[str, Any]]:
    """Return class and confidence from both saved models for one uploaded image."""
    output_dir = Path(model_dir)
    if isinstance(image, Image.Image):
        source = ImageOps.exif_transpose(image).convert("RGB")
    else:
        with Image.open(image) as opened:
            source = ImageOps.exif_transpose(opened).convert("RGB")
    results: dict[str, dict[str, Any]] = {}
    feature = extract_features(source).reshape(1, -1)
    svm_bundle = joblib.load(output_dir / "classical.joblib")
    svm = svm_bundle["model"]
    svm_probabilities = svm.predict_proba(feature)[0]
    svm_classes = list(svm.classes_)
    svm_index = int(np.argmax(svm_probabilities))
    results["classical_svm"] = {
        "label": svm_classes[svm_index],
        "confidence": float(svm_probabilities[svm_index]),
    }
    from avocado_vision.cnn import predict_cnn_image

    results["compact_cnn"] = predict_cnn_image(
        source, output_dir / "compact_cnn.pt"
    )
    return results
