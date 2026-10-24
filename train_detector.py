from __future__ import annotations

import argparse
import json
import random
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from prepare_avocado_dataset import RIPENESS_CLASS_NAMES, parse_yolo_labels


DETECTOR_CLASS_NAMES = {
    **RIPENESS_CLASS_NAMES,
    3: "avocado",
}


def _create_balanced_training_subset(
    dataset_root: Path,
    subset_root: Path,
    images_per_class: int,
) -> dict[str, int]:
    candidates: dict[int, list[tuple[Path, Path]]] = {
        class_id: [] for class_id in DETECTOR_CLASS_NAMES
    }
    for image_path in sorted((dataset_root / "images" / "train").glob("*")):
        if not image_path.is_file():
            continue
        label_path = dataset_root / "labels" / "train" / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing detector label file: {label_path}")
        class_ids = {
            annotation[0]
            for annotation in parse_yolo_labels(
                label_path.read_text(encoding="utf-8"), len(DETECTOR_CLASS_NAMES)
            )
        }
        for class_id in class_ids:
            candidates[class_id].append((image_path, label_path))

    randomizer = random.Random(42)
    selected: dict[Path, Path] = {}
    for class_id, class_candidates in candidates.items():
        randomizer.shuffle(class_candidates)
        if len(class_candidates) < images_per_class:
            raise ValueError(
                f"Detector training needs {images_per_class} scenes for "
                f"{DETECTOR_CLASS_NAMES[class_id]}, but found "
                f"{len(class_candidates)}."
            )
        for image_path, label_path in class_candidates[:images_per_class]:
            selected[image_path] = label_path

    image_dir = subset_root / "images" / "train"
    label_dir = subset_root / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    annotation_counts: Counter[str] = Counter()
    for image_path, label_path in selected.items():
        shutil.copy2(image_path, image_dir / image_path.name)
        shutil.copy2(label_path, label_dir / label_path.name)
        for class_id, *_ in parse_yolo_labels(label_path.read_text(encoding="utf-8")):
            annotation_counts[DETECTOR_CLASS_NAMES[class_id]] += 1
    scene_counts = {DETECTOR_CLASS_NAMES[class_id]: 0 for class_id in candidates}
    for label_path in label_dir.glob("*.txt"):
        for class_id in {
            row[0]
            for row in parse_yolo_labels(label_path.read_text(encoding="utf-8"))
        }:
            scene_counts[DETECTOR_CLASS_NAMES[class_id]] += 1

    yaml_path = subset_root / "data.yaml"
    yaml_path.write_text(
        f"path: {json.dumps(str(dataset_root))}\n"
        f"train: {json.dumps(str(image_dir.resolve()))}\n"
        "val: images/validation\n"
        "test: images/test\n"
        "names:\n"
        "  0: ripe\n"
        "  1: overripe\n"
        "  2: unripe\n"
        "  3: avocado\n",
        encoding="utf-8",
    )
    return {
        "training_scenes": len(selected),
        "scenes_per_class": scene_counts,
        "boxes_per_class": dict(sorted(annotation_counts.items())),
    }


def train_detector(
    data_dir: str | Path = "data/avocado_detection",
    model_dir: str | Path = "models/avocado_detection",
    weights: str = "yolo11n.pt",
    epochs: int = 5,
    image_size: int = 192,
    batch_size: int = 8,
    images_per_class: int = 100,
) -> dict[str, Any]:
    """Fine-tune YOLO on a balanced subset and evaluate on the complete source test split."""
    dataset_root = Path(data_dir).resolve()
    dataset_yaml = dataset_root / "data.yaml"
    if not dataset_yaml.is_file():
        raise FileNotFoundError(
            f"Detection dataset not found at {dataset_yaml}. "
            "Run prepare_detector_dataset.py first."
        )
    if epochs < 1 or image_size < 64 or batch_size < 1 or images_per_class < 1:
        raise ValueError(
            "epochs, image_size, batch_size, and images_per_class must be positive."
        )

    from ultralytics import YOLO

    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="avocado_yolo_training_") as run_dir:
        subset_metrics = _create_balanced_training_subset(
            dataset_root, Path(run_dir) / "balanced_training", images_per_class
        )
        training_yaml = Path(run_dir) / "balanced_training" / "data.yaml"
        model = YOLO(weights)
        model.train(
            data=str(training_yaml),
            epochs=epochs,
            imgsz=image_size,
            batch=batch_size,
            device="cpu",
            workers=0,
            patience=max(4, epochs),
            freeze=10,
            project=run_dir,
            name="ripeness",
            exist_ok=True,
            plots=False,
            save=True,
            verbose=True,
        )
        best_weights = Path(model.trainer.best)
        if not best_weights.is_file():
            raise RuntimeError("YOLO training did not produce best.pt.")
        destination = output_dir / "best.pt"
        shutil.copy2(best_weights, destination)

        detector = YOLO(str(destination))
        validation = detector.val(
            data=str(dataset_yaml),
            split="test",
            imgsz=image_size,
            batch=batch_size,
            device="cpu",
            workers=0,
            project=str(output_dir / "evaluation"),
            name="test",
            exist_ok=True,
            plots=False,
            verbose=True,
        )
        box_metrics = validation.box
        test_images = sorted((dataset_root / "images" / "test").glob("*"))
        if not test_images:
            raise ValueError("The detector test split contains no image files.")
        example_image = test_images[0]
        example_result = detector.predict(
            source=str(example_image),
            imgsz=image_size,
            conf=0.25,
            device="cpu",
            save=False,
            verbose=False,
        )[0]
        example_labels_path = (
            dataset_root / "labels" / "test" / f"{example_image.stem}.txt"
        )
        actual_detections = []
        for line in example_labels_path.read_text(encoding="utf-8").splitlines():
            class_id, center_x, center_y, width, height = line.split()
            actual_detections.append(
                {
                    "class": example_result.names[int(class_id)],
                    "bbox_xywh_normalized": [
                        float(center_x),
                        float(center_y),
                        float(width),
                        float(height),
                    ],
                }
            )
        predicted_detections = []
        if example_result.boxes is not None:
            for box in example_result.boxes:
                predicted_detections.append(
                    {
                        "class": example_result.names[int(box.cls.item())],
                        "confidence": float(box.conf.item()),
                        "bbox_xyxy_pixels": [
                            float(value) for value in box.xyxy[0].tolist()
                        ],
                    }
                )
        metrics = {
            "dataset_dir": str(dataset_root),
            "weights": str(destination.resolve()),
            "base_weights": weights,
            "device": "cpu",
            "epochs_requested": epochs,
            "epochs_completed": int(model.trainer.epoch) + 1,
            "image_size": image_size,
            "batch_size": batch_size,
            "training_subset": subset_metrics,
            "frozen_backbone_layers": 10,
            "test_metrics": {
                "precision": float(box_metrics.mp),
                "recall": float(box_metrics.mr),
                "map50": float(box_metrics.map50),
                "map50_95": float(box_metrics.map),
                "per_class_map50": {
                    name: float(value)
                    for name, value in zip(
                        validation.names.values(),
                        box_metrics.maps,
                        strict=True,
                    )
                },
            },
            "example_run": {
                "image": str(example_image.relative_to(dataset_root).as_posix()),
                "actual_detections": actual_detections,
                "predicted_detections": predicted_detections,
            },
        }
    metrics_path = output_dir / "detection_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and test an avocado ripeness object detector."
    )
    parser.add_argument("--data-dir", default="data/avocado_detection")
    parser.add_argument("--model-dir", default="models/avocado_detection")
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    results = train_detector(
        args.data_dir,
        args.model_dir,
        args.weights,
        args.epochs,
        args.image_size,
        args.batch_size,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
