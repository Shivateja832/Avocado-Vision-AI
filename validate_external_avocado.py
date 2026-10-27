from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import joblib
import numpy as np
import requests
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from avocado_vision.cnn import predict_cnn
from avocado_vision.pipeline import ImageRecord, extract_features

DATASET_ID = "Project-AgML/strawberry_avocado_ripeness_detection"
DATASET_COMMIT = "321578b982118c7395b6cc3c8093396886a2fa99"
ROWS_API = "https://datasets-server.huggingface.co/rows"
AVOCADO_SOURCE_LABELS = {
    4: "unripe",
    5: "ripe",
    6: "ripe",
    7: "overripe",
}
STAGE_NAMES = ("overripe", "ripe", "unripe")
MODEL_DIR = Path("models/avocado_pure")
DEFAULT_DATA_DIR = Path("data/external_avocado_test")


def _fits_image(
    x: float, y: float, width: float, height: float, image_width: int, image_height: int
) -> bool:
    return (
        all(math.isfinite(value) for value in (x, y, width, height))
        and x >= 0
        and y >= 0
        and width > 0
        and height > 0
        and x + width <= image_width
        and y + height <= image_height
    )


def resolve_external_box(
    bbox: list[float], image_width: int, image_height: int
) -> tuple[tuple[int, int, int, int], str] | None:
    """Resolve unambiguous source xywh boxes; skip portrait boxes that fit both ways."""
    if len(bbox) != 4:
        raise ValueError(f"Expected an xywh box with four values, got {bbox!r}.")
    x, y, width, height = (float(value) for value in bbox)
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")

    original_fits = _fits_image(x, y, width, height, image_width, image_height)
    if image_width >= image_height:
        if not original_fits:
            return None
        chosen = (x, y, width, height)
        correction = "native_landscape"
    else:
        aspect = image_width / image_height
        corrected = (x * aspect, y / aspect, width * aspect, height / aspect)
        corrected_fits = _fits_image(*corrected, image_width, image_height)
        if original_fits == corrected_fits:
            return None
        chosen = corrected if corrected_fits else (x, y, width, height)
        correction = "portrait_aspect_corrected" if corrected_fits else "native_portrait"

    left, top, box_width, box_height = chosen
    box = (
        max(0, math.floor(left)),
        max(0, math.floor(top)),
        min(image_width, math.ceil(left + box_width)),
        min(image_height, math.ceil(top + box_height)),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box, correction


def _source_stage(class_id: int) -> str | None:
    return AVOCADO_SOURCE_LABELS.get(class_id)


def _fetch_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with requests.Session() as session:
        for offset in range(0, 758, 25):
            params = {
                "dataset": DATASET_ID,
                "config": "default",
                "split": "train",
                "offset": offset,
                "length": min(25, 758 - offset),
                "revision": DATASET_COMMIT,
            }
            for attempt in range(5):
                try:
                    response = session.get(ROWS_API, params=params, timeout=(20, 90))
                    response.raise_for_status()
                    page = response.json().get("rows")
                    if not isinstance(page, list):
                        raise ValueError(
                            f"Dataset service returned no row list at offset {offset}."
                        )
                    rows.extend(page)
                    break
                except requests.RequestException:
                    if attempt == 4:
                        raise
                    time.sleep(2**attempt)
    if len(rows) != 758:
        raise ValueError(f"Expected 758 external source rows, got {len(rows)}.")
    return rows


def _prepare_row(
    item: dict[str, Any], output_dir: Path
) -> tuple[dict[str, Any] | None, str]:
    row_index = item.get("row_idx")
    row = item.get("row")
    if not isinstance(row, dict):
        raise ValueError(f"External dataset row {row_index} is malformed.")
    objects = row.get("objects")
    if not isinstance(objects, dict):
        raise ValueError(f"External dataset row {row_index} has no object annotations.")
    labels = objects.get("categories")
    boxes = objects.get("bbox")
    if not isinstance(labels, list) or not isinstance(boxes, list) or len(labels) != len(boxes):
        raise ValueError(f"External dataset row {row_index} has invalid annotations.")
    if not labels or not any(class_id in AVOCADO_SOURCE_LABELS for class_id in labels):
        return None, "not_avocado"
    if len(labels) != 1 or labels[0] not in AVOCADO_SOURCE_LABELS:
        return None, "mixed_or_multiple_objects"

    image = row.get("image")
    if not isinstance(image, dict):
        raise ValueError(f"External dataset row {row_index} has no image metadata.")
    image_width = int(image.get("width", 0))
    image_height = int(image.get("height", 0))
    resolved = resolve_external_box(boxes[0], image_width, image_height)
    if resolved is None:
        return None, "ambiguous_or_invalid_box"

    image_url = image.get("src")
    parsed_url = urlparse(image_url if isinstance(image_url, str) else "")
    if parsed_url.scheme != "https" or parsed_url.hostname not in {
        "datasets-server.huggingface.co",
        "huggingface.co",
    }:
        raise ValueError(f"Unexpected image URL in external row {row_index}.")

    class_id = int(labels[0])
    stage = _source_stage(class_id)
    if stage is None:
        raise ValueError(f"Unsupported avocado class ID {class_id} in row {row_index}.")
    record = {
        "row_index": int(row_index),
        "source_class_id": class_id,
        "label": stage,
        "image_url": image_url,
        "image_width": image_width,
        "image_height": image_height,
        "bbox_xyxy": resolved[0],
        "bbox_handling": resolved[1],
        "destination": output_dir / stage / f"row_{row_index:04d}.jpg",
    }
    return record, "selected"


def _download_and_crop(record: dict[str, Any]) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = requests.get(record["image_url"], timeout=(20, 120))
            response.raise_for_status()
            source_bytes = response.content
            with Image.open(BytesIO(source_bytes)) as opened:
                opened.load()
                orientation = opened.getexif().get(274, 1)
                if orientation != 1:
                    raise ValueError(
                        f"Unsupported EXIF orientation {orientation} in source row "
                        f"{record['row_index']}."
                    )
                image = opened.convert("RGB")
            if image.size != (record["image_width"], record["image_height"]):
                raise ValueError(
                    f"Image dimensions changed for source row {record['row_index']}: "
                    f"expected {(record['image_width'], record['image_height'])}, got {image.size}."
                )
            crop = image.crop(record["bbox_xyxy"])
            if crop.width < 64 or crop.height < 64:
                raise ValueError(
                    f"External avocado crop is too small in row {record['row_index']}."
                )
            destination: Path = record["destination"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            crop.save(destination, format="JPEG", quality=95)
            crop_hash = hashlib.sha256(crop.tobytes()).hexdigest()
            return {
                "row_index": record["row_index"],
                "source_class_id": record["source_class_id"],
                "label": record["label"],
                "crop_path": str(destination),
                "crop_sha256": crop_hash,
                "source_image_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "bbox_xyxy": record["bbox_xyxy"],
                "bbox_handling": record["bbox_handling"],
                "crop_width": crop.width,
                "crop_height": crop.height,
            }
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 4:
                time.sleep(2**attempt)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Could not download external image row {record['row_index']}.")


def prepare_external_avocado_test(
    output_dir: str | Path = DEFAULT_DATA_DIR,
) -> dict[str, Any]:
    """Build a test-only avocado crop set from the independent AgML image source."""
    root = Path(output_dir)
    manifest_path = root / "source_manifest.json"
    if manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read existing external manifest: {exc}") from exc
        if (
            previous.get("dataset_id") == DATASET_ID
            and previous.get("dataset_commit") == DATASET_COMMIT
            and previous.get("role") == "external_test_only"
        ):
            return previous
        raise FileExistsError(f"Existing external data at {root} has another source.")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty external data at {root}.")

    rows = _fetch_rows()
    skipped: Counter[str] = Counter()
    selected = []
    for item in rows:
        record, reason = _prepare_row(item, root)
        skipped[reason] += 1
        if record is not None:
            selected.append(record)
    if not selected:
        raise ValueError("No unambiguous avocado-only external test crops were found.")

    downloaded: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(_download_and_crop, record) for record in selected]
        for index, future in enumerate(as_completed(futures), start=1):
            downloaded.append(future.result())
            if index % 50 == 0 or index == len(futures):
                print(f"Prepared {index}/{len(futures)} avocado-only test crops.")

    unique_hashes: set[str] = set()
    records: list[dict[str, Any]] = []
    for record in sorted(downloaded, key=lambda value: value["row_index"]):
        if record["crop_sha256"] in unique_hashes:
            Path(record["crop_path"]).unlink()
            continue
        unique_hashes.add(record["crop_sha256"])
        record["duplicate_crop"] = False
        records.append(record)

    class_counts = dict(Counter(record["label"] for record in records))
    if set(class_counts) != set(STAGE_NAMES):
        raise ValueError(
            f"External test data must cover all three supported stages; got {class_counts}."
        )
    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_commit": DATASET_COMMIT,
        "paper_doi": "10.1016/j.dib.2025.111663",
        "mendeley_record": "https://data.mendeley.com/datasets/zysvgmxcyz/1",
        "dataset_card": (
            "https://huggingface.co/datasets/"
            "Project-AgML/strawberry_avocado_ripeness_detection"
        ),
        "license": "CC BY 4.0",
        "role": "external_test_only",
        "source_split": "single publisher train split; not used for model training",
        "source_rows": len(rows),
        "selected_avocado_crops": len(records),
        "skipped_rows": dict(sorted(skipped.items())),
        "ambiguous_or_invalid_boxes_excluded": skipped["ambiguous_or_invalid_box"],
        "stage_crosswalk": {
            "4": "unripe",
            "5": "ripe (ripening stage merged into the app's ripe class)",
            "6": "ripe",
            "7": "overripe (source wording may mean rotten)",
        },
        "stage_counts": class_counts,
        "bbox_handling_counts": dict(
            sorted(Counter(record["bbox_handling"] for record in records).items())
        ),
        "exact_duplicate_crops_excluded": len(downloaded) - len(records),
        "data": records,
        "notes": [
            "Only single-object avocado source labels 4-7 were retained; strawberry-only and mixed/multiple-object rows were excluded.",
            "The hosted dataset card exposes numeric IDs rather than semantic class names. The stage crosswalk was visually checked against the published four-stage scope and sample crops; it is not an explicit label map in the hosted card.",
            "Portrait boxes were corrected only when exactly one of the source and aspect-corrected boxes fits the image. Ambiguous or invalid boxes were excluded.",
            "This independent collection is from Maharashtra, India and includes farm/market imagery; it is not a multi-farm/camera benchmark and some backgrounds are studio-like.",
            "Partially ripe and ripe source categories are grouped as ripe; this is a coarser mapping than the source labels.",
        ],
    }
    root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _classification_metrics(
    truth: list[str], predictions: list[str]
) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_precision": float(
            precision_score(
                truth, predictions, labels=STAGE_NAMES, average="macro", zero_division=0
            )
        ),
        "macro_recall": float(
            recall_score(
                truth, predictions, labels=STAGE_NAMES, average="macro", zero_division=0
            )
        ),
        "macro_f1": float(
            f1_score(
                truth, predictions, labels=STAGE_NAMES, average="macro", zero_division=0
            )
        ),
        "weighted_f1": float(
            f1_score(
                truth, predictions, labels=STAGE_NAMES, average="weighted", zero_division=0
            )
        ),
        "confusion_matrix": confusion_matrix(
            truth, predictions, labels=STAGE_NAMES
        ).tolist(),
        "per_class": classification_report(
            truth,
            predictions,
            labels=STAGE_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }


def evaluate_external_avocado_test(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    model_dir: str | Path = MODEL_DIR,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate the saved crop classifiers without fitting or tuning on external data."""
    root = Path(data_dir)
    manifest_path = root / "source_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"External test data not found at {root}. "
            "Run this script without --evaluate-only first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("role") != "external_test_only" or manifest.get("dataset_id") != DATASET_ID:
        raise ValueError(f"Unexpected external test manifest in {manifest_path}.")

    manifest_data = manifest.get("data")
    if not isinstance(manifest_data, list):
        raise ValueError("External test manifest has no crop-record list.")
    records: list[ImageRecord] = []
    expected_paths: set[Path] = set()
    for item in manifest_data:
        if (
            not isinstance(item, dict)
            or item.get("label") not in STAGE_NAMES
            or not isinstance(item.get("crop_path"), str)
        ):
            raise ValueError("External test manifest contains an invalid crop record.")
        filename = Path(item["crop_path"]).name
        if not filename or filename in {".", ".."}:
            raise ValueError("External test manifest contains an invalid crop path.")
        path = root / item["label"] / filename
        expected_paths.add(path.resolve())
        records.append(
            ImageRecord(path=path, split="external_test", label=item["label"])
        )
    if len(records) != manifest.get("selected_avocado_crops"):
        raise ValueError(
            "External test image count does not match its provenance manifest."
        )
    actual_paths = {
        path.resolve()
        for path in root.rglob("*.jpg")
    }
    if actual_paths != expected_paths:
        raise ValueError(
            "External test files do not match the avocado-only provenance manifest."
        )
    if not records:
        raise ValueError("External test data contains no avocado crops.")
    missing = [
        str(record.path)
        for record in records
        if not record.path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing external test crop: {missing[0]}")

    svm_path = Path(model_dir) / "classical.joblib"
    cnn_path = Path(model_dir) / "compact_cnn.pt"
    if not svm_path.is_file() or not cnn_path.is_file():
        raise FileNotFoundError(
            f"Both trained crop models are required in {Path(model_dir)}."
        )
    bundle = joblib.load(svm_path)
    svm = bundle["model"]
    feature_rows = []
    for record in records:
        with Image.open(record.path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            feature_rows.append(extract_features(image))
    truth = [record.label for record in records]
    svm_predictions = svm.predict(np.stack(feature_rows)).tolist()
    cnn_predictions, cnn_ms_per_image = predict_cnn(cnn_path, records)

    result = {
        "dataset_id": DATASET_ID,
        "dataset_commit": DATASET_COMMIT,
        "role": "external_test_only",
        "sample_count": len(records),
        "class_counts": dict(Counter(truth)),
        "stage_crosswalk": manifest["stage_crosswalk"],
        "models": {
            "classical_svm": _classification_metrics(truth, svm_predictions),
            "compact_cnn": _classification_metrics(truth, cnn_predictions),
        },
        "cnn_inference_ms_per_image": float(cnn_ms_per_image),
        "interpretation": (
            "Independent-source crop-classification evidence only. "
            "The models were not fitted or tuned on these images."
        ),
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and evaluate an avocado-only external test set."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.evaluate_only:
        manifest_path = args.data_dir / "source_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"External test data not found at {args.data_dir}; "
                "prepare it by running this script without --evaluate-only."
            )
    else:
        manifest = prepare_external_avocado_test(args.data_dir)
        print(
            f"Prepared {manifest['selected_avocado_crops']} avocado-only external test crops."
        )
    results = evaluate_external_avocado_test(
        args.data_dir,
        args.model_dir,
        args.output or args.model_dir / "external_test_metrics.json",
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
