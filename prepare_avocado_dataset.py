from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import requests
from PIL import Image

DATASET_REF = "deivdf/avocado-ripeness-yolo-format"
DATASET_URL = f"https://www.kaggle.com/datasets/{DATASET_REF}"
DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET_REF}"
DATASET_VERSION = 1
DATASET_SIZE_BYTES = 360_908_181
LICENSE = "MIT"
SOURCE_CLASS_NAMES = ("ripe", "overripe", "unripe", "avocado")
RIPENESS_CLASS_NAMES = {
    0: "ripe",
    1: "overripe",
    2: "unripe",
}
SPLIT_NAMES = {"train": "train", "val": "validation", "test": "test"}


def parse_yolo_labels(
    text: str, class_count: int = len(SOURCE_CLASS_NAMES)
) -> list[tuple[int, float, float, float, float]]:
    annotations = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Invalid YOLO annotation on line {line_number}: {line!r}")
        try:
            class_id = int(fields[0])
            center_x, center_y, width, height = map(float, fields[1:])
        except ValueError as exc:
            raise ValueError(
                f"Invalid YOLO annotation on line {line_number}: {line!r}"
            ) from exc
        if (
            class_id not in range(class_count)
            or not all(
                math.isfinite(value)
                for value in (center_x, center_y, width, height)
            )
            or not 0 <= center_x <= 1
            or not 0 <= center_y <= 1
            or not 0 < width <= 1
            or not 0 < height <= 1
        ):
            raise ValueError(f"Out-of-range YOLO annotation on line {line_number}.")
        annotations.append((class_id, center_x, center_y, width, height))
    return annotations


def parse_source_class_names(yaml_text: str) -> tuple[str, ...]:
    lines = yaml_text.splitlines()
    try:
        names_index = lines.index("names:")
    except ValueError as exc:
        raise ValueError("The Kaggle data.yaml file is missing its names list.") from exc
    class_names = []
    for line in lines[names_index + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("- "):
            break
        class_names.append(stripped[2:].strip().strip("'\""))
    names_line = next(
        (line.split(":", 1)[1].strip() for line in lines if line.startswith("nc:")),
        None,
    )
    if names_line != str(len(class_names)):
        raise ValueError("The Kaggle class count does not match its names list.")
    if tuple(class_names) != SOURCE_CLASS_NAMES:
        raise ValueError(
            "The Kaggle dataset class schema changed. "
            f"Expected {SOURCE_CLASS_NAMES}; got {tuple(class_names)}."
        )
    return tuple(class_names)


def normalized_bbox_to_pixels(
    bbox: tuple[int, float, float, float, float], image_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    _, center_x, center_y, width, height = bbox
    image_width, image_height = image_size[0], image_size[1]
    left = max(0, int((center_x - width / 2) * image_width))
    top = max(0, int((center_y - height / 2) * image_height))
    right = min(image_width, math.ceil((center_x + width / 2) * image_width))
    bottom = min(image_height, math.ceil((center_y + height / 2) * image_height))
    if right <= left or bottom <= top:
        raise ValueError("An avocado bounding box produces an empty crop.")
    return left, top, right, bottom


def _download_archive(archive_path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    written = 0
    for attempt in range(4):
        try:
            with requests.get(
                DOWNLOAD_URL,
                params={"datasetVersionNumber": DATASET_VERSION},
                stream=True,
                timeout=(30, 120),
            ) as response:
                response.raise_for_status()
                content_length = int(response.headers.get("content-length", 0))
                if content_length != DATASET_SIZE_BYTES:
                    raise ValueError(
                        "Unexpected Kaggle archive size: "
                        f"expected {DATASET_SIZE_BYTES}, got {content_length}."
                    )
                with archive_path.open("wb") as archive:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            archive.write(chunk)
                            digest.update(chunk)
                            written += len(chunk)
                            if written % (50 * 1024 * 1024) < 1024 * 1024:
                                print(
                                    f"Downloaded {written // (1024 * 1024)} MiB "
                                    f"of {content_length // (1024 * 1024)} MiB."
                                )
            if written != DATASET_SIZE_BYTES:
                raise ValueError(
                    f"Incomplete Kaggle archive: expected {DATASET_SIZE_BYTES} "
                    f"bytes, got {written}."
                )
            return written, digest.hexdigest()
        except requests.RequestException:
            if attempt == 3:
                raise
            written = 0
            digest = hashlib.sha256()
            time.sleep(2**attempt)
    raise RuntimeError("Could not download the Kaggle avocado dataset.")


def _safe_archive_names(archive: zipfile.ZipFile) -> list[str]:
    names = []
    for entry in archive.infolist():
        path = PurePosixPath(entry.filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe path in dataset archive: {entry.filename}")
        names.append(entry.filename)
    return names


def prepare_dataset(
    output_dir: str | Path = "data/avocado_pure",
    max_side: int = 512,
) -> dict[str, Any]:
    """Download the avocado-only Kaggle set and save stage-labeled avocado crops."""
    if max_side < 64:
        raise ValueError("max_side must be at least 64 pixels.")
    root = Path(output_dir)
    manifest_path = root / "source_manifest.json"
    marker_path = root / ".avocado_dataset_preparation"
    if (
        root.exists()
        and any(root.iterdir())
        and not manifest_path.is_file()
        and not marker_path.is_file()
    ):
        raise FileExistsError(
            f"{root} is not empty or a previously prepared avocado-only dataset."
        )
    if manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read existing dataset manifest: {exc}") from exc
        if (
            previous.get("dataset_ref") != DATASET_REF
            or previous.get("dataset_version") != DATASET_VERSION
        ):
            raise ValueError(f"Existing manifest in {root} is for another dataset.")

    root.mkdir(parents=True, exist_ok=True)
    marker_path.write_text("Dataset preparation in progress.", encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="avocado_kaggle_") as temp_dir:
        archive_path = Path(temp_dir) / "dataset.zip"
        archive_size, archive_sha256 = _download_archive(archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            names = _safe_archive_names(archive)
            if archive.testzip() is not None:
                raise ValueError("The Kaggle dataset ZIP failed its CRC integrity check.")
            source_class_names = parse_source_class_names(
                archive.read("data.yaml").decode("utf-8")
            )
            image_names: dict[str, list[str]] = {}
            member_set = set(names)
            for source_split in SPLIT_NAMES:
                prefix = f"{source_split}/images/"
                image_names[source_split] = sorted(
                    name
                    for name in names
                    if name.startswith(prefix)
                    and PurePosixPath(name).suffix.lower()
                    in {".jpg", ".jpeg", ".png"}
                )
                if not image_names[source_split]:
                    raise ValueError(f"The Kaggle archive has no {source_split} images.")

            split_counts: dict[str, Counter[str]] = {
                split: Counter() for split in SPLIT_NAMES.values()
            }
            source_split_image_counts: Counter[str] = Counter()
            source_class_counts: dict[str, Counter[str]] = {
                split: Counter() for split in SPLIT_NAMES
            }
            image_hashes: dict[str, set[str]] = defaultdict(set)
            source_hash_splits: dict[str, set[str]] = defaultdict(set)
            duplicate_source_hashes: set[str] = set()
            sample_records: list[dict[str, Any]] = []
            ignored_generic_avocado_boxes = 0
            empty_label_images = 0
            missing_label_images: list[str] = []

            for source_split, images in image_names.items():
                output_split = SPLIT_NAMES[source_split]
                source_split_image_counts[output_split] = len(images)
                for image_index, image_name in enumerate(images, 1):
                    label_name = (
                        f"{source_split}/labels/"
                        f"{PurePosixPath(image_name).stem}.txt"
                    )
                    if label_name not in member_set:
                        missing_label_images.append(image_name)
                        continue
                    image_bytes = archive.read(image_name)
                    source_sha256 = hashlib.sha256(image_bytes).hexdigest()
                    source_hash_splits[source_sha256].add(output_split)
                    if source_sha256 in image_hashes[output_split]:
                        duplicate_source_hashes.add(source_sha256)
                    image_hashes[output_split].add(source_sha256)
                    label_text = archive.read(label_name).decode("utf-8")
                    annotations = parse_yolo_labels(label_text, len(source_class_names))
                    if not annotations:
                        empty_label_images += 1
                    with Image.open(io.BytesIO(image_bytes)) as opened:
                        image = opened.convert("RGB")
                        for annotation_index, annotation in enumerate(annotations):
                            class_id = annotation[0]
                            source_class_counts[source_split][source_class_names[class_id]] += 1
                            stage = RIPENESS_CLASS_NAMES.get(class_id)
                            if stage is None:
                                ignored_generic_avocado_boxes += 1
                                continue
                            crop_box = normalized_bbox_to_pixels(
                                annotation, image.size
                            )
                            crop = image.crop(crop_box)
                            crop_width, crop_height = crop.size
                            crop.thumbnail(
                                (max_side, max_side), Image.Resampling.LANCZOS
                            )
                            stage_dir = root / output_split / stage
                            stage_dir.mkdir(parents=True, exist_ok=True)
                            target = stage_dir / (
                                f"{PurePosixPath(image_name).stem}"
                                f"__object_{annotation_index:03d}.jpg"
                            )
                            crop.save(
                                target, format="JPEG", quality=92, optimize=True
                            )
                            split_counts[output_split][stage] += 1
                            sample_records.append(
                                {
                                    "source_image": image_name,
                                    "source_label": label_name,
                                    "source_sha256": source_sha256,
                                    "stage": stage,
                                    "source_class_id": class_id,
                                    "annotation_index": annotation_index,
                                    "bbox_xywh_normalized": list(annotation[1:]),
                                    "crop_xyxy_pixels": list(crop_box),
                                    "crop_size": [crop_width, crop_height],
                                    "split": output_split,
                                    "prepared_file": str(
                                        target.relative_to(root).as_posix()
                                    ),
                                }
                            )
                    if image_index % 500 == 0 or image_index == len(images):
                        print(
                            f"Prepared source images {image_index}/{len(images)} "
                            f"from {source_split}."
                        )

            if missing_label_images:
                raise ValueError(
                    f"{len(missing_label_images)} dataset images have no label file; "
                    f"examples: {missing_label_images[:5]}"
                )
            cross_split_hashes = {
                digest for digest, splits in source_hash_splits.items() if len(splits) > 1
            }
            if cross_split_hashes:
                raise ValueError(
                    f"{len(cross_split_hashes)} identical source images cross "
                    "the published train/validation/test splits."
                )
            if any(not split_counts[split] for split in SPLIT_NAMES.values()):
                raise ValueError("One of the prepared splits has no stage-labeled crops.")
            if any(
                set(counts) != set(RIPENESS_CLASS_NAMES.values())
                for counts in split_counts.values()
            ):
                raise ValueError("Each published split must contain all three stages.")

            report = {
                "dataset": "Avocado Ripeness YOLO Format",
                "dataset_ref": DATASET_REF,
                "dataset_uploader": "David Fernando Hernandez Cuero",
                "dataset_version": DATASET_VERSION,
                "dataset_url": DATASET_URL,
                "license": LICENSE,
                "archive_size_bytes": archive_size,
                "archive_sha256": archive_sha256,
                "dataset_class_mapping": {
                    str(class_id): name
                    for class_id, name in enumerate(source_class_names)
                },
                "ripeness_classes_used": RIPENESS_CLASS_NAMES,
                "avocado_only": True,
                "only_avocado_annotations_present": all(
                    name in {"ripe", "overripe", "unripe", "avocado"}
                    for name in source_class_names
                ),
                "ripeness_crop_policy": (
                    "Crops only boxes labeled ripe, overripe, or unripe; "
                    "generic avocado boxes are excluded from stage classification."
                ),
                "source_images_per_split": dict(source_split_image_counts),
                "prepared_crops_per_split": {
                    split: dict(sorted(counts.items()))
                    for split, counts in split_counts.items()
                },
                "source_annotation_counts": {
                    split: dict(sorted(counts.items()))
                    for split, counts in source_class_counts.items()
                },
                "prepared_crop_count": len(sample_records),
                "ignored_generic_avocado_boxes": ignored_generic_avocado_boxes,
                "empty_label_images": empty_label_images,
                "exact_duplicate_source_images": len(duplicate_source_hashes),
                "cross_split_duplicate_source_images": len(cross_split_hashes),
                "max_image_side": max_side,
                "split_source": "The dataset author's published train/val/test partitions.",
                "samples": sample_records,
            }
            manifest_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            marker_path.unlink(missing_ok=True)
            return report


def prepare_detection_dataset(
    output_dir: str | Path = "data/avocado_detection",
) -> dict[str, Any]:
    """Prepare full-scene YOLO data using only avocado source annotations."""
    root = Path(output_dir)
    manifest_path = root / "source_manifest.json"
    marker_path = root / ".avocado_detection_preparation"
    if (
        root.exists()
        and any(root.iterdir())
        and not manifest_path.is_file()
        and not marker_path.is_file()
    ):
        raise FileExistsError(
            f"{root} is not empty or a previously prepared avocado detector dataset."
        )
    if manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read existing detector manifest: {exc}") from exc
        if (
            previous.get("dataset_ref") != DATASET_REF
            or previous.get("dataset_version") != DATASET_VERSION
        ):
            raise ValueError(f"Existing manifest in {root} is for another dataset.")
        (root / "data.yaml").write_text(
            f"path: {json.dumps(str(root.resolve()))}\n"
            "train: images/train\n"
            "val: images/validation\n"
            "test: images/test\n"
            "names:\n"
            "  0: ripe\n"
            "  1: overripe\n"
            "  2: unripe\n"
            "  3: avocado\n",
            encoding="utf-8",
        )
        return previous

    root.mkdir(parents=True, exist_ok=True)
    marker_path.write_text("Detection dataset preparation in progress.", encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="avocado_detector_") as temp_dir:
        archive_path = Path(temp_dir) / "dataset.zip"
        archive_size, archive_sha256 = _download_archive(archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            names = _safe_archive_names(archive)
            if archive.testzip() is not None:
                raise ValueError("The Kaggle dataset ZIP failed its CRC integrity check.")
            source_class_names = parse_source_class_names(
                archive.read("data.yaml").decode("utf-8")
            )
            members = set(names)
            source_image_hashes: dict[str, set[str]] = defaultdict(set)
            split_counts: dict[str, Counter[str]] = {
                split: Counter() for split in SPLIT_NAMES.values()
            }
            image_counts: Counter[str] = Counter()
            missing_labels: list[str] = []
            duplicate_names: set[tuple[str, str]] = set()
            for source_split, output_split in SPLIT_NAMES.items():
                image_prefix = f"{source_split}/images/"
                image_names = sorted(
                    name
                    for name in names
                    if name.startswith(image_prefix)
                    and PurePosixPath(name).suffix.lower()
                    in {".jpg", ".jpeg", ".png"}
                )
                if not image_names:
                    raise ValueError(f"The archive has no {source_split} images.")
                for image_name in image_names:
                    source_path = PurePosixPath(image_name)
                    label_name = (
                        f"{source_split}/labels/{source_path.stem}.txt"
                    )
                    if label_name not in members:
                        missing_labels.append(image_name)
                        continue
                    label_text = archive.read(label_name).decode("utf-8")
                    annotations = parse_yolo_labels(
                        label_text, len(source_class_names)
                    )
                    if not annotations:
                        continue
                    image_bytes = archive.read(image_name)
                    image_hash = hashlib.sha256(image_bytes).hexdigest()
                    source_image_hashes[image_hash].add(output_split)
                    output_image_dir = root / "images" / output_split
                    output_label_dir = root / "labels" / output_split
                    output_image_dir.mkdir(parents=True, exist_ok=True)
                    output_label_dir.mkdir(parents=True, exist_ok=True)
                    output_stem = source_path.stem
                    suffix = source_path.suffix.lower()
                    output_image = output_image_dir / f"{output_stem}{suffix}"
                    output_label = output_label_dir / f"{output_stem}.txt"
                    name_key = (output_split, output_stem)
                    if name_key in duplicate_names or output_image.exists():
                        raise ValueError(
                            "The source contains duplicate image names within "
                            f"the {output_split} split: {output_stem}"
                        )
                    duplicate_names.add(name_key)
                    output_image.write_bytes(image_bytes)
                    output_label.write_text(
                        "".join(
                            f"{class_id} {center_x:.8f} {center_y:.8f} "
                            f"{width:.8f} {height:.8f}\n"
                            for class_id, center_x, center_y, width, height
                            in annotations
                        ),
                        encoding="utf-8",
                    )
                    image_counts[output_split] += 1
                    for class_id, *_ in annotations:
                        split_counts[output_split][source_class_names[class_id]] += 1

            if missing_labels:
                raise ValueError(
                    f"{len(missing_labels)} dataset images have no label file; "
                    f"examples: {missing_labels[:5]}"
                )
            cross_split_hashes = {
                digest for digest, splits in source_image_hashes.items() if len(splits) > 1
            }
            if cross_split_hashes:
                raise ValueError(
                    f"{len(cross_split_hashes)} identical source images cross "
                    "the published train/validation/test splits."
                )
            if any(image_counts[split] == 0 for split in SPLIT_NAMES.values()):
                raise ValueError(
                    "One of the published splits has no stage-labeled avocado images."
                )
            dataset_yaml = (
                f"path: {json.dumps(str(root.resolve()))}\n"
                "train: images/train\n"
                "val: images/validation\n"
                "test: images/test\n"
                "names:\n"
                "  0: ripe\n"
                "  1: overripe\n"
                "  2: unripe\n"
                "  3: avocado\n"
            )
            (root / "data.yaml").write_text(dataset_yaml, encoding="utf-8")
            manifest = {
                "dataset": "Avocado Ripeness YOLO Format",
                "dataset_ref": DATASET_REF,
                "dataset_version": DATASET_VERSION,
                "dataset_url": DATASET_URL,
                "license": LICENSE,
                "archive_size_bytes": archive_size,
                "archive_sha256": archive_sha256,
                "source_class_mapping": {
                    str(class_id): name
                    for class_id, name in enumerate(source_class_names)
                },
                "detector_class_mapping": {
                    str(class_id): name
                    for class_id, name in enumerate(source_class_names)
                },
                "avocado_only": True,
                "annotation_policy": (
                    "Full source scenes are retained when they contain at least "
                    "one avocado annotation. The generic avocado class means "
                    "the source did not provide a ripeness stage."
                ),
                "source_images_per_split": dict(image_counts),
                "annotation_counts_per_split": {
                    split: dict(sorted(counts.items()))
                    for split, counts in split_counts.items()
                },
                "identical_source_images_across_splits": len(cross_split_hashes),
                "split_source": "The dataset author's published train/val/test partitions.",
            }
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            marker_path.unlink(missing_ok=True)
            return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare avocado-only ripeness crops from the public Kaggle dataset."
    )
    parser.add_argument("--output-dir", default="data/avocado_pure")
    parser.add_argument("--max-side", type=int, default=512)
    args = parser.parse_args()
    result = prepare_dataset(args.output_dir, args.max_side)
    print(
        json.dumps(
            {
                "dataset_url": result["dataset_url"],
                "prepared_crop_count": result["prepared_crop_count"],
                "prepared_crops_per_split": result["prepared_crops_per_split"],
                "ignored_generic_avocado_boxes": result[
                    "ignored_generic_avocado_boxes"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
