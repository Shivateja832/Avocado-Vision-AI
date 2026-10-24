from __future__ import annotations

from pathlib import Path
import hashlib
import io
import json
import sys
from types import ModuleType, SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest
import torch
from avocado_vision import detector
from PIL import Image

from avocado_vision.cnn import CompactCNN, predict_cnn, train_cnn
from avocado_vision.pipeline import (
    ImageRecord,
    analyze_dataset,
    extract_features,
    load_dataset_splits,
    load_records,
)
from prepare_avocado_dataset import (
    parse_source_class_names,
    parse_yolo_labels,
    prepare_detection_dataset,
    prepare_dataset,
)
from train_detector import train_detector


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    for split in ("train", "test"):
        for class_index, label in enumerate(("class_a", "class_b")):
            folder = tmp_path / split / label
            folder.mkdir(parents=True)
            for image_index in range(2 if split == "train" else 1):
                color = (220, 20, 20) if class_index == 0 else (20, 20, 220)
                Image.new("RGB", (80 + image_index, 64), color).save(
                    folder / f"{label}_{image_index}.png"
                )
    return tmp_path


def test_load_records_and_analysis_report_counts(tiny_dataset: Path) -> None:
    train, test = load_records(tiny_dataset)
    report = analyze_dataset(tiny_dataset)
    assert len(train) == 4
    assert len(test) == 2
    assert report["total_images"] == 6
    assert report["valid_images"] == 6
    assert report["class_counts"]["class_a"] == {"train": 2, "test": 1}
    assert report["unreadable_count"] == 0
    assert report["validation_images"] == 0


def test_load_dataset_splits_includes_validation(tiny_dataset: Path) -> None:
    for class_index, label in enumerate(("class_a", "class_b")):
        folder = tiny_dataset / "validation" / label
        folder.mkdir(parents=True)
        Image.new("RGB", (64, 64), (class_index * 100, 0, 0)).save(
            folder / "validation.png"
        )

    splits = load_dataset_splits(tiny_dataset)
    report = analyze_dataset(tiny_dataset)
    assert len(splits["validation"]) == 2
    assert report["validation_images"] == 2
    assert report["class_counts"]["class_a"] == {
        "train": 2,
        "validation": 1,
        "test": 1,
    }


def test_yolo_annotation_parser_rejects_invalid_values() -> None:
    assert parse_yolo_labels("3 0.5 0.5 0.2 0.3") == [(3, 0.5, 0.5, 0.2, 0.3)]
    with pytest.raises(ValueError, match="Out-of-range"):
        parse_yolo_labels("4 0.5 0.5 0.2 0.3")
    with pytest.raises(ValueError, match="Invalid YOLO annotation"):
        parse_yolo_labels("4 0.5 0.5 0.2")


def test_source_yaml_is_exactly_the_avocado_dataset_schema() -> None:
    class_names = parse_source_class_names(
        "path: /dataset\nnc: 4\nnames:\n- ripe\n- overripe\n- unripe\n- avocado\n"
    )
    assert class_names == ("ripe", "overripe", "unripe", "avocado")
    with pytest.raises(ValueError, match="schema changed"):
        parse_source_class_names(
            "path: /dataset\nnc: 2\nnames:\n- apple\n- avocado\n"
        )


def test_prepare_dataset_only_writes_stage_labeled_avocado_crops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = tmp_path / "source.zip"
    source_classes = ("ripe", "overripe", "unripe", "avocado")
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "data.yaml",
            "path: /dataset\ntrain: train/images\nval: val/images\n"
            "test: test/images\nnc: 4\nnames:\n- ripe\n- overripe\n"
            "- unripe\n- avocado\n",
        )
        for split_index, split in enumerate(("train", "val", "test")):
            for class_id, class_name in enumerate(source_classes):
                image_path = f"{split}/images/{class_name}.png"
                label_path = f"{split}/labels/{class_name}.txt"
                image = Image.new(
                    "RGB", (40, 40), (class_id * 40, split_index * 40, 10)
                )
                image_bytes = io.BytesIO()
                image.save(image_bytes, format="PNG")
                archive.writestr(image_path, image_bytes.getvalue())
                if class_id == 3:
                    annotation = "3 0.5 0.5 0.5 0.5\n"
                else:
                    annotation = f"{class_id} 0.5 0.5 0.5 0.5\n"
                archive.writestr(label_path, annotation)

    archive_size = archive_path.stat().st_size
    archive_digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()

    def mock_download(target: Path) -> tuple[int, str]:
        target.write_bytes(archive_path.read_bytes())
        return archive_size, archive_digest

    monkeypatch.setattr("prepare_avocado_dataset._download_archive", mock_download)
    result = prepare_dataset(tmp_path / "prepared", max_side=64)

    assert result["prepared_crop_count"] == 9
    assert result["ignored_generic_avocado_boxes"] == 3
    assert set(result["prepared_crops_per_split"]) == {
        "train",
        "validation",
        "test",
    }
    assert all(
        not (tmp_path / "prepared" / split / "avocado").exists()
        for split in ("train", "validation", "test")
    )
    assert len(list((tmp_path / "prepared").glob("*/*/*.jpg"))) == 9
    assert result["archive_size_bytes"] == archive_size


def test_prepare_detection_dataset_keeps_only_avocado_classes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = tmp_path / "source.zip"
    source_classes = ("ripe", "overripe", "unripe", "avocado")
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "data.yaml",
            "path: /dataset\nnc: 4\nnames:\n- ripe\n- overripe\n"
            "- unripe\n- avocado\n",
        )
        for split_index, split in enumerate(("train", "val", "test")):
            for class_id, class_name in enumerate(source_classes):
                stem = f"{class_name}_{split}"
                image_bytes = io.BytesIO()
                Image.new(
                    "RGB", (40, 40), (class_id * 40, split_index * 40, 10)
                ).save(
                    image_bytes, format="PNG"
                )
                archive.writestr(
                    f"{split}/images/{stem}.png", image_bytes.getvalue()
                )
                archive.writestr(
                    f"{split}/labels/{stem}.txt",
                    f"{class_id} 0.5 0.5 0.5 0.5\n",
                )

    archive_size = archive_path.stat().st_size
    archive_digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()

    def mock_download(target: Path) -> tuple[int, str]:
        target.write_bytes(archive_path.read_bytes())
        return archive_size, archive_digest

    monkeypatch.setattr("prepare_avocado_dataset._download_archive", mock_download)
    output = tmp_path / "detector_data"
    report = prepare_detection_dataset(output)

    assert report["avocado_only"] is True
    assert report["identical_source_images_across_splits"] == 0
    assert report["source_images_per_split"] == {
        "train": 4,
        "validation": 4,
        "test": 4,
    }
    assert report["detector_class_mapping"] == {
        "0": "ripe",
        "1": "overripe",
        "2": "unripe",
        "3": "avocado",
    }
    label_files = list((output / "labels").rglob("*.txt"))
    assert len(label_files) == 12
    assert all(
        all(int(line.split()[0]) in {0, 1, 2, 3} for line in path.read_text().splitlines())
        for path in label_files
    )
    assert "3: avocado" in (output / "data.yaml").read_text()
    detector_yaml = (output / "data.yaml").read_text()
    assert json.loads(detector_yaml.splitlines()[0].split(": ", 1)[1]) == str(
        output.resolve()
    )
    assert json.loads((output / "source_manifest.json").read_text())[
        "source_images_per_split"
    ] == report["source_images_per_split"]


def test_full_scene_detector_returns_stage_and_unknown_avocado_boxes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ArrayWrapper:
        def __init__(self, values: list[float] | list[list[float]]) -> None:
            self.values = np.asarray(values)

        def cpu(self) -> ArrayWrapper:
            return self

        def numpy(self) -> np.ndarray:
            return self.values

    boxes = SimpleNamespace(
        xyxy=ArrayWrapper([[1, 2, 20, 30], [5, 6, 25, 35]]),
        cls=ArrayWrapper([1, 3]),
        conf=ArrayWrapper([0.9, 0.99]),
    )
    fake_result = SimpleNamespace(
        boxes=boxes,
        names={0: "ripe", 1: "overripe", 2: "unripe", 3: "avocado"},
        plot=lambda: np.zeros((40, 40, 3), dtype=np.uint8),
    )

    weights_path = tmp_path / "detector.pt"
    weights_path.write_bytes(b"test weights")

    def fake_load_detector(model_path: str, modified_ns: int) -> SimpleNamespace:
        assert model_path == str(weights_path.resolve())
        assert modified_ns == weights_path.stat().st_mtime_ns
        return SimpleNamespace(
            predict=lambda **kwargs: [fake_result]
            if kwargs["device"] == "cpu"
            else []
        )

    monkeypatch.setattr(
        detector,
        "_load_detector",
        fake_load_detector,
    )

    result = detector.predict_scene(
        Image.new("RGB", (40, 40)), weights_path
    )

    assert len(result["detections"]) == 2
    assert result["detections"][0]["stage"] == "overripe"
    assert result["detections"][0]["confidence"] == pytest.approx(0.9)
    assert result["detections"][1]["stage"] == "avocado (stage unknown)"
    assert result["annotated_image"].size == (40, 40)


def test_train_detector_saves_weights_and_test_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "detector_data"
    (dataset / "images" / "test").mkdir(parents=True)
    (dataset / "labels" / "test").mkdir(parents=True)
    (dataset / "images" / "train").mkdir(parents=True)
    (dataset / "labels" / "train").mkdir(parents=True)
    (dataset / "data.yaml").write_text("path: .\n")
    Image.new("RGB", (40, 40)).save(dataset / "images" / "test" / "sample.png")
    (dataset / "labels" / "test" / "sample.txt").write_text(
        "1 0.5 0.5 0.4 0.4\n"
    )
    for class_id in range(4):
        Image.new("RGB", (40, 40), (class_id * 40, 10, 10)).save(
            dataset / "images" / "train" / f"class_{class_id}.png"
        )
        (dataset / "labels" / "train" / f"class_{class_id}.txt").write_text(
            f"{class_id} 0.5 0.5 0.4 0.4\n"
        )

    class FakeYOLO:
        def __init__(self, weights: str) -> None:
            self.weights = weights
            self.names = {0: "ripe", 1: "overripe", 2: "unripe", 3: "avocado"}

        def train(self, **kwargs: object) -> None:
            weights_path = (
                Path(str(kwargs["project"]))
                / str(kwargs["name"])
                / "weights"
                / "best.pt"
            )
            weights_path.parent.mkdir(parents=True)
            weights_path.write_bytes(b"detector")
            self.trainer = SimpleNamespace(best=weights_path, epoch=0)

        def val(self, **kwargs: object) -> SimpleNamespace:
            assert kwargs["split"] == "test"
            return SimpleNamespace(
                box=SimpleNamespace(
                    mp=0.6,
                    mr=0.7,
                    map50=0.8,
                    map=0.5,
                    maps=np.asarray([0.4, 0.5, 0.6, 0.7]),
                ),
                names=self.names,
            )

        def predict(self, **kwargs: object) -> list[SimpleNamespace]:
            assert kwargs["conf"] == 0.25
            return [
                SimpleNamespace(
                    names=self.names,
                    boxes=[
                        SimpleNamespace(
                            cls=SimpleNamespace(item=lambda: 1),
                            conf=SimpleNamespace(item=lambda: 0.8),
                            xyxy=np.asarray([[2.0, 3.0, 20.0, 30.0]]),
                        )
                    ],
                )
            ]

    fake_module = ModuleType("ultralytics")
    fake_module.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)

    output = tmp_path / "detector_models"
    metrics = train_detector(
        dataset,
        output,
        epochs=1,
        image_size=64,
        batch_size=1,
        images_per_class=1,
    )

    assert (output / "best.pt").read_bytes() == b"detector"
    assert metrics["test_metrics"]["map50"] == 0.8
    assert metrics["test_metrics"]["per_class_map50"]["avocado"] == 0.7
    assert metrics["example_run"]["actual_detections"][0]["class"] == "overripe"
    assert metrics["example_run"]["predicted_detections"][0]["class"] == "overripe"
    assert metrics["training_subset"]["training_scenes"] == 4
    assert json.loads((output / "detection_metrics.json").read_text()) == metrics


def test_classical_descriptor_is_finite_and_fixed_width() -> None:
    features = extract_features(Image.new("RGB", (120, 70), (50, 120, 180)))
    assert features.shape == (560,)
    assert np.isfinite(features).all()


def test_compact_cnn_returns_one_logit_per_class() -> None:
    model = CompactCNN(class_count=4)
    logits = model(torch.zeros(2, 3, 64, 64))
    assert tuple(logits.shape) == (2, 4)


def test_cnn_training_saves_loadable_model(tmp_path: Path) -> None:
    records = []
    validation_records = []
    for class_index, label in enumerate(("class_a", "class_b")):
        folder = tmp_path / "train" / label
        folder.mkdir(parents=True)
        color = (220, 20, 20) if class_index == 0 else (20, 20, 220)
        for image_index in range(10):
            image_path = folder / f"{label}_{image_index}.png"
            Image.new("RGB", (64, 64), color).save(image_path)
            records.append(ImageRecord(image_path, "train", label))
        validation_folder = tmp_path / "validation" / label
        validation_folder.mkdir(parents=True)
        validation_path = validation_folder / "validation.png"
        Image.new("RGB", (64, 64), color).save(validation_path)
        validation_records.append(
            ImageRecord(validation_path, "validation", label)
        )

    model_path = tmp_path / "models" / "compact_cnn.pt"
    train_cnn(
        records,
        ["class_a", "class_b"],
        model_path,
        max_epochs=2,
        batch_size=8,
        validation_records=validation_records,
    )
    predictions, latency = predict_cnn(model_path, records[:4])

    assert model_path.is_file()
    assert len(predictions) == 4
    assert set(predictions) <= {"class_a", "class_b"}
    assert latency >= 0
