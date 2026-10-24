from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


@lru_cache(maxsize=2)
def _load_detector(model_path: str, _modified_ns: int) -> Any:
    from ultralytics import YOLO

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Avocado detector weights not found: {path}")
    if path.stat().st_mtime_ns != _modified_ns:
        raise RuntimeError(f"Avocado detector weights changed while loading: {path}")
    return YOLO(str(path))


def predict_scene(
    image: Image.Image,
    model_path: str | Path,
    confidence: float = 0.25,
) -> dict[str, Any]:
    """Detect and classify ripeness boxes in a full-scene image."""
    if not 0 < confidence < 1:
        raise ValueError("Detection confidence must be between 0 and 1.")
    normalized = ImageOps.exif_transpose(image).convert("RGB")
    weights = Path(model_path).resolve()
    try:
        modified_ns = weights.stat().st_mtime_ns
    except OSError as exc:
        raise FileNotFoundError(f"Avocado detector weights not found: {weights}") from exc
    detector = _load_detector(str(weights), modified_ns)
    results = detector.predict(
        source=normalized,
        conf=confidence,
        verbose=False,
        device="cpu",
    )
    if not results:
        return {"detections": [], "annotated_image": normalized}
    result = results[0]
    detections = []
    boxes = result.boxes
    if boxes is not None:
        coordinates = boxes.xyxy.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        scores = boxes.conf.cpu().numpy()
        names = result.names
        for coordinates_row, class_id, score in zip(
            coordinates, class_ids, scores, strict=True
        ):
            class_name = names[int(class_id)]
            stage = (
                class_name
                if class_name in {"ripe", "overripe", "unripe"}
                else "avocado (stage unknown)"
            )
            detections.append(
                {
                    "stage": stage,
                    "confidence": float(score),
                    "x_min": float(coordinates_row[0]),
                    "y_min": float(coordinates_row[1]),
                    "x_max": float(coordinates_row[2]),
                    "y_max": float(coordinates_row[3]),
                }
            )
    plotted = result.plot()
    annotated_image = Image.fromarray(np.asarray(plotted)[:, :, ::-1].copy())
    return {"detections": detections, "annotated_image": annotated_image}
