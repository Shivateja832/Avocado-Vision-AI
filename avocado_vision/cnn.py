from __future__ import annotations

import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from avocado_vision.pipeline import IMAGE_SIZE, ImageRecord

SEED = 42


class AvocadoCropDataset(Dataset):
    def __init__(
        self,
        records: list[ImageRecord],
        class_to_index: dict[str, int],
        augment: bool = False,
    ) -> None:
        self.records = records
        self.class_to_index = class_to_index
        self.augment = augment
        self.images: list[np.ndarray] = []
        for record in records:
            with Image.open(record.path) as source:
                image = source.convert("RGB").resize(
                    (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
                )
                self.images.append(np.asarray(image, dtype=np.uint8).copy())

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        image = Image.fromarray(self.images[index])
        if self.augment:
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.1))
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
        return tensor, torch.tensor(self.class_to_index[record.label], dtype=torch.long)


class CompactCNN(nn.Module):
    def __init__(self, class_count: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Dropout(0.25), nn.Linear(64, class_count))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs).flatten(1))


def train_cnn(
    train_records: list[ImageRecord],
    class_names: list[str],
    model_path: Path,
    max_epochs: int = 16,
    batch_size: int = 32,
    *,
    validation_records: list[ImageRecord] | None = None,
) -> dict[str, Any]:
    torch.set_num_threads(min(4, torch.get_num_threads()))
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    class_to_index = {name: index for index, name in enumerate(class_names)}
    if validation_records is None:
        labels = [record.label for record in train_records]
        fitting, validation = train_test_split(
            train_records, test_size=0.2, random_state=SEED, stratify=labels
        )
    else:
        fitting, validation = train_records, validation_records
        if not validation:
            raise ValueError("The validation split must contain at least one image.")
        if {record.label for record in fitting} != set(class_names):
            raise ValueError("The training split must contain every configured class.")
        if {record.label for record in validation} != set(class_names):
            raise ValueError("The validation split must contain every configured class.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        AvocadoCropDataset(fitting, class_to_index, augment=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    validation_loader = DataLoader(
        AvocadoCropDataset(validation, class_to_index),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = CompactCNN(len(class_names)).to(device)
    frequencies = Counter(record.label for record in fitting)
    weights = torch.tensor(
        [len(fitting) / (len(class_names) * frequencies[name]) for name in class_names],
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(weight=weights)
    best_state = None
    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()

        model.eval()
        val_truth: list[str] = []
        val_predictions: list[str] = []
        with torch.inference_mode():
            for inputs, targets in validation_loader:
                logits = model(inputs.to(device))
                val_predictions.extend(class_names[i] for i in logits.argmax(1).cpu().tolist())
                val_truth.extend(class_names[i] for i in targets.tolist())
        val_f1 = f1_score(
            val_truth, val_predictions, labels=class_names, average="macro", zero_division=0
        )
        if val_f1 > best_f1:
            best_f1 = float(val_f1)
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= 4:
                break

    if best_state is None:
        raise RuntimeError("CNN training did not produce a validation checkpoint.")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": best_state, "class_names": class_names, "image_size": IMAGE_SIZE},
        model_path,
    )
    return {
        "device": str(device),
        "epochs_run": epoch,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
    }


def predict_cnn(
    model_path: str | Path, records: list[ImageRecord], batch_size: int = 32
) -> tuple[list[str], float]:
    torch.set_num_threads(min(4, torch.get_num_threads()))
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    class_names = list(checkpoint["class_names"])
    class_to_index = {name: index for index, name in enumerate(class_names)}
    model = CompactCNN(len(class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    start = time.perf_counter()
    loader = DataLoader(
        AvocadoCropDataset(records, class_to_index),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    predictions: list[str] = []
    with torch.inference_mode():
        for inputs, _ in loader:
            logits = model(inputs)
            predictions.extend(class_names[i] for i in logits.argmax(1).tolist())
    elapsed = time.perf_counter() - start
    return predictions, elapsed * 1000 / max(1, len(records))


def predict_cnn_image(
    image: Image.Image, model_path: str | Path
) -> dict[str, Any]:
    torch.set_num_threads(min(4, torch.get_num_threads()))
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    class_names = list(checkpoint["class_names"])
    model = CompactCNN(len(class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    resized = image.convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
    )
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    index = int(probabilities.argmax())
    return {"label": class_names[index], "confidence": float(probabilities[index])}
