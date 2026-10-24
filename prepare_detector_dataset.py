from __future__ import annotations

import argparse
import json

from prepare_avocado_dataset import prepare_detection_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare full-scene avocado-stage YOLO train/val/test data."
    )
    parser.add_argument("--output-dir", default="data/avocado_detection")
    args = parser.parse_args()
    result = prepare_detection_dataset(args.output_dir)
    print(
        json.dumps(
            {
                "dataset_url": result["dataset_url"],
                "source_images_per_split": result["source_images_per_split"],
                "annotation_counts_per_split": result["annotation_counts_per_split"],
                "weights_classes": result["detector_class_mapping"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
