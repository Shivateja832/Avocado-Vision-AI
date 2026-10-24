import argparse
import json
from pathlib import Path

from avocado_vision.pipeline import train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and compare the avocado-ripeness classification models."
    )
    parser.add_argument(
        "--data-dir",
        default="data/avocado_pure",
        help="Avocado dataset folder containing train/, validation/, and test/",
    )
    parser.add_argument(
        "--model-dir",
        default="models/avocado_pure",
        help="Directory for trained models/results",
    )
    parser.add_argument("--epochs", type=int, default=16, help="Maximum CNN training epochs")
    args = parser.parse_args()
    results = train_and_evaluate(
        Path(args.data_dir), Path(args.model_dir), max_epochs=args.epochs
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
