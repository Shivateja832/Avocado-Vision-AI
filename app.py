import json
from pathlib import Path
import zipfile

import pandas as pd
import requests
import streamlit as st
from PIL import Image, UnidentifiedImageError

from avocado_vision.detector import predict_scene
from avocado_vision.pipeline import analyze_dataset, predict_image, train_and_evaluate
from prepare_avocado_dataset import prepare_dataset, prepare_detection_dataset
from train_detector import train_detector

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data" / "avocado_pure"
DEFAULT_DETECTION_DATA_DIR = ROOT / "data" / "avocado_detection"
DEFAULT_MODEL_DIR = ROOT / "models" / "avocado_pure"
DEFAULT_DETECTOR_MODEL_DIR = ROOT / "models" / "avocado_detection"
DETECTOR_WEIGHTS = DEFAULT_DETECTOR_MODEL_DIR / "best.pt"
RESULTS_PATH = DEFAULT_MODEL_DIR / "metrics.json"
_STREAMLIT_VERSION = tuple(int(part) for part in st.__version__.split(".")[:2])
_FULL_WIDTH_OPTIONS: dict[str, str | bool] = (
    {"width": "stretch"}
    if _STREAMLIT_VERSION >= (1, 50)
    else {"use_container_width": True}
)


def _prepare_classification_dataset(output_dir: Path) -> None:
    try:
        with st.spinner(
            "Downloading the avocado-only dataset, verifying it, and preparing avocado crops..."
        ):
            prepared = prepare_dataset(output_dir)
            st.session_state["dataset_analysis"] = analyze_dataset(output_dir)
            st.session_state.pop("training_results", None)
        st.success(
            f'Prepared {prepared["prepared_crop_count"]} avocado-only crops '
            "in train/validation/test splits."
        )
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        ValueError,
        RuntimeError,
        requests.RequestException,
        zipfile.BadZipFile,
    ) as exc:
        st.error(f"Dataset preparation failed: {exc}")


st.set_page_config(page_title="AvocadoVision AI", page_icon="🥑", layout="wide")
st.title("AvocadoVision AI")
st.caption(
    "Avocado detection and ripeness classification · unripe · ripe · overripe"
)
st.info(
    "Uses the avocado-only ripeness dataset from Kaggle (MIT license)."
)
with st.expander("Dataset source and setup"):
    st.markdown(
        "Source: [Avocado Ripeness YOLO Format on Kaggle]"
        "(https://www.kaggle.com/datasets/deivdf/avocado-ripeness-yolo-format), "
        "dataset version 1, MIT license. All four source labels are avocado "
        "categories: `ripe`, `overripe`, `unripe`, and generic `avocado`. The "
        "generic label is reported as stage unknown by the detector and is not "
        "used as a crop-classifier stage."
    )
    if st.button("Download and prepare avocado dataset", key="prepare_dataset"):
        _prepare_classification_dataset(DEFAULT_DATA_DIR)
    if st.button(
        "Prepare full-scene avocado detection data", key="prepare_detection_dataset"
    ):
        try:
            with st.spinner(
                "Downloading and preparing source images with avocado-stage boxes..."
            ):
                detection_data = prepare_detection_dataset(
                    DEFAULT_DETECTION_DATA_DIR
                )
            st.success(
                "Prepared full-scene avocado detection images: "
                f'{detection_data["source_images_per_split"]}'
            )
        except (
            FileExistsError,
            FileNotFoundError,
            OSError,
            ValueError,
            RuntimeError,
            requests.RequestException,
            zipfile.BadZipFile,
        ) as exc:
            st.error(f"Detection data preparation failed: {exc}")

data_dir_text = st.text_input("Dataset folder", str(DEFAULT_DATA_DIR))
data_dir = Path(data_dir_text).expanduser()
dataset_ready = (data_dir / "train").is_dir() and (data_dir / "test").is_dir()
if "training_results" not in st.session_state and RESULTS_PATH.is_file():
    try:
        saved_results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"Could not load saved evaluation results: {exc}")
    else:
        if data_dir.resolve() == DEFAULT_DATA_DIR.resolve():
            saved_results.setdefault("dataset", {})["data_dir"] = str(
                DEFAULT_DATA_DIR.resolve()
            )
            st.session_state["training_results"] = saved_results
            st.session_state["dataset_analysis"] = saved_results["dataset"]
tab_data, tab_models, tab_predict = st.tabs(
    ["Dataset inspection", "Train and compare", "Example prediction"]
)

with tab_data:
    st.subheader("Dataset quality and class balance")
    if not dataset_ready:
        st.warning(
            f"Dataset splits were not found in {data_dir}. Prepare the avocado "
            "dataset before analysis, or choose a folder containing train/ and test/."
        )
        if data_dir.resolve() == DEFAULT_DATA_DIR.resolve():
            if st.button(
                "Download and prepare dataset now",
                key="prepare_missing_classification_dataset",
            ):
                _prepare_classification_dataset(DEFAULT_DATA_DIR)
        else:
            st.code(
                f'python prepare_avocado_dataset.py --output-dir "{data_dir}"',
                language="powershell",
            )
    if st.button("Analyze dataset", key="analyze", disabled=not dataset_ready):
        try:
            st.session_state["dataset_analysis"] = analyze_dataset(data_dir)
        except (FileNotFoundError, ValueError, OSError) as exc:
            st.error(f"Dataset analysis failed: {exc}")

    analysis = st.session_state.get("dataset_analysis")
    if analysis:
        st.metric("Readable images", f'{analysis["valid_images"]} / {analysis["total_images"]}')
        st.write(
            f'Train: {analysis["train_images"]} images · '
            f'Validation: {analysis["validation_images"]} images · '
            f'Test: {analysis["test_images"]} images · '
            f'Train majority/minority ratio: {analysis["train_imbalance_ratio"]}:1'
        )
        st.dataframe(pd.DataFrame(analysis["class_counts"]).T, **_FULL_WIDTH_OPTIONS)
        st.json(
            {
                "image_size": analysis["image_size"],
                "unreadable_count": analysis["unreadable_count"],
                "exact_duplicate_groups": analysis["exact_duplicate_groups"],
                "cross_split_duplicate_groups": analysis["cross_split_duplicate_groups"],
                "brightness_iqr_outliers": analysis["brightness_iqr_outliers"],
                "mean_sample_brightness_0_255": analysis[
                    "mean_sample_brightness_0_255"
                ],
            }
        )
        for limitation in analysis["limitations"]:
            st.info(limitation)

with tab_models:
    st.subheader("Held-out test comparison")
    max_epochs = st.slider("CNN maximum epochs", min_value=4, max_value=30, value=16)
    if st.button("Train both models", type="primary"):
        try:
            with st.spinner("Training SVM and compact CNN; the test split is held out..."):
                dataset_analysis = st.session_state.get("dataset_analysis")
                if (
                    dataset_analysis is not None
                    and Path(dataset_analysis["data_dir"]).resolve()
                    != data_dir.resolve()
                ):
                    dataset_analysis = None
                st.session_state["training_results"] = train_and_evaluate(
                    data_dir,
                    DEFAULT_MODEL_DIR,
                    max_epochs=max_epochs,
                    dataset_analysis=dataset_analysis,
                )
        except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
            st.error(f"Training failed: {exc}")

    st.divider()
    st.subheader("Train full-scene detector")
    detector_epochs = st.slider(
        "Detector maximum epochs", min_value=1, max_value=30, value=5
    )
    if st.button("Train avocado detector"):
        try:
            with st.spinner(
                "Fine-tuning the avocado detector, then evaluating on the held-out test scenes..."
            ):
                detector_results = train_detector(
                    DEFAULT_DETECTION_DATA_DIR,
                    DEFAULT_DETECTOR_MODEL_DIR,
                    epochs=detector_epochs,
                )
            st.session_state["detector_results"] = detector_results
            st.success("Detector training and held-out test evaluation completed.")
        except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
            st.error(f"Detector training failed: {exc}")

    detector_metrics_path = DEFAULT_DETECTOR_MODEL_DIR / "detection_metrics.json"
    if (
        "detector_results" not in st.session_state
        and detector_metrics_path.is_file()
    ):
        try:
            st.session_state["detector_results"] = json.loads(
                detector_metrics_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            st.error(f"Could not load saved detector metrics: {exc}")
    detector_results = st.session_state.get("detector_results")
    if detector_results:
        st.write("Full-scene detector held-out test metrics")
        detector_metrics = detector_results["test_metrics"]
        detector_rows = {
            metric_name: float(metric_value)
            for metric_name, metric_value in detector_metrics.items()
            if metric_name != "per_class_map50"
        }
        detector_rows.update(
            {
                f"mAP@0.5 ({class_name})": float(score)
                for class_name, score in detector_metrics[
                    "per_class_map50"
                ].items()
            }
        )
        st.dataframe(
            pd.DataFrame.from_dict(
                detector_rows, orient="index", columns=["Score"]
            ),
            **_FULL_WIDTH_OPTIONS,
        )

    results = st.session_state.get("training_results")
    if (
        results
        and data_dir.resolve() != DEFAULT_DATA_DIR.resolve()
        and Path(results["dataset"]["data_dir"]).resolve() != data_dir.resolve()
    ):
        results = None
        st.info("These saved metrics are for a different dataset folder. Train again to compare it.")
    if results:
        best_model_name = max(
            results["metrics"],
            key=lambda model_name: results["metrics"][model_name]["macro_f1"],
        )
        best_model = results["metrics"][best_model_name]
        st.success(
            f"Best macro F1 on this held-out test split: **{best_model_name}** "
            f"(macro F1 {best_model['macro_f1']:.3f}, "
            f"macro recall {best_model['macro_recall']:.3f})."
        )
        st.caption(
            "This identifies the stronger result on this small supplied test split; "
            "it is not a production performance guarantee."
        )
        rows = []
        for model_name, metrics in results["metrics"].items():
            rows.append(
                {
                    "Model": model_name,
                    "Accuracy": metrics["accuracy"],
                    "Macro precision": metrics["macro_precision"],
                    "Macro recall": metrics["macro_recall"],
                    "Macro F1": metrics["macro_f1"],
                    "Weighted F1": metrics["weighted_f1"],
                    "Inference ms/image": metrics["inference_ms_per_image"],
                }
            )
        st.dataframe(
            pd.DataFrame(rows).set_index("Model"), **_FULL_WIDTH_OPTIONS
        )
        st.caption(
            "Macro F1 weights each class equally; macro recall reveals missed classes. "
            "Use accuracy and the confusion matrices as supporting measures."
        )
        selected = st.selectbox("Confusion matrix model", list(results["metrics"]))
        metric = results["metrics"][selected]
        matrix = pd.DataFrame(
            metric["confusion_matrix"],
            index=results["classes"],
            columns=results["classes"],
        )
        st.write("Rows are actual labels; columns are predicted labels.")
        st.dataframe(matrix, **_FULL_WIDTH_OPTIONS)
        st.write("Per-class report")
        per_class = {
            name: {
                key: value
                for key, value in details.items()
                if key in {"precision", "recall", "f1-score", "support"}
            }
            for name, details in metric["per_class"].items()
            if name in results["classes"]
        }
        st.dataframe(pd.DataFrame(per_class).T, **_FULL_WIDTH_OPTIONS)
        st.write("Example test image run")
        st.json(results["example_run"])
        st.info(results["selection_note"])

with tab_predict:
    st.subheader("Detect avocados and classify ripeness")
    detector_available = DETECTOR_WEIGHTS.is_file()
    prediction_mode = st.radio(
        "Prediction mode",
        ["Full-scene detection (YOLO)", "Crop classification (SVM + CNN)"],
        horizontal=True,
        index=0 if detector_available else 1,
        key="prediction_mode",
    )
    model_files_present = all(
        (DEFAULT_MODEL_DIR / filename).is_file()
        for filename in ("classical.joblib", "compact_cnn.pt")
    )
    if prediction_mode == "Full-scene detection (YOLO)":
        st.info(
            "Upload a full image. The detector returns avocado bounding boxes "
            "and ripeness stages; only ripe, overripe, and unripe are supported."
        )
        if detector_available:
            st.success("The trained avocado detector is available.")
        else:
            st.warning(
                "The full-scene detector is not trained yet. Prepare detection "
                "data, then run `python train_detector.py`."
            )
    else:
        st.info(
            "Upload a close-up crop containing one avocado. These classifiers "
            "classify the crop only. Use full-scene detection mode to locate "
            "avocados in a complete image."
        )
        if model_files_present:
            st.success("Both crop-classification models are available.")
        else:
            st.info(
                "No complete crop-classification model set is available. "
                "Train both models first."
            )
    uploaded_image = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )
    if uploaded_image is not None:
        try:
            uploaded_image.seek(0)
            with Image.open(uploaded_image) as source:
                image = source.convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            st.error(f"Could not read the uploaded image: {exc}")
        else:
            st.image(image, caption=uploaded_image.name, width=320)
            can_predict = (
                detector_available
                if prediction_mode == "Full-scene detection (YOLO)"
                else model_files_present
            )
            if st.button(
                "Predict",
                key="predict",
                disabled=not can_predict,
            ):
                try:
                    if prediction_mode == "Full-scene detection (YOLO)":
                        result = predict_scene(image, DETECTOR_WEIGHTS)
                        st.image(
                            result["annotated_image"],
                            caption="Detected avocado boxes and predicted stage",
                            **_FULL_WIDTH_OPTIONS,
                        )
                        if result["detections"]:
                            st.dataframe(
                                pd.DataFrame(result["detections"]),
                                **_FULL_WIDTH_OPTIONS,
                            )
                        else:
                            st.info("No avocado-stage boxes were detected.")
                        st.caption(
                            "Detector confidence is a model score, not a calibrated "
                            "probability. Stage labels are limited to this dataset's "
                            "ripe, overripe, and unripe annotations."
                        )
                    else:
                        predictions = predict_image(image, DEFAULT_MODEL_DIR)
                        st.dataframe(
                            pd.DataFrame.from_dict(predictions, orient="index").rename_axis(
                                "Model"
                            ),
                            **_FULL_WIDTH_OPTIONS,
                        )
                        st.caption(
                            "Confidence is the model's score for the predicted source-dataset class; "
                            "it is not a calibrated probability or a ripeness score."
                        )
                except (OSError, ValueError, RuntimeError) as exc:
                    st.error(f"Prediction failed: {exc}")
