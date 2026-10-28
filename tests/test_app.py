import json
from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_renders_without_streamlit_exceptions() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert set(app.tabs[2].radio[0].options) == {
        "Full-scene detection (YOLO)",
        "Crop classification (SVM + CNN)",
    }
    detector_weights = app_path.parent / "models" / "avocado_detection" / "best.pt"
    if detector_weights.is_file():
        assert any(
            "trained avocado detector" in item.value.lower()
            for item in app.tabs[2].success
        )
    else:
        assert any(
            "detector is not trained yet" in item.value.lower()
            for item in app.tabs[2].warning
        )
    app.tabs[2].radio[0].set_value("Crop classification (SVM + CNN)")
    app.run()

    assert not app.exception
    assert app.title[0].value == "AvocadoVision AI"
    assert not any("Dataset mismatch" in warning.value for warning in app.warning)
    assert any("overripe" in item.value for item in app.caption)
    assert any("close-up crop containing one avocado" in item.value for item in app.info)
    assert [tab.label for tab in app.tabs] == [
        "Dataset inspection",
        "Train and compare",
        "Example prediction",
    ]
    assert app.tabs[0].metric[0].value == "3996 / 3996"
    assert app.tabs[1].success[0].value.startswith(
        "Best macro F1 on this held-out test split:"
    )
    assert len(app.tabs[1].dataframe) == 4
    assert json.loads(app.tabs[1].json[0].value)["actual_label"] == "overripe"
    assert (
        app.tabs[2].success[0].value
        == "Both crop-classification models are available."
    )
    assert (
        Path(__file__).resolve().parents[1]
        / "models"
        / "avocado_pure"
        / "metrics.json"
    ).is_file() == ("training_results" in app.session_state)
    assert "Analyze dataset" in [button.label for button in app.button]
    assert any(button.label == "Train both models" for button in app.button)
    assert any(button.label == "Train avocado detector" for button in app.button)
    assert "Download and prepare avocado dataset" in [
        button.label for button in app.button
    ]


def test_app_guides_user_when_dataset_splits_are_missing(tmp_path: Path) -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    missing_dataset = tmp_path / "not_prepared"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    app.text_input[0].set_value(str(missing_dataset))
    app.run()

    assert not app.exception
    assert any("Dataset splits were not found" in item.value for item in app.warning)
    analyze_button = next(
        button for button in app.button if button.label == "Analyze dataset"
    )
    assert analyze_button.disabled
    assert any(
        "prepare_avocado_dataset.py --output-dir" in item.value
        for item in app.code
    )
