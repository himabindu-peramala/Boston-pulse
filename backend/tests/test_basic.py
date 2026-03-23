"""Basic tests for Boston Pulse model development."""


def test_eval_dataset_loads():
    from evaluation.eval_dataset import EVAL_DATASET
    assert len(EVAL_DATASET) == 42


def test_eval_dataset_categories():
    from evaluation.eval_dataset import get_questions_by_category
    crime = get_questions_by_category("crime")
    assert len(crime) == 6


def test_bias_detector_handles_empty():
    from evaluation.bias_detector import run_bias_analysis
    result = run_bias_analysis({})
    assert "error" in result


def test_config_loads():
    from app.core.config import settings
    assert settings.chroma_collection_name == "boston_pulse"
