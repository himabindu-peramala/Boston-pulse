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

def test_sensitivity_analysis_empty():
    from evaluation.sensitivity_analysis import analyze_topk_sensitivity
    result = analyze_topk_sensitivity({})
    assert "error" in result


def test_sensitivity_analysis_with_data():
    from evaluation.sensitivity_analysis import analyze_topk_sensitivity
    mock_results = {
        3: {"retrieval_accuracy": 0.6, "avg_keyword_coverage": 0.5, "avg_similarity_score": 0.7, "avg_response_time_ms": 50},
        5: {"retrieval_accuracy": 0.7, "avg_keyword_coverage": 0.6, "avg_similarity_score": 0.75, "avg_response_time_ms": 55},
        10: {"retrieval_accuracy": 0.75, "avg_keyword_coverage": 0.65, "avg_similarity_score": 0.8, "avg_response_time_ms": 60},
    }
    result = analyze_topk_sensitivity(mock_results)
    assert "best_value" in result
    assert result["best_value"] == 10


def test_slice_by_category_empty():
    from evaluation.bias_detector import slice_by_category
    result = slice_by_category([])
    assert result == {}


def test_mlflow_compare_runs_returns_list():
    import os
    os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"
    from evaluation.mlflow_tracker import compare_runs
    result = compare_runs()
    assert isinstance(result, list)


def test_eval_dataset_difficulty_levels():
    from evaluation.eval_dataset import get_questions_by_difficulty
    easy = get_questions_by_difficulty("easy")
    hard = get_questions_by_difficulty("hard")
    medium = get_questions_by_difficulty("medium")
    assert len(easy) > 0
    assert len(hard) > 0
    assert len(medium) > 0
