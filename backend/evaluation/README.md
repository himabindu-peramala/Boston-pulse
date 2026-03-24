# Boston Pulse — Model Development & Evaluation

This directory contains the full RAG evaluation pipeline for Boston Pulse, including embedding model comparison, bias detection, sensitivity analysis, and MLflow experiment tracking.

## Results Summary

| Embedding Model | Retrieval Accuracy | Keyword Coverage |
|---|---|---|
| all-MiniLM-L6-v2 | 71.4% | — |
| all-mpnet-base-v2 | 69.0% | — |
| **BAAI/bge-base-en-v1.5** | **76.2%** ✅ | **61.9%** |

**Best model: BGE-base-en-v1.5** — selected for production.

### Per-Category Accuracy (BGE model)
| Category | Accuracy |
|---|---|
| 311 Service Requests | 100% |
| BERDO (Building Emissions) | 100% |
| Food Inspections | 100% |
| Street Sweeping | 100% |
| Cross-Dataset | 100% |
| Crime | 33.3% ⚠️ |
| CityScore | 0.0% ⚠️ |

---

## Files

| File | Description |
|---|---|
| `eval_dataset.py` | 42 test questions across 7 categories and 3 difficulty levels |
| `eval_runner.py` | Runs questions through the retrieval pipeline and measures accuracy |
| `bias_detector.py` | Detects performance disparities across dataset slices |
| `sensitivity_analysis.py` | Analyzes how top-k and temperature affect performance |
| `mlflow_tracker.py` | Logs all experiments to MLflow for comparison |
| `run_all_experiments.py` | Orchestrates the full experiment pipeline end-to-end |
| `results/` | JSON output files from all experiments |

---

## Evaluation Dataset

42 questions across 7 categories and 3 difficulty levels:

- **Crime** (6 questions) — incident types, districts, time-of-day patterns
- **311 Service Requests** (6 questions) — complaint types, resolution times, neighborhoods
- **Food Inspections** (6 questions) — pass/fail results, specific restaurants
- **BERDO** (6 questions) — building emissions, energy scores
- **CityScore** (6 questions) — city performance metrics and targets
- **Street Sweeping** (6 questions) — schedules, districts, tow zones
- **Cross-Dataset** (6 questions) — queries requiring multiple data sources

---

## How to Run

### Prerequisites
```bash
cd backend
export GOOGLE_APPLICATION_CREDENTIALS=./secrets/gcp-key.json
```

### Run individual evaluation
```bash
python -m evaluation.eval_runner
```

### Run bias detection on saved results
```bash
python -m evaluation.bias_detector
```

### Run full experiment pipeline (all models, sensitivity, bias)
```bash
python -m evaluation.run_all_experiments
```

### View MLflow experiment dashboard
```bash
mlflow ui
# Open http://localhost:5000
```

---

## Bias Detection

6 disparities detected in the BGE model evaluation:

- **Crime** retrieval accuracy: 33.3% vs average 76.2% — chunks may need reformatting
- **CityScore** retrieval accuracy: 0.0% vs average 76.2% — data may not be ingested correctly

Recommendations are saved to `results/bias_report.json`.

---

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/model_pipeline.yml`) runs on every push to `mukul/model-development`:

1. **Lint & Code Quality** — ruff check
2. **Unit Tests** — pytest + dataset/chunking verification
3. **Model Evaluation** — runs eval pipeline if GCP data is available
4. **Registry Push** — pushes best model to MLflow registry (main/prod only)
