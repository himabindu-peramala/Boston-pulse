"""
Boston Pulse — Automated Experiment Pipeline
Runs the complete model development evaluation workflow:

  - Baseline evaluation on all 42 test questions
  - Embedding model comparison (re-ingest + re-evaluate per model)
  - Top-K retrieval sensitivity (k = 3, 5, 10)
  - Temperature sensitivity via Gemini LLM generation
  - MLflow experiment tracking for all runs
  - Bias detection across data slices
  - Sensitivity analysis report
  - Final summary with best configuration

Usage:
    cd backend
    export GOOGLE_APPLICATION_CREDENTIALS=./secrets/gcp-key.json
    python -m evaluation.run_all_experiments
"""
import logging
import json
import os
import sys
import time
import shutil
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.eval_runner import run_evaluation, print_summary
from evaluation.eval_dataset import EVAL_DATASET
from evaluation.mlflow_tracker import track_experiment, compare_runs
from evaluation.bias_detector import run_bias_analysis, print_bias_report
from evaluation.sensitivity_analysis import (
    analyze_topk_sensitivity,
    analyze_temperature_sensitivity,
    print_sensitivity_report,
)
from app.services.retriever import retrieve, _get_collection
from app.core.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = "evaluation/results"
EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-base-en-v1.5",
]
TOPK_VALUES = [3, 5, 10]
TEMPERATURE_VALUES = [0.0, 0.2, 0.5, 0.8]
# One representative question per category to keep LLM calls efficient
TEMPERATURE_TEST_IDS = [
    "crime_01", "311_01", "food_01", "berdo_01",
    "city_01", "sweep_01", "cross_01",
]


# ── Helpers ──────────────────────────────────────────────────

def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _reset_pipeline():
    """Force-reload the embedding model and ChromaDB collection."""
    import app.services.embedder as emb
    import app.services.retriever as ret
    emb._model = None
    emb._model_name = None
    ret._client = None
    ret._collection = None
    # Force chromadb to release any cached connections
    import chromadb
    chromadb._default_api = None  # clear internal singleton if present


def _clear_chromadb():
    """Delete and recreate ChromaDB directory for fresh embeddings."""
    import app.services.retriever as ret
    # Disconnect first
    ret._client = None
    ret._collection = None
    path = settings.chroma_persist_dir
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    logger.info(f"Cleared ChromaDB at {path}")


def _reingest_fresh():
    """Full clean re-ingest using subprocess to avoid stale ChromaDB state."""
    _clear_chromadb()
    _reset_pipeline()
    import subprocess
    env = os.environ.copy()
    env["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS", "./secrets/gcp-key.json"
    )
    result = subprocess.run(
        [sys.executable, "scripts/ingest.py"],
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        env=env,
        capture_output=True,
        text=True,
    )
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    if result.returncode != 0:
        print(f"  Ingest stderr: {result.stderr[-300:]}")
    _reset_pipeline()





def _save_json(data: dict, filename: str):
    path = f"{OUTPUT_DIR}/{filename}"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  → Saved {path}")
    return path


def _log_to_mlflow(summary, run_name, extra_params=None):
    params = {
        "embedding_model": settings.embedding_model,
        "llm_model": settings.gemini_model,
        "llm_temperature": settings.gemini_temperature,
        "top_k_results": settings.top_k_results,
    }
    if extra_params:
        params.update(extra_params)
    return track_experiment(summary, run_name=run_name, params=params)


# ── Baseline Evaluation ─────────────────────────────────────

def run_baseline():
    print(f"\n{'='*60}")
    print("BASELINE EVALUATION")
    print(f"{'='*60}")

    summary = run_evaluation()
    print_summary(summary)

    _log_to_mlflow(summary, "baseline_k5_t0.2_minilm",
                   {"experiment_type": "baseline"})
    _save_json(summary, "baseline_eval.json")
    return summary


# ── Embedding Model Comparison ───────────────────────────────

def run_embedding_comparison():
    print(f"\n{'='*60}")
    print("EMBEDDING MODEL COMPARISON")
    print(f"{'='*60}")

    results = {}

    for i, model in enumerate(EMBEDDING_MODELS):
        short = model.split("/")[-1]
        print(f"\n  [{i+1}/{len(EMBEDDING_MODELS)}] {short}")

        settings.embedding_model = model
        _clear_chromadb()
        _reset_pipeline()

        print("    Ingesting...")
        _reingest_fresh()
        _reset_pipeline()

        print("    Evaluating...")
        summary = run_evaluation()
        results[model] = summary

        acc = summary["retrieval_accuracy"]
        kw = summary["avg_keyword_coverage"]
        print(f"    accuracy={acc:.1%}  keyword_coverage={kw:.1%}")

        _log_to_mlflow(summary, f"embed_{short}",
                       {"experiment_type": "embedding_comparison"})

    # Determine best
    best_model = max(results, key=lambda m: (
        results[m]["retrieval_accuracy"],
        results[m]["avg_keyword_coverage"],
    ))

    print(f"\n  Best embedding model: {best_model.split('/')[-1]}")

    _save_json({
        model.split("/")[-1]: {
            "retrieval_accuracy": s["retrieval_accuracy"],
            "avg_keyword_coverage": s["avg_keyword_coverage"],
            "avg_similarity_score": s["avg_similarity_score"],
            "avg_response_time_ms": s["avg_response_time_ms"],
            "per_category": s["per_category"],
        }
        for model, s in results.items()
    } | {"best_model": best_model}, "embedding_comparison.json")

    return results, best_model


# ── Top-K Sensitivity ────────────────────────────────────────

def run_topk_sensitivity():
    print(f"\n{'='*60}")
    print("TOP-K SENSITIVITY")
    print(f"{'='*60}")

    results = {}
    for k in TOPK_VALUES:
        settings.top_k_results = k
        summary = run_evaluation()
        results[k] = summary

        acc = summary["retrieval_accuracy"]
        kw = summary["avg_keyword_coverage"]
        print(f"  k={k:<4}  accuracy={acc:.1%}  keyword_coverage={kw:.1%}")

        _log_to_mlflow(summary, f"topk_{k}",
                       {"experiment_type": "topk_sensitivity"})

    settings.top_k_results = 5  # reset
    _save_json({
        str(k): {
            "retrieval_accuracy": s["retrieval_accuracy"],
            "avg_keyword_coverage": s["avg_keyword_coverage"],
            "avg_similarity_score": s["avg_similarity_score"],
            "avg_response_time_ms": s["avg_response_time_ms"],
        }
        for k, s in results.items()
    }, "topk_sensitivity.json")

    return results


# ── Temperature Sensitivity (LLM Generation) ─────────────────

def run_temperature_sensitivity():
    print(f"\n{'='*60}")
    print("TEMPERATURE SENSITIVITY (LLM Generation)")
    print(f"{'='*60}")

    try:
        import google.generativeai as genai
        genai.configure(api_key=settings.gemini_api_key)
    except Exception as e:
        print(f"  Gemini API unavailable ({e}) — skipping")
        return {}, None

    test_questions = [q for q in EVAL_DATASET if q["id"] in TEMPERATURE_TEST_IDS]
    results = {}

    for temp in TEMPERATURE_VALUES:
        temp_scores = []

        for q in test_questions:
            chunks = retrieve(q["question"])
            context = "\n".join(c.content for c in chunks[:5])

            prompt = (
                "You are Boston Pulse, a helpful assistant for Boston city data.\n"
                "Answer the question based ONLY on the context provided.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {q['question']}\n\nAnswer:"
            )

            try:
                model = genai.GenerativeModel(settings.gemini_model)
                resp = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temp, max_output_tokens=512,
                    ),
                )
                answer = resp.text
            except Exception as e:
                logger.warning(f"Gemini error: {e}")
                answer = ""

            answer_lower = answer.lower()
            kw_hits = [kw for kw in q["expected_keywords"] if kw.lower() in answer_lower]
            kw_cov = len(kw_hits) / len(q["expected_keywords"]) if q["expected_keywords"] else 0
            temp_scores.append({
                "id": q["id"],
                "keyword_coverage": kw_cov,
                "answer_length": len(answer),
            })

        avg_kw = sum(r["keyword_coverage"] for r in temp_scores) / len(temp_scores)
        avg_len = sum(r["answer_length"] for r in temp_scores) / len(temp_scores)
        results[temp] = {
            "avg_keyword_coverage": round(avg_kw, 4),
            "avg_answer_length": round(avg_len, 1),
            "details": temp_scores,
        }
        print(f"  t={temp:<5}  keyword_coverage={avg_kw:.1%}  avg_len={avg_len:.0f}")

        # Log to MLflow
        mlflow_summary = {
            "retrieval_accuracy": 0,
            "avg_keyword_coverage": round(avg_kw, 4),
            "avg_similarity_score": 0,
            "avg_response_time_ms": 0,
            "total_questions": len(temp_scores),
            "per_category": {},
            "per_difficulty": {},
        }
        _log_to_mlflow(mlflow_summary, f"llm_temp_{temp}",
                       {"experiment_type": "temperature_sensitivity_llm",
                        "llm_temperature": temp})

    best_temp = max(results, key=lambda t: results[t]["avg_keyword_coverage"]) if results else None
    if best_temp is not None:
        print(f"\n  Best temperature: {best_temp}")

    _save_json(
        {str(t): v for t, v in results.items()} | {"best_temperature": best_temp},
        "temperature_comparison.json",
    )

    return results, best_temp


# ── Bias Detection ───────────────────────────────────────────

def run_bias_detection(baseline_summary):
    print(f"\n{'='*60}")
    print("BIAS DETECTION")
    print(f"{'='*60}")

    report = run_bias_analysis(baseline_summary)
    print_bias_report(report)
    _save_json(report, "bias_report.json")
    return report


# ── Sensitivity Analysis ─────────────────────────────────────

def run_sensitivity_report(topk_results, temp_results):
    print(f"\n{'='*60}")
    print("SENSITIVITY ANALYSIS")
    print(f"{'='*60}")

    topk_report = analyze_topk_sensitivity(topk_results)

    # Build temp summary compatible with analyzer
    temp_summaries = {}
    for t, data in temp_results.items():
        temp_summaries[t] = {
            "retrieval_accuracy": 0,
            "avg_keyword_coverage": data["avg_keyword_coverage"],
            "avg_similarity_score": 0,
            "avg_response_time_ms": 0,
        }
    temp_report = analyze_temperature_sensitivity(temp_summaries) if temp_summaries else {}

    print_sensitivity_report(topk_report=topk_report, temp_report=temp_report)
    _save_json({
        "topk_sensitivity": topk_report,
        "temperature_sensitivity": temp_report,
    }, "sensitivity_report.json")

    return topk_report, temp_report


# ── Re-ingest with Best Model ────────────────────────────────

def reingest_best_model(best_model):
    if settings.embedding_model == best_model:
        print(f"\n  ChromaDB already uses best model ({best_model.split('/')[-1]})")
        return

    print(f"\n  Re-ingesting with best model: {best_model.split('/')[-1]}")
    settings.embedding_model = best_model
    _clear_chromadb()
    _reset_pipeline()
    _reingest_fresh()
    _reset_pipeline()


# ── Final Summary ────────────────────────────────────────────

def save_final_summary(baseline, embed_results, best_model,
                        topk_results, temp_results, best_temp,
                        bias_report, topk_report, temp_report):
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")

    collection = _get_collection()

    final = {
        "timestamp": datetime.utcnow().isoformat(),
        "pipeline_config": {
            "best_embedding_model": best_model,
            "llm_model": settings.gemini_model,
            "best_temperature": best_temp,
            "best_top_k": topk_report.get("best_value", 5),
            "vector_db": "chromadb",
            "total_chunks": collection.count(),
        },
        "baseline_evaluation": {
            "total_questions": baseline["total_questions"],
            "retrieval_accuracy": baseline["retrieval_accuracy"],
            "avg_keyword_coverage": baseline["avg_keyword_coverage"],
            "avg_similarity_score": baseline["avg_similarity_score"],
            "avg_response_time_ms": baseline["avg_response_time_ms"],
            "per_category": baseline["per_category"],
            "per_difficulty": baseline["per_difficulty"],
        },
        "embedding_comparison": {
            m.split("/")[-1]: {
                "retrieval_accuracy": s["retrieval_accuracy"],
                "avg_keyword_coverage": s["avg_keyword_coverage"],
            }
            for m, s in embed_results.items()
        },
        "topk_sensitivity": {
            str(k): {
                "retrieval_accuracy": s["retrieval_accuracy"],
                "avg_keyword_coverage": s["avg_keyword_coverage"],
            }
            for k, s in topk_results.items()
        },
        "temperature_sensitivity": {
            str(t): {"avg_keyword_coverage": d["avg_keyword_coverage"]}
            for t, d in temp_results.items()
        } if temp_results else "skipped",
        "bias_analysis": {
            "bias_detected": bias_report.get("bias_detected", False),
            "total_disparities": bias_report.get("total_disparities", 0),
            "recommendations": bias_report.get("recommendations", []),
        },
        "sensitivity_insights": {
            "topk_insight": topk_report.get("insight", ""),
            "temperature_insight": temp_report.get("insight", ""),
        },
        "output_files": [
            f"{OUTPUT_DIR}/baseline_eval.json",
            f"{OUTPUT_DIR}/embedding_comparison.json",
            f"{OUTPUT_DIR}/topk_sensitivity.json",
            f"{OUTPUT_DIR}/temperature_comparison.json",
            f"{OUTPUT_DIR}/bias_report.json",
            f"{OUTPUT_DIR}/sensitivity_report.json",
            f"{OUTPUT_DIR}/final_summary.json",
        ],
    }

    _save_json(final, "final_summary.json")

    print(f"\n  Best embedding model:  {best_model.split('/')[-1]}")
    print(f"  Best temperature:      {best_temp}")
    print(f"  Best top_k:            {topk_report.get('best_value', 5)}")
    print(f"  Baseline accuracy:     {baseline['retrieval_accuracy']:.1%}")
    print(f"  Bias disparities:      {bias_report.get('total_disparities', 0)}")
    print(f"  MLflow runs logged:    {len(compare_runs())}")

    return final


# ── Main ─────────────────────────────────────────────────────

def main():
    start = time.time()

    print("=" * 60)
    print("BOSTON PULSE — AUTOMATED EXPERIMENT PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Preflight check
    collection = _get_collection()
    count = collection.count()
    print(f"\nChromaDB chunks: {count}")
    if count == 0:
        print("ERROR: ChromaDB is empty. Run python scripts/ingest.py first.")
        sys.exit(1)

    _ensure_output_dir()

    # Run everything
    baseline = run_baseline()
    embed_results, best_model = run_embedding_comparison()

    # Re-ingest with best model before top-k tests
    reingest_best_model(best_model)

    topk_results = run_topk_sensitivity()
    temp_results, best_temp = run_temperature_sensitivity()
    bias_report = run_bias_detection(baseline)
    topk_report, temp_report = run_sensitivity_report(topk_results, temp_results)

    save_final_summary(
        baseline, embed_results, best_model,
        topk_results, temp_results, best_temp,
        bias_report, topk_report, temp_report,
    )

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE — {elapsed:.0f}s")
    print(f"{'='*60}")
    print("\nNext:")
    print("  mlflow ui                         # view experiment dashboard")
    print("  git add evaluation/ mlruns/")
    print("  git commit -m 'Complete model development: eval, experiments, bias, sensitivity'")
    print("  git push origin mukul/model-development")


if __name__ == "__main__":
    main()
