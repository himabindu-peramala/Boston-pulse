"""
Boston Pulse — RAG Evaluation Runner
Runs eval questions through the retrieval pipeline and measures quality.

Usage:
    cd backend
    python -m evaluation.eval_runner
"""
import logging
import time
import json
from datetime import datetime

from evaluation.eval_dataset import EVAL_DATASET
from app.services.retriever import retrieve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_retrieval(question: dict) -> dict:
    """
    Evaluate a single question against the retrieval pipeline.

    Returns dict with metrics:
        - retrieval_accuracy: did correct dataset appear in top-k?
        - keyword_hits: how many expected keywords found in retrieved text
        - keyword_coverage: fraction of expected keywords found
        - avg_score: average similarity score of retrieved chunks
        - response_time_ms: time taken for retrieval
        - retrieved_datasets: which datasets were actually retrieved
    """
    q_text = question["question"]
    expected_ds = question["expected_dataset"]
    expected_kw = question["expected_keywords"]

    # Normalize expected_dataset to list
    if isinstance(expected_ds, str):
        expected_ds = [expected_ds]

    # Time the retrieval
    start = time.time()
    chunks = retrieve(q_text)
    elapsed_ms = round((time.time() - start) * 1000, 2)

    # Check which datasets were retrieved
    retrieved_datasets = [c.dataset for c in chunks]

    # Retrieval accuracy — did any expected dataset appear?
    retrieval_hit = any(ds in retrieved_datasets for ds in expected_ds)

    # Combine all retrieved text for keyword checking
    all_text = " ".join(c.content.lower() for c in chunks)

    # Keyword coverage
    kw_hits = [kw for kw in expected_kw if kw.lower() in all_text]
    kw_coverage = len(kw_hits) / len(expected_kw) if expected_kw else 0

    # Average similarity score
    avg_score = round(sum(c.score for c in chunks) / len(chunks), 4) if chunks else 0

    return {
        "id": question["id"],
        "question": q_text,
        "category": question["category"],
        "difficulty": question["difficulty"],
        "retrieval_hit": retrieval_hit,
        "keyword_coverage": round(kw_coverage, 4),
        "keyword_hits": kw_hits,
        "keyword_misses": [kw for kw in expected_kw if kw.lower() not in all_text],
        "avg_score": avg_score,
        "response_time_ms": elapsed_ms,
        "retrieved_datasets": retrieved_datasets,
        "expected_datasets": expected_ds,
        "num_chunks": len(chunks),
    }


def run_evaluation(questions: list = None) -> dict:
    """
    Run full evaluation across all questions.

    Returns:
        Summary dict with overall metrics and per-question results.
    """
    if questions is None:
        questions = EVAL_DATASET

    logger.info(f"Running evaluation on {len(questions)} questions...")

    results = []
    for i, q in enumerate(questions):
        logger.info(f"[{i+1}/{len(questions)}] {q['id']}: {q['question'][:60]}...")
        try:
            result = evaluate_retrieval(q)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed on {q['id']}: {e}")
            results.append({
                "id": q["id"],
                "question": q["question"],
                "category": q["category"],
                "difficulty": q["difficulty"],
                "retrieval_hit": False,
                "keyword_coverage": 0,
                "error": str(e),
            })

    # Compute summary metrics
    total = len(results)
    hits = sum(1 for r in results if r.get("retrieval_hit", False))
    avg_kw_coverage = sum(r.get("keyword_coverage", 0) for r in results) / total
    avg_score = sum(r.get("avg_score", 0) for r in results) / total
    avg_time = sum(r.get("response_time_ms", 0) for r in results) / total

    # Per-category breakdown
    categories = set(r["category"] for r in results)
    per_category = {}
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        cat_hits = sum(1 for r in cat_results if r.get("retrieval_hit", False))
        per_category[cat] = {
            "total": len(cat_results),
            "retrieval_accuracy": round(cat_hits / len(cat_results), 4),
            "avg_keyword_coverage": round(
                sum(r.get("keyword_coverage", 0) for r in cat_results) / len(cat_results), 4
            ),
        }

    # Per-difficulty breakdown
    difficulties = set(r["difficulty"] for r in results)
    per_difficulty = {}
    for diff in difficulties:
        diff_results = [r for r in results if r["difficulty"] == diff]
        diff_hits = sum(1 for r in diff_results if r.get("retrieval_hit", False))
        per_difficulty[diff] = {
            "total": len(diff_results),
            "retrieval_accuracy": round(diff_hits / len(diff_results), 4),
        }

    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "total_questions": total,
        "retrieval_accuracy": round(hits / total, 4),
        "avg_keyword_coverage": round(avg_kw_coverage, 4),
        "avg_similarity_score": round(avg_score, 4),
        "avg_response_time_ms": round(avg_time, 2),
        "per_category": per_category,
        "per_difficulty": per_difficulty,
        "results": results,
    }

    return summary


def print_summary(summary: dict):
    """Print a readable evaluation summary."""
    print("\n" + "=" * 60)
    print("BOSTON PULSE RAG — EVALUATION RESULTS")
    print("=" * 60)
    print(f"Timestamp: {summary['timestamp']}")
    print(f"Questions evaluated: {summary['total_questions']}")
    print("\nOverall metrics:")
    print(f"  Retrieval accuracy:    {summary['retrieval_accuracy']:.1%}")
    print(f"  Avg keyword coverage:  {summary['avg_keyword_coverage']:.1%}")
    print(f"  Avg similarity score:  {summary['avg_similarity_score']:.4f}")
    print(f"  Avg response time:     {summary['avg_response_time_ms']:.0f} ms")

    print("\nPer category:")
    for cat, metrics in sorted(summary["per_category"].items()):
        print(f"  {cat:20s}  accuracy={metrics['retrieval_accuracy']:.1%}  "
              f"kw_coverage={metrics['avg_keyword_coverage']:.1%}")

    print("\nPer difficulty:")
    for diff, metrics in sorted(summary["per_difficulty"].items()):
        print(f"  {diff:10s}  accuracy={metrics['retrieval_accuracy']:.1%}  "
              f"(n={metrics['total']})")
    print("=" * 60)


if __name__ == "__main__":
    summary = run_evaluation()
    print_summary(summary)

    # Save results to file
    output_path = "evaluation/eval_results.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nDetailed results saved to {output_path}")