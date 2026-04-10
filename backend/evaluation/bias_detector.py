"""
Boston Pulse — Bias Detector
Analyzes RAG evaluation results for performance disparities
across data slices: dataset category, difficulty, and neighborhood.

Uses slicing techniques as required by Model Development Guidelines.

Usage:
    from evaluation.bias_detector import run_bias_analysis
    report = run_bias_analysis(eval_summary)
"""
import logging

logger = logging.getLogger(__name__)

# Disparity threshold — flag if accuracy gap exceeds this
DISPARITY_THRESHOLD = 0.15  # 15% gap


def slice_by_category(results: list) -> dict:
    """
    Slice evaluation results by dataset category.
    Checks if some datasets get better retrieval than others.

    Example bias: Crime queries retrieve well (90%) but
    BERDO queries fail (50%) — system is biased toward crime data.
    """
    slices = {}
    for r in results:
        cat = r["category"]
        if cat not in slices:
            slices[cat] = {"total": 0, "hits": 0, "kw_coverage": []}
        slices[cat]["total"] += 1
        if r.get("retrieval_hit", False):
            slices[cat]["hits"] += 1
        slices[cat]["kw_coverage"].append(r.get("keyword_coverage", 0))

    # Compute metrics per slice
    for cat, data in slices.items():
        data["retrieval_accuracy"] = round(data["hits"] / data["total"], 4)
        data["avg_keyword_coverage"] = round(
            sum(data["kw_coverage"]) / len(data["kw_coverage"]), 4
        )
        del data["kw_coverage"]

    return slices


def slice_by_difficulty(results: list) -> dict:
    """
    Slice by question difficulty.
    Expected: hard questions perform worse. But if easy questions
    also fail, there's a fundamental retrieval problem.
    """
    slices = {}
    for r in results:
        diff = r["difficulty"]
        if diff not in slices:
            slices[diff] = {"total": 0, "hits": 0}
        slices[diff]["total"] += 1
        if r.get("retrieval_hit", False):
            slices[diff]["hits"] += 1

    for diff, data in slices.items():
        data["retrieval_accuracy"] = round(data["hits"] / data["total"], 4)

    return slices


def slice_by_retrieved_dataset(results: list) -> dict:
    """
    Analyze which datasets are over/under-represented in retrieval.
    If crime chunks dominate every query (even food questions),
    the embedding space is biased.
    """
    dataset_counts = {}
    total_chunks = 0

    for r in results:
        for ds in r.get("retrieved_datasets", []):
            dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
            total_chunks += 1

    # Convert to proportions
    proportions = {}
    for ds, count in dataset_counts.items():
        proportions[ds] = {
            "count": count,
            "proportion": round(count / total_chunks, 4) if total_chunks > 0 else 0,
        }

    return proportions


def detect_disparities(slices: dict, metric: str = "retrieval_accuracy") -> list:
    """
    Detect significant performance disparities between slices.

    Returns list of flagged disparities where the gap exceeds threshold.
    """
    if len(slices) < 2:
        return []

    values = {k: v[metric] for k, v in slices.items() if metric in v}
    if not values:
        return []

    overall_avg = sum(values.values()) / len(values)
    max_val = max(values.values())
    min_val = min(values.values())
    gap = max_val - min_val

    disparities = []

    # Flag if overall gap exceeds threshold
    if gap > DISPARITY_THRESHOLD:
        best = max(values, key=values.get)
        worst = min(values, key=values.get)
        disparities.append({
            "type": "performance_gap",
            "metric": metric,
            "best_slice": best,
            "best_value": max_val,
            "worst_slice": worst,
            "worst_value": min_val,
            "gap": round(gap, 4),
            "threshold": DISPARITY_THRESHOLD,
            "severity": "high" if gap > 0.3 else "medium",
        })

    # Flag individual slices that are significantly below average
    for slice_name, value in values.items():
        if overall_avg - value > DISPARITY_THRESHOLD:
            disparities.append({
                "type": "underperforming_slice",
                "metric": metric,
                "slice": slice_name,
                "value": value,
                "average": round(overall_avg, 4),
                "gap_from_avg": round(overall_avg - value, 4),
                "severity": "high" if overall_avg - value > 0.3 else "medium",
            })

    return disparities


def run_bias_analysis(eval_summary: dict) -> dict:
    """
    Run full bias analysis on evaluation results.

    Args:
        eval_summary: Output from eval_runner.run_evaluation()

    Returns:
        Bias report with slices, disparities, and recommendations.
    """
    results = eval_summary.get("results", [])

    if not results:
        return {"error": "No evaluation results to analyze"}

    # Slice the data
    category_slices = slice_by_category(results)
    difficulty_slices = slice_by_difficulty(results)
    retrieval_slices = slice_by_retrieved_dataset(results)

    # Detect disparities
    category_disparities = detect_disparities(category_slices)
    kw_disparities = detect_disparities(category_slices, "avg_keyword_coverage")

    all_disparities = category_disparities + kw_disparities

    # Generate recommendations
    recommendations = []
    for d in all_disparities:
        if d["type"] == "underperforming_slice":
            recommendations.append(
                f"Improve retrieval for '{d['slice']}' queries — "
                f"accuracy is {d['value']:.1%} vs average {d['average']:.1%}. "
                f"Consider adding more training data or adjusting chunk format "
                f"for this dataset."
            )
        elif d["type"] == "performance_gap":
            recommendations.append(
                f"Large performance gap between '{d['best_slice']}' "
                f"({d['best_value']:.1%}) and '{d['worst_slice']}' "
                f"({d['worst_value']:.1%}). "
                f"Review embedding quality and chunk content for "
                f"'{d['worst_slice']}' dataset."
            )

    report = {
        "bias_detected": len(all_disparities) > 0,
        "total_disparities": len(all_disparities),
        "slices": {
            "by_category": category_slices,
            "by_difficulty": difficulty_slices,
            "by_retrieved_dataset": retrieval_slices,
        },
        "disparities": all_disparities,
        "recommendations": recommendations,
    }

    return report


def print_bias_report(report: dict):
    """Print a readable bias analysis report."""
    print("\n" + "=" * 60)
    print("BOSTON PULSE RAG — BIAS ANALYSIS REPORT")
    print("=" * 60)

    if report.get("error"):
        print(f"Error: {report['error']}")
        return

    print(f"\nBias detected: {'YES' if report['bias_detected'] else 'NO'}")
    print(f"Total disparities found: {report['total_disparities']}")

    print("\n--- Performance by dataset category ---")
    for cat, metrics in sorted(report["slices"]["by_category"].items()):
        bar = "█" * int(metrics["retrieval_accuracy"] * 20)
        print(f"  {cat:20s} {bar:20s} {metrics['retrieval_accuracy']:.1%} "
              f"(n={metrics['total']})")

    print("\n--- Performance by difficulty ---")
    for diff, metrics in sorted(report["slices"]["by_difficulty"].items()):
        bar = "█" * int(metrics["retrieval_accuracy"] * 20)
        print(f"  {diff:10s} {bar:20s} {metrics['retrieval_accuracy']:.1%} "
              f"(n={metrics['total']})")

    print("\n--- Retrieved dataset distribution ---")
    for ds, metrics in sorted(report["slices"]["by_retrieved_dataset"].items()):
        bar = "█" * int(metrics["proportion"] * 40)
        print(f"  {ds:20s} {bar:20s} {metrics['proportion']:.1%} "
              f"(n={metrics['count']})")

    if report["disparities"]:
        print("\n--- Flagged disparities ---")
        for d in report["disparities"]:
            severity = "⚠️" if d["severity"] == "medium" else "🚨"
            print(f"  {severity} {d['type']}: {d['metric']}")
            if d["type"] == "performance_gap":
                print(f"     Best: {d['best_slice']} ({d['best_value']:.1%})")
                print(f"     Worst: {d['worst_slice']} ({d['worst_value']:.1%})")
                print(f"     Gap: {d['gap']:.1%}")
            elif d["type"] == "underperforming_slice":
                print(f"     Slice: {d['slice']} ({d['value']:.1%} vs avg {d['average']:.1%})")

    if report["recommendations"]:
        print("\n--- Recommendations ---")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"  {i}. {rec}")

    print("=" * 60)


if __name__ == "__main__":
    import json

    # Try to load saved eval results
    try:
        with open("evaluation/eval_results.json") as f:
            eval_summary = json.load(f)
        report = run_bias_analysis(eval_summary)
        print_bias_report(report)
    except FileNotFoundError:
        print("No eval_results.json found. Run eval_runner.py first.")
