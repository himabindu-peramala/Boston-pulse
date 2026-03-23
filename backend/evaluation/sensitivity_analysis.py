"""
Boston Pulse — Sensitivity Analysis
Measures how each parameter affects RAG performance.
Required by Model Development Guidelines.

Analyzes:
1. Dataset ablation — remove one dataset at a time, measure impact
2. Top-k sensitivity — how does changing k affect accuracy
3. Chunk content sensitivity — which fields matter most

Usage:
    cd backend
    python -m evaluation.sensitivity_analysis
"""
import logging


logger = logging.getLogger(__name__)


def analyze_topk_sensitivity(eval_results_by_k: dict) -> dict:
    """
    Compare evaluation results across different top_k values.

    Args:
        eval_results_by_k: dict mapping k value to eval_summary
            e.g. {3: summary_k3, 5: summary_k5, 10: summary_k10}

    Returns:
        Sensitivity report showing how k affects metrics.
    """
    if not eval_results_by_k:
        return {"error": "No results provided"}

    analysis = {}
    for k, summary in sorted(eval_results_by_k.items()):
        analysis[k] = {
            "retrieval_accuracy": summary.get("retrieval_accuracy", 0),
            "avg_keyword_coverage": summary.get("avg_keyword_coverage", 0),
            "avg_similarity_score": summary.get("avg_similarity_score", 0),
            "avg_response_time_ms": summary.get("avg_response_time_ms", 0),
        }

    # Find optimal k
    best_k = max(analysis.items(), key=lambda x: x[1]["retrieval_accuracy"])

    return {
        "parameter": "top_k",
        "values_tested": list(analysis.keys()),
        "results": analysis,
        "best_value": best_k[0],
        "best_accuracy": best_k[1]["retrieval_accuracy"],
        "insight": _generate_topk_insight(analysis),
    }


def _generate_topk_insight(analysis: dict) -> str:
    """Generate human-readable insight about top-k sensitivity."""
    k_values = sorted(analysis.keys())
    if len(k_values) < 2:
        return "Need at least 2 k values to analyze sensitivity."

    first_acc = analysis[k_values[0]]["retrieval_accuracy"]
    last_acc = analysis[k_values[-1]]["retrieval_accuracy"]
    diff = last_acc - first_acc

    if abs(diff) < 0.05:
        return (
            f"Retrieval accuracy is stable across k={k_values[0]} to "
            f"k={k_values[-1]} (change: {diff:+.1%}). "
            f"Top-k is not a sensitive parameter for this system."
        )
    elif diff > 0:
        return (
            f"Accuracy improves with higher k ({diff:+.1%} from "
            f"k={k_values[0]} to k={k_values[-1]}). More context helps."
        )
    else:
        return (
            f"Accuracy decreases with higher k ({diff:+.1%} from "
            f"k={k_values[0]} to k={k_values[-1]}). "
            f"Extra chunks add noise — lower k is better."
        )


def analyze_temperature_sensitivity(eval_results_by_temp: dict) -> dict:
    """
    Compare evaluation results across different temperature values.

    Args:
        eval_results_by_temp: dict mapping temperature to eval_summary

    Returns:
        Sensitivity report.
    """
    if not eval_results_by_temp:
        return {"error": "No results provided"}

    analysis = {}
    for temp, summary in sorted(eval_results_by_temp.items()):
        analysis[temp] = {
            "retrieval_accuracy": summary.get("retrieval_accuracy", 0),
            "avg_keyword_coverage": summary.get("avg_keyword_coverage", 0),
        }

    best_temp = max(analysis.items(), key=lambda x: x[1]["avg_keyword_coverage"])

    return {
        "parameter": "temperature",
        "values_tested": list(analysis.keys()),
        "results": analysis,
        "best_value": best_temp[0],
        "best_keyword_coverage": best_temp[1]["avg_keyword_coverage"],
        "insight": _generate_temp_insight(analysis),
    }


def _generate_temp_insight(analysis: dict) -> str:
    """Generate human-readable insight about temperature sensitivity."""
    temps = sorted(analysis.keys())
    if len(temps) < 2:
        return "Need at least 2 temperature values to analyze."

    low_kw = analysis[temps[0]]["avg_keyword_coverage"]
    high_kw = analysis[temps[-1]]["avg_keyword_coverage"]

    if low_kw > high_kw:
        return (
            f"Lower temperature ({temps[0]}) produces better keyword coverage "
            f"({low_kw:.1%} vs {high_kw:.1%}). Factual, deterministic "
            f"responses work better for civic data queries."
        )
    else:
        return (
            f"Higher temperature ({temps[-1]}) produces better keyword coverage "
            f"({high_kw:.1%} vs {low_kw:.1%}). More creative responses "
            f"capture more relevant information."
        )


def analyze_dataset_ablation(full_results: dict, ablation_results: dict) -> dict:
    """
    Compare full-dataset performance vs removing one dataset at a time.

    Args:
        full_results: eval_summary with all datasets
        ablation_results: dict mapping removed_dataset to eval_summary

    Returns:
        Report showing which datasets are most important.
    """
    if not full_results or not ablation_results:
        return {"error": "Need both full and ablation results"}

    full_accuracy = full_results.get("retrieval_accuracy", 0)
    impacts = {}

    for removed_ds, summary in ablation_results.items():
        ablated_accuracy = summary.get("retrieval_accuracy", 0)
        impact = full_accuracy - ablated_accuracy
        impacts[removed_ds] = {
            "accuracy_without": ablated_accuracy,
            "accuracy_drop": round(impact, 4),
            "importance": "high" if impact > 0.1 else "medium" if impact > 0.05 else "low",
        }

    # Rank by importance
    ranked = sorted(impacts.items(), key=lambda x: x[1]["accuracy_drop"], reverse=True)

    return {
        "analysis": "dataset_ablation",
        "full_accuracy": full_accuracy,
        "impacts": impacts,
        "ranking": [ds for ds, _ in ranked],
        "most_important": ranked[0][0] if ranked else None,
        "least_important": ranked[-1][0] if ranked else None,
    }


def print_sensitivity_report(topk_report: dict = None,
                              temp_report: dict = None,
                              ablation_report: dict = None):
    """Print a readable sensitivity analysis report."""
    print("\n" + "=" * 60)
    print("BOSTON PULSE RAG — SENSITIVITY ANALYSIS")
    print("=" * 60)

    if topk_report and "error" not in topk_report:
        print("\n--- Top-K Sensitivity ---")
        for k, metrics in sorted(topk_report["results"].items()):
            bar = "█" * int(metrics["retrieval_accuracy"] * 20)
            print(f"  k={k:<4}  {bar:20s}  {metrics['retrieval_accuracy']:.1%}")
        print(f"  Best: k={topk_report['best_value']}")
        print(f"  Insight: {topk_report['insight']}")

    if temp_report and "error" not in temp_report:
        print("\n--- Temperature Sensitivity ---")
        for temp, metrics in sorted(temp_report["results"].items()):
            bar = "█" * int(metrics["avg_keyword_coverage"] * 20)
            print(f"  t={temp:<5}  {bar:20s}  kw={metrics['avg_keyword_coverage']:.1%}")
        print(f"  Best: t={temp_report['best_value']}")
        print(f"  Insight: {temp_report['insight']}")

    if ablation_report and "error" not in ablation_report:
        print("\n--- Dataset Ablation ---")
        print(f"  Full accuracy: {ablation_report['full_accuracy']:.1%}")
        for ds, impact in sorted(
            ablation_report["impacts"].items(),
            key=lambda x: x[1]["accuracy_drop"],
            reverse=True,
        ):
            bar = "█" * int(impact["accuracy_drop"] * 40)
            print(
                f"  Remove {ds:20s}  drop={impact['accuracy_drop']:+.1%}  "
                f"[{impact['importance']}]"
            )
        print(f"  Most important dataset: {ablation_report['most_important']}")
        print(f"  Least important dataset: {ablation_report['least_important']}")

    print("=" * 60)


if __name__ == "__main__":
    print("Sensitivity analysis requires experiment results.")
    print("Run experiments first, then call the analysis functions.")
    print("\nExample usage:")
    print("  topk_report = analyze_topk_sensitivity({3: summary_k3, 5: summary_k5})")
    print("  print_sensitivity_report(topk_report=topk_report)")