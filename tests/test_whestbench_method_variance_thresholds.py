import numpy as np

from repro.aicrowd_whestbench.analyze_estimator_mlp_thresholds import (
    analyze_thresholds,
)


def _method(submission_id, local_rank, fresh_rank, scores):
    return {
        "submission_id": submission_id,
        "method": f"method-{submission_id}",
        "local_full100_rank": local_rank,
        "fresh_flops_only_adjusted_rank": fresh_rank,
        "per_mlp": [
            {"corrected_flops_only_adjusted_score": score} for score in scores
        ],
    }


def test_threshold_analysis_uses_paired_per_mlp_differences():
    artifact = {
        "results": {
            "methods": [
                _method(1, 1, 1, [1.0, 2.0, 3.0, 4.0]),
                _method(2, 2, 2, [1.5, 2.5, 3.5, 4.5]),
            ]
        }
    }

    result = analyze_thresholds(artifact, target_sizes=(4,))

    first = result["method_across_mlp_variance"][0]
    pair = result["pairwise_details"][0]
    assert first["sample_variance"] == np.var([1.0, 2.0, 3.0, 4.0], ddof=1)
    assert pair["mean_left_minus_right"] == -0.5
    assert pair["difference_standard_deviation"] == 0.0
    assert (
        result["pairwise_summary"]["threshold_by_suite_size"]["4"][
            "one_sided_95"
        ]["median"]
        == 0.0
    )
