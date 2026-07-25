import json

import numpy as np

from repro.aicrowd_whestbench.analyze_mlp_variance import (
    bootstrap_report_rankings,
    generate_official_weights,
)


def test_generate_official_weights_is_deterministic_he_normal_float32():
    first = generate_official_weights(12345, width=64, depth=4)
    second = generate_official_weights(12345, width=64, depth=4)

    assert first.shape == (4, 64, 64)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert abs(float(np.mean(first))) < 0.01
    assert abs(float(np.var(first)) - 2.0 / 64) < 0.002


def test_rank_bootstrap_reads_aligned_full100_registry(tmp_path):
    names = [f"mlp-{index}" for index in range(100)]
    report_paths = []
    for method_index, multiplier in enumerate((1.0, 1.2)):
        report = {
            "results": {
                "per_mlp": [
                    {
                        "mlp_name": name,
                        "adjusted_final_layer_score": multiplier
                        * (1.0 + index / 1000),
                    }
                    for index, name in enumerate(names)
                ]
            }
        }
        path = tmp_path / f"report-{method_index}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        report_paths.append(path)

    registry = {
        "entries": [
            {
                "submission_id": 1,
                "method": "better",
                "local_eval": str(report_paths[0]),
            },
            {
                "submission_id": 2,
                "method": "worse",
                "local_eval": str(report_paths[1]),
            },
        ]
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    result = bootstrap_report_rankings(
        registry_path=registry_path,
        official_status_path=None,
        suite_size=50,
        repeats=100,
        top_methods=0,
        seed=7,
    )

    assert result["selected_method_count"] == 2
    assert result["full100_top1_retained_probability"] == 1.0
    assert result["methods"][0]["method"] == "better"
