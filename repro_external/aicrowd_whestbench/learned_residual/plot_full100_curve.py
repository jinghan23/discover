"""Plot local full-100 monitoring records from a learned-residual run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _monitor_records(metrics_path: Path) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    with metrics_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if (
                record.get("event") == "full100_monitor"
                and record.get("return_code") == 0
            ):
                records.append(record)
    if not records:
        raise ValueError(f"no successful full-100 records in {metrics_path}")
    return records


def plot_curve(run_dir: Path, output_path: Path, title: str) -> None:
    records = _monitor_records(run_dir / "metrics.jsonl")
    steps = [int(record["step"]) for record in records]
    final_mse = [float(record["full100_final_layer_mse"]) for record in records]
    all_layer_mse = [float(record["full100_all_layers_mse"]) for record in records]
    adjusted = [
        float(record["full100_adjusted_final_layer_score"]) for record in records
    ]
    failed = [int(record["full100_n_failed_mlps"]) for record in records]

    best_index = min(range(len(records)), key=final_mse.__getitem__)
    final_improvement = 100.0 * (1.0 - final_mse[-1] / final_mse[0])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (mse_axis, score_axis) = plt.subplots(
        2,
        1,
        figsize=(20, 13),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 0.86], "hspace": 0.07},
    )
    fig.suptitle(title, fontsize=29, fontweight="bold", y=0.975)
    fig.text(
        0.5,
        0.94,
        (
            f"{len(records)} monitoring evaluations · "
            f"{max(failed)}/100 failures max · "
            f"final improvement {final_improvement:.1f}%"
        ),
        ha="center",
        fontsize=16,
        color="#475569",
    )

    mse_axis.plot(
        steps,
        final_mse,
        color="#2563eb",
        marker="o",
        markersize=9,
        linewidth=3.2,
        label="Final-layer MSE",
    )
    mse_axis.plot(
        steps,
        all_layer_mse,
        color="#f59e0b",
        marker="s",
        markersize=8,
        linewidth=3.0,
        label="All-layer MSE",
    )
    mse_axis.scatter(
        [steps[best_index]],
        [final_mse[best_index]],
        s=310,
        facecolors="none",
        edgecolors="#dc2626",
        linewidths=3,
        zorder=5,
    )
    annotation_x = steps[best_index] - max(steps[-1] * 0.17, 1.0)
    annotation_y = final_mse[best_index] + (
        max(final_mse) - min(final_mse)
    ) * 0.20
    mse_axis.annotate(
        (
            "best monitored final MSE\n"
            f"{final_mse[best_index]:.3e} @ {steps[best_index]:,}"
        ),
        xy=(steps[best_index], final_mse[best_index]),
        xytext=(annotation_x, annotation_y),
        color="#b91c1c",
        fontsize=15,
        arrowprops={"arrowstyle": "->", "color": "#dc2626", "lw": 1.8},
    )
    mse_axis.set_ylabel("Raw MSE", fontsize=15)
    mse_axis.tick_params(labelsize=13)
    mse_axis.legend(loc="upper right", fontsize=14)
    mse_axis.margins(x=0.05, y=0.16)

    score_axis.semilogy(
        steps,
        adjusted,
        color="#059669",
        marker="o",
        markersize=9,
        linewidth=3.2,
        label="Adjusted final-layer score",
    )
    score_axis.axhline(
        1e-8,
        color="#dc2626",
        linestyle="--",
        linewidth=2.2,
        label="1e-8 target",
    )
    score_axis.scatter(
        [steps[best_index]],
        [adjusted[best_index]],
        s=310,
        facecolors="none",
        edgecolors="#dc2626",
        linewidths=3,
        zorder=5,
    )
    score_axis.annotate(
        f"final {adjusted[-1]:.3e}",
        xy=(steps[-1], adjusted[-1]),
        xytext=(steps[-1] * 0.86, adjusted[-1] / 2.3),
        color="#047857",
        fontsize=15,
        arrowprops={"arrowstyle": "->", "color": "#059669", "lw": 1.8},
    )
    score_axis.set_ylabel("Adjusted score (log scale)", fontsize=15)
    score_axis.set_xlabel("Training step", fontsize=15)
    score_axis.tick_params(labelsize=13)
    score_axis.legend(loc="lower left", fontsize=14)
    score_axis.set_ylim(7e-9, max(adjusted) * 1.35)
    score_axis.margins(x=0.05)

    for axis in (mse_axis, score_axis):
        axis.grid(True, which="both", alpha=0.24)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--title",
        default="Large Learned Residual - Local Full-100 During Training",
    )
    args = parser.parse_args()
    output = args.output or args.run_dir / "full100_training_curve.png"
    plot_curve(args.run_dir, output, args.title)
    print(output)


if __name__ == "__main__":
    main()
