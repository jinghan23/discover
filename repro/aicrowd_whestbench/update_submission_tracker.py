#!/usr/bin/env python3
"""Refresh the local-vs-official WhestBench submission tracker."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RECORD_DIR = ROOT / "repro_external" / "aicrowd_whestbench"
REGISTRY_PATH = RECORD_DIR / "submissions" / "registry.json"
CACHE_PATH = RECORD_DIR / "submissions" / "official_status.json"
REPORT_PATH = RECORD_DIR / "SUBMISSION_TRACKER.md"
SUBMISSION_URL = (
    "https://www.aicrowd.com/challenges/"
    "arc-white-box-estimation-challenge-2026/submissions/{submission_id}"
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _official_snapshot(entries: list[dict[str, Any]], offline: bool) -> dict[str, Any]:
    cached: dict[str, Any] = {}
    if CACHE_PATH.exists():
        cached = _read_json(CACHE_PATH)
    statuses = dict(cached.get("submissions", {}))

    if not offline:
        from whestbench.aicrowd_client import AIcrowdClient
        from whestbench.aicrowd_config import load_api_key

        client = AIcrowdClient(api_key=load_api_key())
        for entry in entries:
            submission_id = entry["submission_id"]
            try:
                raw = client.get_submission_status(submission_id)
                statuses[str(submission_id)] = {
                    "created_at": raw.get("created_at"),
                    "grading_message": raw.get("grading_message"),
                    "score": raw.get("score"),
                    "score_secondary": raw.get("score_secondary"),
                    "status": raw.get("grading_status_cd"),
                }
            except Exception as exc:  # Keep the last good value for transient API errors.
                previous = statuses.get(str(submission_id), {})
                previous["refresh_error"] = f"{type(exc).__name__}: {exc}"
                statuses[str(submission_id)] = previous

    snapshot = {
        "challenge_slug": "arc-white-box-estimation-challenge-2026",
        "refreshed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "submissions": statuses,
    }
    if not offline:
        _write_json(CACHE_PATH, snapshot)
    return snapshot


def _rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[indexed[position][0]] = rank
        start = end
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_norm = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_norm = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if x_norm == 0.0 or y_norm == 0.0:
        return None
    return numerator / (x_norm * y_norm)


def _group_stats(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    comparable = [row for row in rows if row["official_score"] is not None]
    local = [row["local_score"] for row in comparable]
    official = [row["official_score"] for row in comparable]
    deltas = [(remote / nearby - 1.0) * 100.0 for nearby, remote in zip(local, official)]
    return {
        "count": len(comparable),
        "mean_delta_pct": statistics.fmean(deltas) if deltas else None,
        "median_abs_delta_pct": statistics.median(abs(value) for value in deltas) if deltas else None,
        "pearson": _pearson(local, official),
        "spearman": _pearson(_rank(local), _rank(official)),
    }


def _fmt_score(value: float | None) -> str:
    return "-" if value is None else f"`{value:.6e}`"


def _fmt_number(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _build_rows(
    entries: list[dict[str, Any]], snapshot: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    official = snapshot.get("submissions", {})
    for entry in entries:
        local_path = ROOT / entry["local_eval"]
        artifact_path = ROOT / entry["artifact"]
        local = _read_json(local_path)
        dataset = local["dataset"]
        summary = local["summary"]
        remote = official.get(str(entry["submission_id"]), {})
        official_score = remote.get("score")
        official_mse = remote.get("score_secondary")
        current_artifact_sha256 = _sha256(artifact_path)
        submitted_artifact_sha256 = entry.get(
            "artifact_sha256",
            current_artifact_sha256,
        )
        rows.append(
            {
                **entry,
                "artifact_current_sha256": current_artifact_sha256,
                "artifact_hash_matches": (
                    submitted_artifact_sha256 == current_artifact_sha256
                ),
                "artifact_sha256": submitted_artifact_sha256,
                "created_at": remote.get("created_at"),
                "local_cb": summary.get("mean_compute_utilization"),
                "local_eval_sha256": local.get("code_sha256"),
                "local_failures": summary.get("n_failed_mlps"),
                "local_final_mse": summary.get("final_layer_mse"),
                "local_n": dataset.get("n_mlps"),
                "local_score": summary.get("adjusted_final_layer_score"),
                "local_source": local.get("source"),
                "official_implied_multiplier": (
                    official_score / official_mse
                    if official_score is not None and official_mse not in (None, 0)
                    else None
                ),
                "official_mse": official_mse,
                "official_score": official_score,
                "official_status": remote.get("status", "unknown"),
                "refresh_error": remote.get("refresh_error"),
            }
        )
    return rows


def _render_report(rows: list[dict[str, Any]], snapshot: dict[str, Any]) -> str:
    graded = [row for row in rows if row["official_score"] is not None]
    best_official = min(graded, key=lambda row: row["official_score"])
    protocol_groups: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        protocol_groups.setdefault(row["local_n"], []).append(row)

    lines = [
        "# WhestBench Submission Tracker",
        "",
        f"Generated at `{snapshot['refreshed_at']}` from the local registry and the AIcrowd API.",
        "Lower adjusted score is better. This file is generated; do not edit it by hand.",
        "",
        "## Summary",
        "",
        f"- registered submissions: **{len(rows)}**; graded with an official score: **{len(graded)}**",
        (
            f"- best official result among registered submissions: "
            f"[#{best_official['submission_id']}]"
            f"({SUBMISSION_URL.format(submission_id=best_official['submission_id'])}) "
            f"{best_official['method']} at `{best_official['official_score']:.9e}`"
        ),
        "- official score is measured on the hosted public-50 suite; local rows use either "
        "mini first-10 or mini full-100 and are not interchangeable",
        "- protocol and column glossary: [`docs/PROTOCOLS.md`](docs/PROTOCOLS.md)",
        "",
        "## Local-to-official drift",
        "",
        "`Delta` is `(official / local - 1) * 100%`; negative means the official public score "
        "was lower (better). Correlations measure ordering, not absolute agreement.",
        "",
        "| Local protocol | Submissions | Mean delta | Median abs delta | Pearson r | Spearman rho |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for local_n in sorted(protocol_groups, reverse=True):
        stats = _group_stats(protocol_groups[local_n])
        lines.append(
            f"| mini n={local_n} | {stats['count']} | "
            f"{_fmt_number(stats['mean_delta_pct'])}% | "
            f"{_fmt_number(stats['median_abs_delta_pct'])}% | "
            f"{_fmt_number(stats['pearson'])} | {_fmt_number(stats['spearman'])} |"
        )

    family_groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        family_groups.setdefault((row["local_n"], row["family"]), []).append(row)
    lines.extend(
        [
            "",
            "### Drift by protocol and family",
            "",
            "| Local protocol | Family | Submissions | Mean delta | Median abs delta | Spearman rho |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for (local_n, family), grouped_rows in sorted(
        family_groups.items(), key=lambda item: (-item[0][0], item[0][1])
    ):
        stats = _group_stats(grouped_rows)
        lines.append(
            f"| mini n={local_n} | {family} | {stats['count']} | "
            f"{_fmt_number(stats['mean_delta_pct'])}% | "
            f"{_fmt_number(stats['median_abs_delta_pct'])}% | "
            f"{_fmt_number(stats['spearman'])} |"
        )

    lines.extend(
        [
            "",
            "## All submissions",
            "",
            "Official `MSE` is AIcrowd's `score_secondary` for these Phase 1 submissions. "
            "`S/MSE` is the ratio of the two hosted aggregates; treat it as an implied compute "
            "multiplier, not an exact mean C/B.",
            "",
            "| ID | Method | Local n | Local adjusted | Local MSE | Local C/B | Fail | Official adjusted | Official MSE | S/MSE | Delta | Status |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(rows, key=lambda item: item["submission_id"]):
        delta = (
            (row["official_score"] / row["local_score"] - 1.0) * 100.0
            if row["official_score"] is not None
            else None
        )
        delta_text = "-" if delta is None else f"{_fmt_number(delta, 2)}%"
        link = SUBMISSION_URL.format(submission_id=row["submission_id"])
        lines.append(
            f"| [#{row['submission_id']}]({link}) | {row['method']} | {row['local_n']} | "
            f"{_fmt_score(row['local_score'])} | {_fmt_score(row['local_final_mse'])} | "
            f"{_fmt_number(row['local_cb'], 4)} | {row['local_failures']} | "
            f"{_fmt_score(row['official_score'])} | {_fmt_score(row['official_mse'])} | "
            f"{_fmt_number(row['official_implied_multiplier'], 4)} | "
            f"{delta_text} | {row['official_status']} |"
        )

    lines.extend(
        [
            "",
            "## Provenance",
            "",
            "| ID | Family | Local source | Local evaluation | Artifact | Submitted SHA-256 |",
            "|---:|---|---|---|---|---|",
        ]
    )
    for row in sorted(rows, key=lambda item: item["submission_id"]):
        lines.append(
            f"| #{row['submission_id']} | {row['family']} | `{row['local_source']}` | "
            f"`{row['local_eval']}` | `{row['artifact']}` | "
            f"`{row['artifact_sha256']}` |"
        )

    artifact_mismatches = [row for row in rows if not row["artifact_hash_matches"]]
    if artifact_mismatches:
        lines.extend(
            [
                "",
                "### Artifact integrity notes",
                "",
                "The registry pins the SHA-256 of the archive that was actually submitted. "
                "A mismatch means the same local path was rebuilt later; it must not silently "
                "replace the submitted provenance.",
                "",
            ]
        )
        for row in artifact_mismatches:
            lines.append(
                f"- #{row['submission_id']}: submitted `{row['artifact_sha256']}`, "
                f"current file `{row['artifact_current_sha256']}` at `{row['artifact']}`."
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Compare absolute local and official values only within the same local protocol. "
            "A first-10 estimate has high subset variance and was also used during candidate "
            "selection, so it is optimistically biased.",
            "- The local full-100 suite and hosted public-50 suite have different aggregation "
            "sets. A systematic score shift does not by itself demonstrate an evaluator bug.",
            "- Prefer full-100 results for promotion decisions. Use first-10 only for failure, "
            "runtime, and rough budget screening.",
            "",
            "## Refresh",
            "",
            "```bash",
            "PYTHONPATH=/tmp/whest-official-deps:$PWD \\",
            "  python repro/aicrowd_whestbench/update_submission_tracker.py",
            "```",
            "",
            "Use `--offline` to regenerate from the last cached official API response. Add new "
            "submission mappings to "
            "`repro_external/aicrowd_whestbench/submissions/registry.json` "
            "before refreshing.",
            "",
        ]
    )
    errors = [row for row in rows if row["refresh_error"]]
    if errors:
        lines.extend(["## Refresh warnings", ""])
        for row in errors:
            lines.append(f"- #{row['submission_id']}: `{row['refresh_error']}`")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use the last official status cache instead of querying AIcrowd.",
    )
    args = parser.parse_args()

    registry = _read_json(REGISTRY_PATH)
    entries = registry["entries"]
    submission_ids = [entry["submission_id"] for entry in entries]
    if len(submission_ids) != len(set(submission_ids)):
        raise ValueError("submissions/registry.json contains duplicate submission IDs")

    snapshot = _official_snapshot(entries, offline=args.offline)
    rows = _build_rows(entries, snapshot)
    REPORT_PATH.write_text(_render_report(rows, snapshot), encoding="utf-8")
    print(f"updated {REPORT_PATH.relative_to(ROOT)} with {len(rows)} submissions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
