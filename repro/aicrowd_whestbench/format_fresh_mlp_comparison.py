#!/usr/bin/env python3
"""Format local full-100, hosted, and fresh-suite ranks/scores as Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def format_table(artifact: dict[str, Any]) -> str:
    rows = sorted(
        artifact["results"]["methods"],
        key=lambda row: int(row["local_full100_rank"]),
    )
    lines = [
        "| 方法 | 原 full-100 排名 | 原 full-100 分数 | 官方排名 | 官方分数 | "
        "新100排名 | 新100分数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        method = f'#{row["submission_id"]} {row["method"]}'.replace("|", r"\|")
        lines.append(
            f"| {method} "
            f'| {row["local_full100_rank"]} '
            f'| {float(row["local_full100_adjusted_score"]):.6e} '
            f'| {row["hosted_rank_among_selected"]} '
            f'| {float(row["hosted_public50_adjusted_score"]):.6e} '
            f'| {row["fresh_flops_only_adjusted_rank"]} '
            f'| {float(row["fresh_corrected_flops_only_adjusted_score"]):.6e} |'
        )
    return "\n".join(lines) + "\n"


def format_split_table(
    fresh100: dict[str, Any],
    fresh50_a: dict[str, Any],
    fresh50_b: dict[str, Any],
) -> str:
    def by_submission(artifact: dict[str, Any]) -> dict[int, dict[str, Any]]:
        return {
            int(row["submission_id"]): row
            for row in artifact["results"]["methods"]
        }

    combined = by_submission(fresh100)
    first = by_submission(fresh50_a)
    second = by_submission(fresh50_b)
    if combined.keys() != first.keys() or combined.keys() != second.keys():
        raise ValueError("artifacts do not contain the same submission IDs")
    rows = sorted(
        combined.values(),
        key=lambda row: int(row["local_full100_rank"]),
    )
    lines = [
        "| 方法 | 新50-A排名 | 新50-A分数 | 新50-B排名 | 新50-B分数 | "
        "新100排名 | 新100分数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for combined_row in rows:
        submission_id = int(combined_row["submission_id"])
        first_row = first[submission_id]
        second_row = second[submission_id]
        method = f'#{submission_id} {combined_row["method"]}'.replace("|", r"\|")
        lines.append(
            f"| {method} "
            f'| {first_row["fresh_flops_only_adjusted_rank"]} '
            f'| {float(first_row["fresh_corrected_flops_only_adjusted_score"]):.6e} '
            f'| {second_row["fresh_flops_only_adjusted_rank"]} '
            f'| {float(second_row["fresh_corrected_flops_only_adjusted_score"]):.6e} '
            f'| {combined_row["fresh_flops_only_adjusted_rank"]} '
            f'| {float(combined_row["fresh_corrected_flops_only_adjusted_score"]):.6e} |'
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fresh50-a", type=Path)
    parser.add_argument("--fresh50-b", type=Path)
    parser.add_argument("--split-output", type=Path)
    args = parser.parse_args()
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    table = format_table(artifact)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(table, end="")
    split_paths = (args.fresh50_a, args.fresh50_b, args.split_output)
    if any(path is not None for path in split_paths):
        if not all(path is not None for path in split_paths):
            parser.error(
                "--fresh50-a, --fresh50-b, and --split-output must be used together"
            )
        first = json.loads(args.fresh50_a.read_text(encoding="utf-8"))
        second = json.loads(args.fresh50_b.read_text(encoding="utf-8"))
        split_table = format_split_table(artifact, first, second)
        args.split_output.parent.mkdir(parents=True, exist_ok=True)
        args.split_output.write_text(split_table, encoding="utf-8")
        print(f"wrote {args.split_output}")


if __name__ == "__main__":
    main()
