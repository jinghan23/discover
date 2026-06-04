#!/usr/bin/env python3
"""Run EvolveAgent for one copied AlphaResearchComp problem."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path


RUNS_ROOT = Path(__file__).resolve().parents[1]
ALPHARESEARCH_ROOT = RUNS_ROOT.parent / "_source"


def _missing(value: object) -> bool:
    return value is None or value == "" or value == (None,)


def _patch_api_config(config: object, model: str | None, api_base: str | None) -> None:
    env_key = os.environ.get("OPENAI_API_KEY")
    env_base = api_base or os.environ.get("OPENAI_API_BASE")
    env_model = model or os.environ.get("OPENAI_MODEL")

    for item in list(config.llm.models) + list(config.llm.evaluator_models):
        if env_model:
            item.name = env_model
        if _missing(item.api_key):
            item.api_key = env_key
        if env_base and _missing(item.api_base):
            item.api_base = env_base

    if config.rewardmodel.model_type == "api":
        if env_model:
            config.rewardmodel.model_name = env_model
        if _missing(config.rewardmodel.api_key):
            config.rewardmodel.api_key = env_key
        if env_base and _missing(config.rewardmodel.base_url):
            config.rewardmodel.base_url = env_base


async def run_async(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(ALPHARESEARCH_ROOT))

    try:
        from evolve_agent import EvolveAgent
        from evolve_agent.config import load_config
    except Exception as exc:
        print("Failed to import EvolveAgent from public _source.")
        print("This usually means optional dependencies are missing, for example vllm.")
        print(f"Import error: {exc!r}")
        return 1

    problem_dir = RUNS_ROOT / args.problem
    work_dir = problem_dir / "work"
    if not work_dir.exists():
        print(f"Problem work directory not found: {work_dir}")
        return 2

    config_path = Path(args.config).resolve()
    config = load_config(str(config_path))
    _patch_api_config(config, model=args.model, api_base=args.api_base)
    config.rewardmodel.jsonl_file = str(problem_dir / "evolve_agent_output" / "reward_results.jsonl")

    output_dir = Path(args.output).resolve() if args.output else problem_dir / "evolve_agent_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    old_cwd = Path.cwd()
    try:
        os.chdir(work_dir)
        agent = EvolveAgent(
            initial_program_path=str(work_dir / "initial_program.py"),
            initial_proposal_path=str(work_dir / "initial_proposal.txt"),
            evaluation_file=str(work_dir / "evaluator.py"),
            config=config,
            output_dir=str(output_dir),
        )
        best = await agent.run(iterations=args.iterations, target_score=args.target_score)
    finally:
        os.chdir(old_cwd)

    print("Evolution complete")
    print(f"Problem: {args.problem}")
    print(f"Output: {output_dir}")
    print(f"Best metrics: {best.metrics}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("problem")
    parser.add_argument("--iterations", "-i", type=int, default=5)
    parser.add_argument("--target-score", "-t", type=float, default=None)
    parser.add_argument("--config", "-c", default=str(RUNS_ROOT / "_common" / "config_quick.yaml"))
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-base", default=None)
    args = parser.parse_args()
    return asyncio.run(run_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
