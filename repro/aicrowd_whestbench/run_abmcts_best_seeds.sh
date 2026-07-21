#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash repro/aicrowd_whestbench/run_abmcts_best_seeds.sh [best|all|SEED ...] [options]

Seeds:
  meanonly affinef64 german013 gaussiancv rqmc030 sphericalrb layerrescale

Options:
  --search-only       Run AB-MCTS search and export each tree's best candidate
  --evaluate-only     Re-export existing tree winners and run local full-100
  --dry-run           Print resolved launch configs without starting search/eval
  -h, --help

Environment overrides:
  RUN_WHEST_ABMCTS_TAG=...                    Stable run suffix (default 20260720)
  RUN_WHEST_ABMCTS_LOG_ROOT=...               Default codex_runs/aicrowd_whestbench
  RUN_WHEST_ABMCTS_NUM_EPOCHS=...             Default 4
  RUN_WHEST_ABMCTS_GROUPS_PER_BATCH=...        Default 4; group_size is fixed at 1
  RUN_WHEST_ABMCTS_ROOT_MIN_WIDTH=...          Default 8
  RUN_WHEST_ABMCTS_SEARCH_N_MLPS=...           Search fidelity (default full-100)
  RUN_WHEST_ABMCTS_FULL_N_MLPS=...             Default full-100
  RUN_WHEST_ABMCTS_EVAL_TOP_K=...              Novel tree candidates to full-evaluate (default 2)
  RUN_WHEST_ABMCTS_SMOKE=1                    1 epoch, width 4, first-1, eval first-10
  RUN_WHEST_ABMCTS_FORCE_EVAL=1                Replace an existing full-100 report
  RUN_WHEST_ABMCTS_CODEX_MODEL=...             Default gpt-5.5
  RUN_WHEST_ABMCTS_REASONING_EFFORT=...         Default high; smoke defaults medium
  RUN_WHEST_ABMCTS_CLI_TIMEOUT=...              Per-action timeout in seconds (default 600)
  PYTHON=...                                   Default .venv/bin/python
  HF_HOME=...                                  Default /tmp/hf-whest-cache
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ALL_SEEDS=(meanonly affinef64 german013 gaussiancv rqmc030 sphericalrb layerrescale)
BEST_SEEDS=(meanonly affinef64 german013 gaussiancv)
SEEDS=()
MODE=all
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        best)
            SEEDS+=("${BEST_SEEDS[@]}")
            ;;
        all)
            SEEDS+=("${ALL_SEEDS[@]}")
            ;;
        meanonly|affinef64|german013|gaussiancv|rqmc030|sphericalrb|layerrescale)
            SEEDS+=("$1")
            ;;
        --search-only)
            MODE=search
            ;;
        --evaluate-only)
            MODE=evaluate
            ;;
        --dry-run)
            DRY_RUN=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if [[ ${#SEEDS[@]} -eq 0 ]]; then
    SEEDS=("${BEST_SEEDS[@]}")
fi

# Preserve order while removing duplicates.
UNIQUE_SEEDS=()
for seed in "${SEEDS[@]}"; do
    seen=0
    for existing in "${UNIQUE_SEEDS[@]-}"; do
        if [[ "$existing" == "$seed" ]]; then
            seen=1
            break
        fi
    done
    if [[ "$seen" == 0 ]]; then
        UNIQUE_SEEDS+=("$seed")
    fi
done
SEEDS=("${UNIQUE_SEEDS[@]}")

PYTHON_BIN="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
LOG_ROOT="${RUN_WHEST_ABMCTS_LOG_ROOT:-$REPO_ROOT/codex_runs/aicrowd_whestbench}"
TAG="${RUN_WHEST_ABMCTS_TAG:-20260720}"
NUM_EPOCHS="${RUN_WHEST_ABMCTS_NUM_EPOCHS:-4}"
GROUPS_PER_BATCH="${RUN_WHEST_ABMCTS_GROUPS_PER_BATCH:-4}"
ROOT_MIN_WIDTH="${RUN_WHEST_ABMCTS_ROOT_MIN_WIDTH:-8}"
SEARCH_N_MLPS="${RUN_WHEST_ABMCTS_SEARCH_N_MLPS:-100}"
FULL_N_MLPS="${RUN_WHEST_ABMCTS_FULL_N_MLPS:-100}"
EVAL_TOP_K="${RUN_WHEST_ABMCTS_EVAL_TOP_K:-2}"
CODEX_MODEL="${RUN_WHEST_ABMCTS_CODEX_MODEL:-gpt-5.5}"
REASONING_EFFORT="${RUN_WHEST_ABMCTS_REASONING_EFFORT:-high}"
DEFAULT_CODEX_COMMAND=codex
if [[ -x /Applications/ChatGPT.app/Contents/Resources/codex ]]; then
    DEFAULT_CODEX_COMMAND=/Applications/ChatGPT.app/Contents/Resources/codex
fi
CODEX_COMMAND="${RUN_WHEST_ABMCTS_CODEX_COMMAND:-$DEFAULT_CODEX_COMMAND}"
CLI_TIMEOUT="${RUN_WHEST_ABMCTS_CLI_TIMEOUT:-600}"
EVAL_TIMEOUT="${RUN_WHEST_ABMCTS_EVAL_TIMEOUT:-4000}"
MAX_CONCURRENT_REQUESTS="${RUN_WHEST_ABMCTS_MAX_CONCURRENT_REQUESTS:-4}"
HF_HOME="${HF_HOME:-/tmp/hf-whest-cache}"
SEED_SOURCE_ROOT="$LOG_ROOT/_abmcts_seed_sources"

if [[ "${RUN_WHEST_ABMCTS_SMOKE:-0}" == "1" ]]; then
    NUM_EPOCHS="${RUN_WHEST_ABMCTS_NUM_EPOCHS:-1}"
    GROUPS_PER_BATCH="${RUN_WHEST_ABMCTS_GROUPS_PER_BATCH:-4}"
    ROOT_MIN_WIDTH="${RUN_WHEST_ABMCTS_ROOT_MIN_WIDTH:-4}"
    SEARCH_N_MLPS="${RUN_WHEST_ABMCTS_SEARCH_N_MLPS:-1}"
    FULL_N_MLPS="${RUN_WHEST_ABMCTS_FULL_N_MLPS:-10}"
    REASONING_EFFORT="${RUN_WHEST_ABMCTS_REASONING_EFFORT:-medium}"
fi

export HF_HOME
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$DRY_RUN" != "1" ]]; then
    if [[ ! -x "$PYTHON_BIN" ]]; then
        echo "Python executable not found: $PYTHON_BIN" >&2
        exit 1
    fi
    "$PYTHON_BIN" -c 'import flopscope, whestbench' 2>/dev/null || {
        echo "Missing pinned WhestBench dependencies in $PYTHON_BIN." >&2
        echo "Run: uv pip install --python .venv/bin/python 'whestbench==0.12.0rc5' 'flopscope==0.8.0rc5'" >&2
        exit 1
    }
fi

seed_spec() {
    case "$1" in
        meanonly)
            echo "autoevolve_n100_mean_only_bff13d51_phase1_20260716.tar.gz|badd395c7264afe62873e9fd61c5a271acc6bb2cbca5322ca7dedc9bcd80c7a0"
            ;;
        affinef64)
            echo "autoevolve_n100_affine_f64_1cb0eb3a_phase1_20260716.tar.gz|1cc9ebd9dc10f1330c149c162c9e1c869f189da20d70dc48af587caed5eead70"
            ;;
        german013)
            echo "german_budget013_phase1_20260714.tar.gz|a1395b694c3bbbeb717ee2f8430f3677d90983c4fa7e8e86a17669da1b453a4f"
            ;;
        gaussiancv)
            echo "autoevolve_gaussian_cv_25d94716_phase1_20260714.tar.gz|515139e490d642b6350d3fc81171c3c2942304a7434f5137f2ac0a149177c809"
            ;;
        rqmc030)
            echo "rqmc_budget030_phase1_20260714.tar.gz|6288a7fb59c7ae0db30d5f9fa75176c87fcd4c08694149660630a966d39b3a09"
            ;;
        sphericalrb)
            # This is the current repackaged duplicate (#316628), not the overwritten
            # archive SHA originally submitted as #316625.
            echo "whitened_antithetic_spherical_rqmc_rb_budget010_phase1_20260716.tar.gz|309f97f132bc87c9bd4ec771ecf19b122cda5be03b0a27551e346bfffb1dc778"
            ;;
        layerrescale)
            echo "autoevolve_layer_rescale_f4443aad_phase1_20260714.tar.gz|ff4437d67e38527b37435360f0c4ecdbd778f1d28a09398ebeb91903abe0ada5"
            ;;
    esac
}

sha256_file() {
    shasum -a 256 "$1" | awk '{print $1}'
}

prepare_seed() {
    local seed="$1"
    local spec package_name expected_sha package_path actual_sha seed_dir seed_path
    spec="$(seed_spec "$seed")"
    IFS='|' read -r package_name expected_sha <<<"$spec"
    package_path="$REPO_ROOT/repro_external/aicrowd_whestbench/submissions/packages/$package_name"
    if [[ ! -f "$package_path" ]]; then
        echo "Missing seed package: $package_path" >&2
        exit 1
    fi
    actual_sha="$(sha256_file "$package_path")"
    if [[ "$actual_sha" != "$expected_sha" ]]; then
        echo "Seed package checksum mismatch for $seed" >&2
        echo "expected: $expected_sha" >&2
        echo "actual:   $actual_sha" >&2
        exit 1
    fi

    seed_dir="$SEED_SOURCE_ROOT/$seed"
    seed_path="$seed_dir/estimator.py"
    if [[ "$DRY_RUN" != "1" ]]; then
        mkdir -p "$seed_dir"
        tar -xOzf "$package_path" estimator.py >"$seed_path.tmp"
        mv "$seed_path.tmp" "$seed_path"
        sha256_file "$seed_path" >"$seed_dir/estimator.sha256"
        printf '%s\n' "$actual_sha" >"$seed_dir/archive.sha256"
    fi
    echo "$seed_path"
}

run_search() {
    local seed="$1" seed_path="$2" experiment_name="$3" log_dir="$4"
    local config_file
    config_file="$(mktemp /tmp/run_whest_abmcts.XXXXXX)"

    cat >"$config_file" <<YAML
defaults:
  env_type: examples.aicrowd_whestbench.env:WhestBenchEnv
  problem_type: arc_whestbench_2026
  algorithm: abmcts
  eval_runner: in_process
  backend: cli
  model_name: "$CODEX_MODEL"
  cli_command: "$CODEX_COMMAND"
  cli_sandbox: read-only
  cli_timeout: $CLI_TIMEOUT
  cli_reasoning_effort: "$REASONING_EFFORT"
  eval_timeout: $EVAL_TIMEOUT
  num_cpus_per_task: 1
  max_concurrent_requests: $MAX_CONCURRENT_REQUESTS
  group_size: 1
  groups_per_batch: $GROUPS_PER_BATCH
  remove_constant_reward_groups: false
  wandb_project: ""
  abmcts_variant: a
  abmcts_actions:
    - local_refine
    - rqmc_sampling
    - analytic_moments
    - hybrid_control_variate
  abmcts_root_min_width: $ROOT_MIN_WIDTH
  abmcts_dist_type: gaussian
  abmcts_model_selection_strategy: multiarm_bandit_thompson
  abmcts_invalid_score: -100.0
  abmcts_reward_scale: 10000000.0
  abmcts_prior_mean: -3.0
  abmcts_prior_std: 1.0
  abmcts_prior_strength: 1.0
  abmcts_seed: 20260720
  env:
    PYTHONUNBUFFERED: "1"
    HF_HOME: "$HF_HOME"
    WHEST_DATASET: aicrowd/arc-whestbench-public-2026
    WHEST_DATASET_REVISION: v1-phase1
    WHEST_DATASET_SPLIT: mini
    WHEST_DATASET_STREAMING: "0"
    WHEST_N_MLPS: "$SEARCH_N_MLPS"
    WHEST_FLOP_BUDGET: "272000000000"
    WHEST_LAMBDA_FLOPS_PER_SECOND: "100000000000"
    WHEST_WALL_TIME_LIMIT: "60"
    WHEST_RUNNER: subprocess
    WHEST_INITIAL_ESTIMATOR_PATH: "$seed_path"

runs:
  search:
    experiment_name: "$experiment_name"
    log_path: "$log_dir"
    num_epochs: $NUM_EPOCHS
YAML

    echo "[$seed] AB-MCTS search: n=$SEARCH_N_MLPS epochs=$NUM_EPOCHS batch=$GROUPS_PER_BATCH root_width=$ROOT_MIN_WIDTH"
    if [[ "$DRY_RUN" == "1" ]]; then
        "$PYTHON_BIN" repro/run_from_yaml.py --config "$config_file" --run search --dry-run
    else
        "$PYTHON_BIN" repro/run_from_yaml.py --config "$config_file" --run search
    fi
    rm -f "$config_file"
}

export_and_evaluate() {
    local seed="$1" log_dir="$2"
    local promoted_dir="" candidate="" summary="" report="" best_node=""
    local rank=0 ranked_candidate="" ranked_summary="" ranked_report="" ranked_node=""
    promoted_dir="$log_dir/promoted"
    candidate="$promoted_dir/${seed}_abmcts_best.py"
    summary="$promoted_dir/${seed}_abmcts_best.json"
    report="$promoted_dir/${seed}_abmcts_best_full${FULL_N_MLPS}.json"

    mkdir -p "$promoted_dir"
    "$PYTHON_BIN" repro/aicrowd_whestbench/select_abmcts_best.py \
        "$log_dir" "$candidate" --summary "$summary"
    best_node="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["node_id"])' "$summary")"
    if [[ -f "$report" && "${RUN_WHEST_ABMCTS_FORCE_EVAL:-0}" != "1" ]]; then
        echo "[$seed] keeping existing evaluation: $report"
    else
        echo "[$seed] evaluating promoted candidate on local mini n=$FULL_N_MLPS"
        "$PYTHON_BIN" repro/aicrowd_whestbench/evaluate_public_reproduction.py \
            "$candidate" "$report" --n-mlps "$FULL_N_MLPS"
    fi

    for ((rank = 1; rank <= EVAL_TOP_K; rank++)); do
        ranked_candidate="$promoted_dir/${seed}_abmcts_novel_rank${rank}.py"
        ranked_summary="$promoted_dir/${seed}_abmcts_novel_rank${rank}.json"
        ranked_report="$promoted_dir/${seed}_abmcts_novel_rank${rank}_full${FULL_N_MLPS}.json"
        "$PYTHON_BIN" repro/aicrowd_whestbench/select_abmcts_best.py \
            "$log_dir" "$ranked_candidate" --summary "$ranked_summary" \
            --exclude-root --rank "$rank"
        ranked_node="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["node_id"])' "$ranked_summary")"
        if [[ "$ranked_node" == "$best_node" ]]; then
            echo "[$seed] novel rank $rank is already the promoted best (node $best_node)"
            continue
        fi
        if [[ -f "$ranked_report" && "${RUN_WHEST_ABMCTS_FORCE_EVAL:-0}" != "1" ]]; then
            echo "[$seed] keeping existing novel rank $rank evaluation: $ranked_report"
            continue
        fi
        echo "[$seed] evaluating novel rank $rank on local mini n=$FULL_N_MLPS"
        "$PYTHON_BIN" repro/aicrowd_whestbench/evaluate_public_reproduction.py \
            "$ranked_candidate" "$ranked_report" --n-mlps "$FULL_N_MLPS"
    done
}

mkdir -p "$LOG_ROOT"
for seed in "${SEEDS[@]}"; do
    seed_path="$(prepare_seed "$seed")"
    experiment_name="whestbench_abmcts_a_${seed}_${TAG}"
    log_dir="$LOG_ROOT/$experiment_name"

    if [[ "$MODE" != "evaluate" ]]; then
        run_search "$seed" "$seed_path" "$experiment_name" "$log_dir"
    fi
    if [[ "$DRY_RUN" != "1" && "$MODE" != "search" ]]; then
        export_and_evaluate "$seed" "$log_dir"
    elif [[ "$DRY_RUN" != "1" && "$MODE" == "search" ]]; then
        mkdir -p "$log_dir/promoted"
        "$PYTHON_BIN" repro/aicrowd_whestbench/select_abmcts_best.py \
            "$log_dir" "$log_dir/promoted/${seed}_abmcts_best.py" \
            --summary "$log_dir/promoted/${seed}_abmcts_best.json"
    fi
done
