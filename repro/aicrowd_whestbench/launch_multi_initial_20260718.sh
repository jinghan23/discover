#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SESSION="${WHEST_MULTI_SESSION:-whest-multi-initial-0718}"
DEPS_PATH="${WHEST_DEPS_PATH:-/tmp/whest-official-deps}"
HF_CACHE="${HF_HOME:-/tmp/hf-whest-cache}"
RUN_ROOT="${WHEST_MULTI_RUN_ROOT:-$REPO_ROOT/codex_runs/aicrowd_whestbench/multi_initial_20260718}"
INITIAL_ROOT="$RUN_ROOT/initial_states"
CONSOLE_ROOT="$RUN_ROOT/console"
EXPERIMENT_PREFIX="${WHEST_EXPERIMENT_PREFIX:-whest_0718}"

readonly -a SEEDS=(
    german013
    affinef64
    gaussiancv
    rqmc030
    sphericalrb
    layerrescale
)

package_for_seed() {
    case "$1" in
        german013)
            echo "repro_external/aicrowd_whestbench/submissions/packages/german_budget013_phase1_20260714.tar.gz"
            ;;
        affinef64)
            echo "repro_external/aicrowd_whestbench/submissions/packages/autoevolve_n100_affine_f64_1cb0eb3a_phase1_20260716.tar.gz"
            ;;
        gaussiancv)
            echo "repro_external/aicrowd_whestbench/submissions/packages/autoevolve_gaussian_cv_25d94716_phase1_20260714.tar.gz"
            ;;
        rqmc030)
            echo "repro_external/aicrowd_whestbench/submissions/packages/rqmc_budget030_phase1_20260714.tar.gz"
            ;;
        sphericalrb)
            echo "repro_external/aicrowd_whestbench/submissions/packages/whitened_antithetic_spherical_rqmc_rb_budget010_phase1_20260716.tar.gz"
            ;;
        layerrescale)
            echo "repro_external/aicrowd_whestbench/submissions/packages/autoevolve_layer_rescale_f4443aad_phase1_20260714.tar.gz"
            ;;
        *)
            echo "unknown seed: $1" >&2
            return 2
            ;;
    esac
}

expected_hash_for_seed() {
    case "$1" in
        german013) echo "68d409403de29c8fa7807a8274066494e9ce00380bbbe5bf3d1d52e4ba35aa64" ;;
        affinef64) echo "1cb0eb3a875702386c346d605880190c7827b392860c7ddb6fb690f9f9b71435" ;;
        gaussiancv) echo "d49b8b5218ff9b9462a331812e688e125fcfff1f86e8a26e9be9c8f59eb48ebb" ;;
        rqmc030) echo "dc60550f76e9f4f6cb5940af295cab777a4082412590a49665e9127b535eb19f" ;;
        sphericalrb) echo "146a8c786ee31dfefc941712ded736c3ec2625343dd5cd184f80dde915a9d130" ;;
        layerrescale) echo "98da6a0882cbc02d32cbd20e295c80b21bb8c173e3e23bb0d6880233d611d7ff" ;;
        *)
            echo "unknown seed: $1" >&2
            return 2
            ;;
    esac
}

prepare_initial_states() {
    mkdir -p "$INITIAL_ROOT" "$CONSOLE_ROOT"
    local seed package seed_dir estimator actual expected
    for seed in "${SEEDS[@]}"; do
        package="$(package_for_seed "$seed")"
        seed_dir="$INITIAL_ROOT/$seed"
        estimator="$seed_dir/estimator.py"
        mkdir -p "$seed_dir"
        tar -xzf "$package" -C "$seed_dir" estimator.py
        actual="$(sha256sum "$estimator" | awk '{print $1}')"
        expected="$(expected_hash_for_seed "$seed")"
        if [[ "$actual" != "$expected" ]]; then
            echo "estimator hash mismatch for $seed: expected $expected, got $actual" >&2
            return 1
        fi
        python -m py_compile "$estimator"
    done
}

check_dependencies() {
    if [[ ! -d "$DEPS_PATH/whestbench" || ! -d "$DEPS_PATH/flopscope" ]]; then
        echo "missing WhestBench dependencies under $DEPS_PATH" >&2
        return 1
    fi
    if ! command -v codex >/dev/null 2>&1; then
        echo "codex CLI is not available" >&2
        return 1
    fi
    if ! command -v tmux >/dev/null 2>&1; then
        echo "tmux is not available" >&2
        return 1
    fi
    PYTHONPATH="$DEPS_PATH:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
        python -c 'import flopscope, whestbench'
}

run_one() {
    local kind="$1"
    local seed="$2"
    local replica="$3"
    local estimator="$INITIAL_ROOT/$seed/estimator.py"
    local experiment console_log

    export WHEST_DEPS_PATH="$DEPS_PATH"
    export PYTHONPATH="$DEPS_PATH:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    export HF_HOME="$HF_CACHE"
    export WHEST_INITIAL_ESTIMATOR_PATH="$estimator"
    export WHEST_SEARCH_N_MLPS="${WHEST_SEARCH_N_MLPS:-100}"
    export RUN_WHEST_LOG_ROOT="$RUN_ROOT"
    export RUN_WHEST_CODEX_MODEL="${RUN_WHEST_CODEX_MODEL:-gpt-5.6-sol}"
    export RUN_WHEST_CODEX_COMMAND="${RUN_WHEST_CODEX_COMMAND:-codex}"
    export RUN_WHEST_NUM_CPUS_PER_TASK="${RUN_WHEST_NUM_CPUS_PER_TASK:-1}"
    export RUN_WHEST_EVAL_TIMEOUT="${RUN_WHEST_EVAL_TIMEOUT:-4000}"
    export RUN_WHEST_WANDB_PROJECT="${RUN_WHEST_WANDB_PROJECT:-}"
    export OPENBLAS_NUM_THREADS="${WHEST_MULTI_BLAS_THREADS:-1}"
    export OMP_NUM_THREADS="${WHEST_MULTI_BLAS_THREADS:-1}"
    export MKL_NUM_THREADS="${WHEST_MULTI_BLAS_THREADS:-1}"
    export NUMEXPR_NUM_THREADS="${WHEST_MULTI_BLAS_THREADS:-1}"
    export PYTHONUNBUFFERED=1

    case "$kind" in
        ttt)
            experiment="${EXPERIMENT_PREFIX}_${seed}_ttt"
            console_log="$CONSOLE_ROOT/${experiment}.log"
            export RUN_WHEST_TTT_EXPERIMENT_NAME="$experiment"
            export RUN_WHEST_TTT_NUM_EPOCHS="${RUN_WHEST_TTT_NUM_EPOCHS:-50}"
            export RUN_WHEST_TTT_GROUP_SIZE="${RUN_WHEST_TTT_GROUP_SIZE:-4}"
            export RUN_WHEST_TTT_GROUPS_PER_BATCH="${RUN_WHEST_TTT_GROUPS_PER_BATCH:-1}"
            export RUN_WHEST_TTT_CLI_TIMEOUT="${RUN_WHEST_TTT_CLI_TIMEOUT:-900}"
            export RUN_WHEST_TTT_MAX_CONCURRENT_REQUESTS="${RUN_WHEST_TTT_MAX_CONCURRENT_REQUESTS:-4}"
            bash repro/aicrowd_whestbench/run_official_mini.sh ttt 2>&1 | tee -a "$console_log"
            ;;
        auto)
            experiment="${EXPERIMENT_PREFIX}_${seed}_auto_r${replica}"
            console_log="$CONSOLE_ROOT/${experiment}.log"
            export RUN_WHEST_AUTO_EXPERIMENT_NAME="$experiment"
            export RUN_WHEST_AUTO_NUM_EPOCHS="${RUN_WHEST_AUTO_NUM_EPOCHS:-20}"
            export RUN_WHEST_AUTO_GROUP_SIZE=1
            export RUN_WHEST_AUTO_GROUPS_PER_BATCH="${RUN_WHEST_AUTO_GROUPS_PER_BATCH:-1}"
            export RUN_WHEST_AUTO_CLI_TIMEOUT="${RUN_WHEST_AUTO_CLI_TIMEOUT:-21600}"
            export RUN_WHEST_AUTO_MAX_CONCURRENT_REQUESTS=1
            export RUN_WHEST_VERIFIER_WORKERS="${RUN_WHEST_VERIFIER_WORKERS:-1}"
            export TTT_BLACKBOX_EVAL_SOCKET="/tmp/wb_${seed}_a${replica}.sock"
            export TTT_BLACKBOX_EVAL_LOG_DIR="/tmp/wb_${seed}_a${replica}_logs"
            bash repro/aicrowd_whestbench/run_official_mini.sh autoevolve 2>&1 | tee -a "$console_log"
            ;;
        *)
            echo "unknown run kind: $kind" >&2
            return 2
            ;;
    esac
}

start_all() {
    check_dependencies
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "tmux session already exists: $SESSION" >&2
        return 1
    fi
    if [[ -e "$RUN_ROOT/.launched" ]]; then
        echo "run root was already launched: $RUN_ROOT" >&2
        return 1
    fi

    prepare_initial_states
    date --iso-8601=seconds > "$RUN_ROOT/.launched"

    tmux new-session -d -s "$SESSION" -n bootstrap
    tmux set-option -t "$SESSION" remain-on-exit on

    local seed
    for seed in "${SEEDS[@]}"; do
        tmux new-window -d -t "$SESSION" -n "t_${seed}" \
            "bash '$REPO_ROOT/repro/aicrowd_whestbench/launch_multi_initial_20260718.sh' _run ttt '$seed' 1"
        tmux new-window -d -t "$SESSION" -n "a_${seed}_1" \
            "bash '$REPO_ROOT/repro/aicrowd_whestbench/launch_multi_initial_20260718.sh' _run auto '$seed' 1"
        tmux new-window -d -t "$SESSION" -n "a_${seed}_2" \
            "bash '$REPO_ROOT/repro/aicrowd_whestbench/launch_multi_initial_20260718.sh' _run auto '$seed' 2"
    done

    tmux kill-window -t "$SESSION:bootstrap"
    status_all
}

validate_all() {
    check_dependencies
    prepare_initial_states
    RUN_WHEST_DRY_RUN=1 run_one ttt german013 1
    RUN_WHEST_DRY_RUN=1 run_one auto german013 1
}

status_all() {
    if ! tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "tmux session not found: $SESSION"
        return 1
    fi
    tmux list-windows -t "$SESSION" -F '#{window_index}\t#{window_name}\t#{pane_dead}\t#{pane_pid}\t#{pane_current_command}'
}

case "${1:-start}" in
    start)
        start_all
        ;;
    validate)
        validate_all
        ;;
    status)
        status_all
        ;;
    _run)
        if [[ $# -ne 4 ]]; then
            echo "internal usage: $0 _run <ttt|auto> <seed> <replica>" >&2
            exit 2
        fi
        run_one "$2" "$3" "$4"
        ;;
    *)
        echo "usage: $0 [start|validate|status]" >&2
        exit 2
        ;;
esac
