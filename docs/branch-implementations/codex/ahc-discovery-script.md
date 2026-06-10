# codex/ahc-discovery-script

## Summary

新增 AHC 场景的 Codex discovery 运行脚本和评测辅助：扩展 AHC env/case runner/prompt，加入 g++ 包装、time wrapper、自循环评估脚本和 repro/run_discovery.py。

## Branch State

- Worktree: `/opt/tiger/discover-ahc-discovery-script`
- HEAD: `abf4052`
- Base used for comparison: `2234282`
- Commits ahead of base: `1`
- Commits behind base: `0`
- Group: `other`
- Implementation location: `committed branch diff`

## Implemented Behavior

- The summary above is based on reading the changed Python code, added config fields, and sampler/prompt hooks in this branch.

## Added Markers

### Config fields

- `initial_pools`
- `uses_gpu`
- `paths`

### Constants

- `AHC039_BASELINE_CODE_VALUE`
- `AHC039_BASELINE_CODE`
- `SCRIPT_DIR`
- `REPO_ROOT`
- `DEFAULT_CACHE_DIR`
- `CACHE_DIR`
- `PROBLEM_ID`
- `DISCOVERY_EXTRA_ARGS`

### Classes

- `TaskSpec`

### Functions

- `_failure_result`
- `check_format`
- `_initial_submission_code`
- `build_autonomous_prompt`
- `ensure_cpp20_compiler`
- `parse_args`
- `main`
- `_elapsed_label`
- `_load_cap_set_env`
- `_load_erdos_env`
- `_load_gpu_mode_env`
- `_load_ahc_env`
- `_load_kakeya_env`
- `_existing_paths`
- `_task_spec`
- `_none_if_empty`
- `_append_paths`
- `_configure_gpu_env`

## Diff Summary

- Worktree tracked shortstat: `(empty)`
- Committed diff stat tail: `10 files changed, 867 insertions(+), 12 deletions(-)`
- Untracked files: `0`

### Worktree Status

````text
(empty)
````
### Committed Branch Files Compared To Base

````text
M	.gitignore
M	examples/ahc/env.py
M	examples/ahc/lib/tool_wrappers/case_runner.py
M	examples/ahc/prompt.py
A	repro/ahc/__init__.py
A	repro/ahc/bin/g++-12
A	repro/ahc/run_0609.sh
A	repro/ahc/self_loop_eval.py
A	repro/ahc/time_wrapper.py
A	repro/run_discovery.py
````
## Diff Stat

### Committed Diff Stat Compared To Base

````text
 .gitignore                                    |   3 +
 examples/ahc/env.py                           | 114 ++++++-
 examples/ahc/lib/tool_wrappers/case_runner.py |  17 +-
 examples/ahc/prompt.py                        |   4 +-
 repro/ahc/__init__.py                         |   1 +
 repro/ahc/bin/g++-12                          |  25 ++
 repro/ahc/run_0609.sh                         |  98 ++++++
 repro/ahc/self_loop_eval.py                   |  80 +++++
 repro/ahc/time_wrapper.py                     |  80 +++++
 repro/run_discovery.py                        | 457 ++++++++++++++++++++++++++
 10 files changed, 867 insertions(+), 12 deletions(-)
````
## Raw Diff

<details>
<summary>Committed patch compared to base</summary>

````diff
diff --git a/.gitignore b/.gitignore
index 8b92741..f4f27b9 100644
--- a/.gitignore
+++ b/.gitignore
@@ -251,3 +251,6 @@ gpu_mode/gpu_scores.tsv
 tinker_log/
 openproblems/
 data/
+
+# AHC local evaluator cache
+examples/ahc/lib/cache/
diff --git a/examples/ahc/env.py b/examples/ahc/env.py
index 2d656e7..9dc4ae5 100644
--- a/examples/ahc/env.py
+++ b/examples/ahc/env.py
@@ -1,3 +1,6 @@
+from pathlib import Path
+import shlex
+
 from examples.ahc.prompt import AHC039_PROMPT, AHC058_PROMPT
 from examples.ahc.lib.eval_task import run_ale_bench_task
 from ttt_discover import Environment, BaseRewardEvaluator, State, DiscoverConfig, discover
@@ -5,24 +8,67 @@ from ttt_discover import Environment, BaseRewardEvaluator, State, DiscoverConfig
 
 CPUS_PER_TASK = 2
 
+AHC039_BASELINE_CODE_VALUE = 1.0
+AHC039_BASELINE_CODE = r"""
+#include <bits/stdc++.h>
+using namespace std;
+
+int main() {
+    ios::sync_with_stdio(false);
+    cin.tie(nullptr);
+
+    int N;
+    cin >> N;
+    for (int i = 0; i < 2 * N; ++i) {
+        int x, y;
+        cin >> x >> y;
+    }
+
+    cout << 4 << '\n';
+    cout << "0 0\n";
+    cout << "100000 0\n";
+    cout << "100000 100000\n";
+    cout << "0 100000\n";
+    return 0;
+}
+""".strip()
+
 
 class AhcRewardEvaluator(BaseRewardEvaluator):
     def __init__(self, *args, **kwargs):
         self.problem_type = kwargs.get("problem_type")
         self.log_dir = kwargs.get("log_dir")
+        self.num_cpus_per_task = max(
+            1,
+            int(kwargs.get("num_cpus_per_task", CPUS_PER_TASK)),
+        )
+        self.eval_timeout = max(1, int(kwargs.get("eval_timeout", 530)))
+        self.lite_version = bool(kwargs.get("lite_version", False))
+
+    def _failure_result(self, raw: dict) -> dict:
+        raw_score = float(raw.get("raw_score", raw.get("performance", raw.get("score", 0.0))) or 0.0)
+        return {
+            "reward": float(raw.get("reward", 0.0) or 0.0),
+            "msg": str(raw.get("msg", "Evaluation failed")),
+            "correctness": float(raw.get("correctness", 0.0) or 0.0),
+            "raw_score": raw_score,
+            "result_construction": [],
+            "stdout": "",
+            "metrics": {"error": str(raw.get("msg", "Evaluation failed"))},
+        }
 
     def get_reward(self, code: str, state: State) -> float:
 
         raw = run_ale_bench_task(
             code,
             problem_id=self.problem_type,
-            lite_version=False,
+            lite_version=self.lite_version,
             log_dir=self.log_dir,
-            num_cpus_per_task=CPUS_PER_TASK,
+            num_cpus_per_task=self.num_cpus_per_task,
         )
         # If lib returned an error dict, pass it through.
         if "case_results" not in raw:
-            return raw
+            return self._failure_result(raw)
         
         case_results = raw["case_results"]
         
@@ -64,8 +110,12 @@ class AhcEnv(Environment):
     @classmethod
     def create_initial_state(cls, problem_type: str) -> State:
         if problem_type == "ahc039":
-            from examples.ahc.prompt import AHC039_BEST_CODE, AHC039_BEST_CODE_VALUE
-            return State(timestep=-1, code=AHC039_BEST_CODE, value=AHC039_BEST_CODE_VALUE, construction=None)
+            return State(
+                timestep=-1,
+                code=AHC039_BASELINE_CODE,
+                value=AHC039_BASELINE_CODE_VALUE,
+                construction=None,
+            )
         if problem_type == "ahc058":
             return State(timestep=-1, code="", value=0.0, construction=None)
         raise ValueError(f"Unknown problem_type: {problem_type}")
@@ -84,7 +134,7 @@ class AhcEnv(Environment):
 
         prompt = AHC039_PROMPT if self.problem_type == "ahc039" else AHC058_PROMPT
 
-        state_ctx = state.to_prompt(target, metric_name="performance", maximize=True)
+        state_ctx = state.to_prompt(target, metric_name="performance", maximize=True, language="cpp")
 
         return f'''{prompt}
 
@@ -104,6 +154,56 @@ Try diverse approaches to solve the problem. The best solution will make efficie
     def _should_keep_code_separators(self) -> bool:
         return False  # ALE Bench doesn't keep separators
 
+    def check_format(self, parsed_code: str) -> bool:
+        code = (parsed_code or "").lower()
+        return bool(code.strip()) and ("int main" in code or "void main" in code)
+
+    def _initial_submission_code(self) -> str:
+        if self.initial_state and self.initial_state.code:
+            return self.initial_state.code
+        return ""
+
+    def build_autonomous_prompt(
+        self,
+        *,
+        prompt: str,
+        workspace: Path,
+        eval_timeout: int,
+        num_cpus_per_task: int,
+    ) -> str:
+        submission_path = workspace / "submission.py"
+        submission_path.write_text(self._initial_submission_code(), encoding="utf-8")
+        evaluator_cmd = (
+            "python -m repro.ahc.self_loop_eval "
+            f"--candidate {shlex.quote(str(submission_path))} "
+            f"--problem-type {shlex.quote(str(self.problem_type))} "
+            f"--log-dir {shlex.quote(str(workspace / 'eval_tmp'))} "
+            f"--eval-timeout {int(eval_timeout)} "
+            f"--num-cpus-per-task {int(num_cpus_per_task)}"
+        )
+
+        return f"""{prompt}
+
+--- Autonomous AHC Search Mode ---
+You may inspect files and run shell commands, but keep all edits inside this workspace:
+{workspace}
+
+Editable candidate:
+{submission_path}
+
+The candidate is C++20 source code even though the outer runner reads it from
+`submission.py`.
+
+Run this evaluator after each revision:
+{evaluator_cmd}
+
+When done, put the best C++20 implementation in:
+{submission_path}
+
+The outer discovery runner will score the final contents of that `submission.py`
+before considering any code block in your final response.
+"""
+
 
 def discover_ahc039():
     # Explicitly define config for clarity
@@ -155,4 +255,4 @@ def discover_ahc058():
 
 if __name__ == "__main__":
     discover_ahc039()
-    # discover_ahc058()
\ No newline at end of file
+    # discover_ahc058()
diff --git a/examples/ahc/lib/tool_wrappers/case_runner.py b/examples/ahc/lib/tool_wrappers/case_runner.py
index 2c3d271..ade82a1 100644
--- a/examples/ahc/lib/tool_wrappers/case_runner.py
+++ b/examples/ahc/lib/tool_wrappers/case_runner.py
@@ -5,7 +5,9 @@ import logging
 import math
 import os
 import re
+import shlex
 import shutil
+import sys
 import tempfile
 import time
 import uuid
@@ -111,11 +113,14 @@ def build_compile_command(
     Returns:
         str: The compile command.
     """
+    work_object_file = f"{constants.WORK_DIR}/{object_file_relative_path}"
+    tmp_object_file = f"/tmp/{object_file_relative_path}"
     compile_command = get_compile_command(code_language, judge_version)
     compile_command += (
-        f"; cp {constants.WORK_DIR}/{object_file_relative_path} /tmp/{object_file_relative_path}"
+        f' && if [ "{work_object_file}" != "{tmp_object_file}" ]; then '
+        f'cp "{work_object_file}" "{tmp_object_file}"; fi'
     )
-    compile_command += f"; chmod 744 /tmp/{object_file_relative_path}"
+    compile_command += f' && chmod 744 "{tmp_object_file}"'
     return compile_command
 
 
@@ -271,8 +276,14 @@ def build_batch_run_command(code_language: CodeLanguage, judge_version: JudgeVer
     """
     run_command = get_run_command(code_language, judge_version)
     run_command += f" < {constants.INPUT_FILE} > {constants.OUTPUT_FILE}"
+    time_command = os.environ.get("ALE_BENCH_TIME_BIN")
+    if time_command is None:
+        if Path("/usr/bin/time").exists():
+            time_command = "/usr/bin/time"
+        else:
+            time_command = f"{shlex.quote(sys.executable)} -m repro.ahc.time_wrapper"
     run_command = (
-        "/usr/bin/time "
+        f"{time_command} "
         f'-f "{constants.TIME_OUTPUT_FORMAT}" '
         f"-o {constants.PROFILES_FILE} {run_command}"
     )  # NOTE: We use the GNU Time to measure the resource usage
diff --git a/examples/ahc/prompt.py b/examples/ahc/prompt.py
index 73fd4f5..b4ad51b 100644
--- a/examples/ahc/prompt.py
+++ b/examples/ahc/prompt.py
@@ -868,9 +868,9 @@ void simulated_annealing_main() {
 
 
     // Output the best polygon
-    std::cout << best_state.poly.size() << "\n";
+    std::cout << best_state.poly.size() << "\\n";
     for (const auto& p : best_state.poly) {
-        std::cout << p.x << " " << p.y << "\n";
+        std::cout << p.x << " " << p.y << "\\n";
     }
 }
 
diff --git a/repro/ahc/__init__.py b/repro/ahc/__init__.py
new file mode 100644
index 0000000..f3bde81
--- /dev/null
+++ b/repro/ahc/__init__.py
@@ -0,0 +1 @@
+"""AHC reproduction helpers."""
diff --git a/repro/ahc/bin/g++-12 b/repro/ahc/bin/g++-12
new file mode 100755
index 0000000..7bdf240
--- /dev/null
+++ b/repro/ahc/bin/g++-12
@@ -0,0 +1,25 @@
+#!/usr/bin/env bash
+set -euo pipefail
+
+stderr_file="$(mktemp)"
+trap 'rm -f "${stderr_file}"' EXIT
+
+if g++ "$@" 2>"${stderr_file}"; then
+    cat "${stderr_file}" >&2
+    exit 0
+fi
+
+status=$?
+if grep -Eq 'cannot find -lgmp(xx)?' "${stderr_file}"; then
+    filtered=()
+    for arg in "$@"; do
+        case "${arg}" in
+            -lgmp|-lgmpxx) ;;
+            *) filtered+=("${arg}") ;;
+        esac
+    done
+    exec g++ "${filtered[@]}"
+fi
+
+cat "${stderr_file}" >&2
+exit "${status}"
diff --git a/repro/ahc/run_0609.sh b/repro/ahc/run_0609.sh
new file mode 100755
index 0000000..a5f2e4b
--- /dev/null
+++ b/repro/ahc/run_0609.sh
@@ -0,0 +1,98 @@
+#!/usr/bin/env bash
+set -euo pipefail
+
+SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
+REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
+cd "${REPO_ROOT}"
+
+export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
+export PYTHONUNBUFFERED=1
+if ! command -v g++-12 >/dev/null 2>&1; then
+    export PATH="${REPO_ROOT}/repro/ahc/bin:${PATH}"
+fi
+
+python - <<'PY'
+import importlib.util
+import sys
+
+required = {
+    "ahocorapy": "ahocorapy",
+    "cairosvg": "CairoSVG",
+    "modal": "modal",
+    "polars": "polars",
+}
+missing = [pkg for module, pkg in required.items() if importlib.util.find_spec(module) is None]
+if missing:
+    print("Missing AHC dependencies: " + ", ".join(missing), file=sys.stderr)
+    print("Install them with: python -m pip install -r requirements/requirements-ahc.txt", file=sys.stderr)
+    raise SystemExit(1)
+PY
+
+DEFAULT_CACHE_DIR="${REPO_ROOT}/examples/ahc/lib/cache"
+CACHE_DIR="${ALE_BENCH_CACHE:-${DEFAULT_CACHE_DIR}}"
+PROBLEM_ID="ahc039"
+DISCOVERY_EXTRA_ARGS=()
+if [[ "${AHC_DRY_RUN:-0}" == "1" ]]; then
+    DISCOVERY_EXTRA_ARGS+=(--dry-run)
+fi
+
+has_ahc_cache() {
+    local input_matches=("${CACHE_DIR}/public_inputs_150/${PROBLEM_ID}"_*.json)
+    [[ -f "${CACHE_DIR}/tester_binaries/${PROBLEM_ID}_tester" ]] && [[ -e "${input_matches[0]:-}" ]]
+}
+
+if ! has_ahc_cache; then
+    if [[ "${CACHE_DIR}" != "${DEFAULT_CACHE_DIR}" ]]; then
+        echo "AHC cache missing under ALE_BENCH_CACHE=${CACHE_DIR}" >&2
+        echo "Populate tester_binaries/ and public_inputs_150/ before running." >&2
+        exit 1
+    fi
+    bash examples/ahc/get_cache.sh
+fi
+
+if ! has_ahc_cache; then
+    echo "AHC cache is still missing after setup: ${CACHE_DIR}" >&2
+    exit 1
+fi
+
+# TTT Discover: non-auto, short Codex samples driven by the sampler/evaluator.
+# AHC is CPU-heavy; these defaults are intentionally smaller than paper-scale runs.
+python repro/run_discovery.py \
+    --task ahc039 \
+    --runner codex_no_finetune \
+    --experiment-name ahc039_0609_ttt_discover_cpu \
+    --log-root codex_runs/ahc_exec_workspaces \
+    --num-epochs 12 \
+    --group-size 4 \
+    --groups-per-batch 1 \
+    --num-cpus-per-task 2 \
+    --eval-timeout 530 \
+    --wandb-project "" \
+    --codex-backend cli \
+    --codex-model-name gpt-5.5 \
+    --codex-cli-command codex \
+    --codex-cli-sandbox read-only \
+    --codex-cli-timeout 900 \
+    --codex-max-concurrent-requests 2 \
+    "${DISCOVERY_EXTRA_ARGS[@]}"
+
+# Codex AutoEvolve: one writable workspace per sample for deeper local search.
+python repro/run_discovery.py \
+    --task ahc039 \
+    --runner codex_no_finetune \
+    --experiment-name ahc039_0609_codex_autoevolve_cpu \
+    --log-root codex_runs/ahc_exec_workspaces \
+    --num-epochs 4 \
+    --group-size 1 \
+    --groups-per-batch 1 \
+    --num-cpus-per-task 2 \
+    --eval-timeout 530 \
+    --wandb-project "" \
+    --codex-backend cli \
+    --codex-model-name gpt-5.5 \
+    --codex-cli-command codex \
+    --codex-cli-sandbox workspace-write \
+    --codex-cli-timeout 7200 \
+    --codex-max-concurrent-requests 1 \
+    --codex-autonomous \
+    "${DISCOVERY_EXTRA_ARGS[@]}"
diff --git a/repro/ahc/self_loop_eval.py b/repro/ahc/self_loop_eval.py
new file mode 100644
index 0000000..f0eeaa3
--- /dev/null
+++ b/repro/ahc/self_loop_eval.py
@@ -0,0 +1,80 @@
+from __future__ import annotations
+
+import argparse
+import json
+import os
+import shutil
+from pathlib import Path
+
+
+def ensure_cpp20_compiler() -> None:
+    repo_root = Path(__file__).resolve().parents[2]
+    pythonpath = os.environ.get("PYTHONPATH", "")
+    if str(repo_root) not in pythonpath.split(os.pathsep):
+        os.environ["PYTHONPATH"] = (
+            f"{repo_root}{os.pathsep}{pythonpath}" if pythonpath else str(repo_root)
+        )
+    if shutil.which("g++-12") is not None:
+        return
+    wrapper_dir = repo_root / "repro" / "ahc" / "bin"
+    os.environ["PATH"] = f"{wrapper_dir}:{os.environ.get('PATH', '')}"
+
+
+ensure_cpp20_compiler()
+
+from examples.ahc.env import AhcRewardEvaluator  # noqa: E402
+from ttt_discover import State  # noqa: E402
+
+
+def parse_args() -> argparse.Namespace:
+    parser = argparse.ArgumentParser(
+        description="Evaluate one AHC C++ candidate with the local ALE-Bench cache."
+    )
+    parser.add_argument("--candidate", required=True, help="Path to the C++ source file.")
+    parser.add_argument(
+        "--problem-type",
+        choices=("ahc039", "ahc058"),
+        default="ahc039",
+    )
+    parser.add_argument("--log-dir", required=True)
+    parser.add_argument("--eval-timeout", type=int, default=530)
+    parser.add_argument("--num-cpus-per-task", type=int, default=2)
+    parser.add_argument(
+        "--lite-version",
+        action="store_true",
+        help="Evaluate on the lite public cache when available.",
+    )
+    return parser.parse_args()
+
+
+def main() -> None:
+    args = parse_args()
+    candidate_path = Path(args.candidate)
+    code = candidate_path.read_text(encoding="utf-8")
+
+    evaluator = AhcRewardEvaluator(
+        problem_type=args.problem_type,
+        log_dir=args.log_dir,
+        eval_timeout=args.eval_timeout,
+        num_cpus_per_task=args.num_cpus_per_task,
+        lite_version=args.lite_version,
+    )
+    result = evaluator.get_reward(
+        code,
+        state=State(timestep=-1, construction=None, code="", value=0.0),
+    )
+    summary = {
+        "problem_type": args.problem_type,
+        "candidate": str(candidate_path),
+        "reward": result.get("reward"),
+        "correctness": result.get("correctness"),
+        "raw_score": result.get("raw_score"),
+        "msg": result.get("msg"),
+    }
+    print(json.dumps(summary, indent=2, sort_keys=True))
+    if float(result.get("correctness", 0.0) or 0.0) <= 0.0:
+        raise SystemExit(1)
+
+
+if __name__ == "__main__":
+    main()
diff --git a/repro/ahc/time_wrapper.py b/repro/ahc/time_wrapper.py
new file mode 100644
index 0000000..7a69f1d
--- /dev/null
+++ b/repro/ahc/time_wrapper.py
@@ -0,0 +1,80 @@
+from __future__ import annotations
+
+import argparse
+import json
+import os
+import resource
+import subprocess
+import sys
+import time
+
+
+def _elapsed_label(seconds: float) -> str:
+    minutes, secs = divmod(max(0.0, seconds), 60.0)
+    hours, minutes = divmod(int(minutes), 60)
+    if hours:
+        return f"{hours}:{minutes:02d}:{secs:05.2f}"
+    return f"{minutes}:{secs:05.2f}"
+
+
+def parse_args() -> argparse.Namespace:
+    parser = argparse.ArgumentParser(add_help=False)
+    parser.add_argument("-f", "--format", default="")
+    parser.add_argument("-o", "--output", required=True)
+    parser.add_argument("command", nargs=argparse.REMAINDER)
+    return parser.parse_args()
+
+
+def main() -> None:
+    args = parse_args()
+    if not args.command:
+        raise SystemExit("missing command")
+
+    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
+    start = time.perf_counter()
+    completed = subprocess.run(args.command)
+    elapsed = time.perf_counter() - start
+    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
+
+    user_cpu = max(0.0, usage_after.ru_utime - usage_before.ru_utime)
+    system_cpu = max(0.0, usage_after.ru_stime - usage_before.ru_stime)
+    cpu_percent = "0%"
+    if elapsed > 0:
+        cpu_percent = f"{int(round(((user_cpu + system_cpu) / elapsed) * 100))}%"
+
+    profile = {
+        "command": " ".join(args.command),
+        "exit_status": int(completed.returncode),
+        "elapsed_time": _elapsed_label(elapsed),
+        "elapsed_time_seconds": elapsed,
+        "system_cpu_seconds": system_cpu,
+        "user_cpu_seconds": user_cpu,
+        "cpu_percent": cpu_percent,
+        "max_resident_set_size_kbytes": int(usage_after.ru_maxrss),
+        "average_resident_set_size_kbytes": 0,
+        "average_total_memory_kbytes": 0,
+        "average_unshared_data_kbytes": 0,
+        "average_unshared_stack_kbytes": 0,
+        "average_shared_text_kbytes": 0,
+        "page_size_bytes": int(os.sysconf("SC_PAGE_SIZE")),
+        "major_page_faults": int(usage_after.ru_majflt - usage_before.ru_majflt),
+        "minor_page_faults": int(usage_after.ru_minflt - usage_before.ru_minflt),
+        "swaps": int(usage_after.ru_nswap - usage_before.ru_nswap),
+        "involuntary_context_switches": int(usage_after.ru_nivcsw - usage_before.ru_nivcsw),
+        "voluntary_context_switches": int(usage_after.ru_nvcsw - usage_before.ru_nvcsw),
+        "file_system_inputs": int(usage_after.ru_inblock - usage_before.ru_inblock),
+        "file_system_outputs": int(usage_after.ru_oublock - usage_before.ru_oublock),
+        "socket_messages_received": int(usage_after.ru_msgrcv - usage_before.ru_msgrcv),
+        "socket_messages_sent": int(usage_after.ru_msgsnd - usage_before.ru_msgsnd),
+        "signals_delivered": int(usage_after.ru_nsignals - usage_before.ru_nsignals),
+    }
+
+    with open(args.output, "w", encoding="utf-8") as handle:
+        json.dump(profile, handle)
+        handle.write("\n")
+
+    raise SystemExit(completed.returncode)
+
+
+if __name__ == "__main__":
+    main()
diff --git a/repro/run_discovery.py b/repro/run_discovery.py
new file mode 100644
index 0000000..92a3e33
--- /dev/null
+++ b/repro/run_discovery.py
@@ -0,0 +1,457 @@
+from __future__ import annotations
+
+import argparse
+import asyncio
+import os
+import sys
+from dataclasses import dataclass
+from pathlib import Path
+from typing import Callable
+
+
+REPO_ROOT = Path(__file__).resolve().parents[1]
+if str(REPO_ROOT) not in sys.path:
+    sys.path.insert(0, str(REPO_ROOT))
+
+
+EnvLoader = Callable[[], type]
+
+
+@dataclass(frozen=True)
+class TaskSpec:
+    env_loader: EnvLoader
+    problem_type: str
+    experiment_name: str
+    wandb_project: str | None
+    eval_timeout: int
+    num_epochs: int
+    group_size: int
+    groups_per_batch: int
+    initial_pools: tuple[Path, ...] = ()
+    uses_gpu: bool = False
+
+
+def _load_cap_set_env() -> type:
+    from examples.cap_set_priority.env import CapSetPriorityEnv
+
+    return CapSetPriorityEnv
+
+
+def _load_erdos_env() -> type:
+    from examples.erdos_min_overlap.env import ErdosMinOverlapEnv
+
+    return ErdosMinOverlapEnv
+
+
+def _load_gpu_mode_env() -> type:
+    from examples.gpu_mode.env import GpuModeEnv
+
+    return GpuModeEnv
+
+
+def _load_ahc_env() -> type:
+    from examples.ahc.env import AhcEnv
+
+    return AhcEnv
+
+
+def _load_kakeya_env() -> type:
+    from examples.kakeya.env import KakeyaEnv
+
+    return KakeyaEnv
+
+
+def _existing_paths(paths: tuple[Path, ...]) -> tuple[Path, ...]:
+    return tuple(path for path in paths if path.exists())
+
+
+def _task_spec(args: argparse.Namespace) -> TaskSpec:
+    task = args.task
+    if task == "trimul":
+        task = "gpu_mode:trimul"
+    elif task == "mla_decode_nvidia":
+        task = "gpu_mode:mla_decode_nvidia"
+
+    if task == "cap_set":
+        dimension = args.dimension if args.dimension is not None else 8
+        return TaskSpec(
+            env_loader=_load_cap_set_env,
+            problem_type=str(dimension),
+            experiment_name=f"cap-set-priority-{dimension}",
+            wandb_project="cap-set-priority",
+            eval_timeout=45,
+            num_epochs=10,
+            group_size=1,
+            groups_per_batch=1,
+            initial_pools=_existing_paths(
+                (REPO_ROOT / "repro/cap_set/initial_pool_400_to_512.json",)
+            ),
+        )
+
+    if task == "erdos":
+        return TaskSpec(
+            env_loader=_load_erdos_env,
+            problem_type="",
+            experiment_name="erdos-min-overlap",
+            wandb_project="erdos-min-overlap",
+            eval_timeout=1100,
+            num_epochs=50,
+            group_size=64,
+            groups_per_batch=8,
+            initial_pools=_existing_paths(
+                (
+                    REPO_ROOT
+                    / "repro/erdos/initial_pool_reference_plus_codex_20260603.json",
+                )
+            ),
+        )
+
+    if task == "kakeya":
+        dimension = args.dimension if args.dimension is not None else 3
+        if dimension != 3:
+            raise ValueError("The Kakeya environment currently supports only d=3")
+        primes = args.primes or "5,7,13"
+        return TaskSpec(
+            env_loader=_load_kakeya_env,
+            problem_type=f"{dimension}:{primes}",
+            experiment_name="kakeya",
+            wandb_project="kakeya",
+            eval_timeout=120,
+            num_epochs=10,
+            group_size=1,
+            groups_per_batch=1,
+        )
+
+    if task in {"ahc039", "ahc058"}:
+        return TaskSpec(
+            env_loader=_load_ahc_env,
+            problem_type=task,
+            experiment_name=f"{task}-codex",
+            wandb_project=None,
+            eval_timeout=530,
+            num_epochs=10,
+            group_size=1,
+            groups_per_batch=1,
+        )
+
+    if task == "gpu_mode:trimul":
+        return TaskSpec(
+            env_loader=_load_gpu_mode_env,
+            problem_type="trimul",
+            experiment_name="gpu-mode-trimul",
+            wandb_project=None,
+            eval_timeout=1200,
+            num_epochs=50,
+            group_size=1,
+            groups_per_batch=1,
+            uses_gpu=True,
+        )
+
+    if task == "gpu_mode:mla_decode_nvidia":
+        return TaskSpec(
+            env_loader=_load_gpu_mode_env,
+            problem_type="mla_decode_nvidia",
+            experiment_name="gpu-mode-mla-decode-nvidia",
+            wandb_project=None,
+            eval_timeout=1200,
+            num_epochs=50,
+            group_size=1,
+            groups_per_batch=1,
+            uses_gpu=True,
+        )
+
+    raise ValueError(f"Unknown task: {args.task}")
+
+
+def _none_if_empty(value: str | None) -> str | None:
+    if value is None or value == "":
+        return None
+    return value
+
+
+def _append_paths(
+    defaults: tuple[Path, ...],
+    extras: list[str] | None,
+    *,
+    use_defaults: bool,
+) -> tuple[str, ...]:
+    paths: list[str] = []
+    if use_defaults:
+        paths.extend(str(path) for path in defaults)
+    paths.extend(extras or [])
+    return tuple(paths)
+
+
+def parse_args() -> argparse.Namespace:
+    parser = argparse.ArgumentParser(
+        description=(
+            "Launch a TTT-Discover task through a small task registry. Existing "
+            "task-specific repro runners remain supported; this is a shared "
+            "entrypoint for new or routine runs."
+        )
+    )
+    parser.add_argument(
+        "--task",
+        required=True,
+        choices=(
+            "cap_set",
+            "erdos",
+            "kakeya",
+            "ahc039",
+            "ahc058",
+            "gpu_mode:trimul",
+            "gpu_mode:mla_decode_nvidia",
+            "trimul",
+            "mla_decode_nvidia",
+        ),
+    )
+    parser.add_argument(
+        "--problem-type",
+        default=None,
+        help="Override the registry problem_type passed to the environment.",
+    )
+    parser.add_argument("--dimension", type=int, default=None)
+    parser.add_argument(
+        "--primes",
+        default=None,
+        help="Kakeya-only comma-separated prime list, e.g. '5,7,13'.",
+    )
+    parser.add_argument("--experiment-name", default=None)
+    parser.add_argument("--log-root", default="tinker_log")
+    parser.add_argument(
+        "--wandb-project",
+        default=None,
+        help="Override WANDB project. Pass '' to disable.",
+    )
+    parser.add_argument(
+        "--no-default-initial-pool",
+        action="store_true",
+        help="Do not load default initial-pool files from the task registry.",
+    )
+    parser.add_argument(
+        "--codex-initial-pool",
+        action="append",
+        default=None,
+        help="Reusable Codex initial-pool state JSON. May be repeated.",
+    )
+    parser.add_argument(
+        "--codex-initial-program",
+        action="append",
+        default=None,
+        help="Seed program to verify and add to the Codex sampler pool.",
+    )
+
+    parser.add_argument(
+        "--runner",
+        choices=("tinker_rl", "codex_no_finetune"),
+        default="codex_no_finetune",
+    )
+    parser.add_argument(
+        "--model-name",
+        choices=("openai/gpt-oss-120b", "openai/gpt-oss-20b"),
+        default="openai/gpt-oss-120b",
+    )
+    parser.add_argument("--num-epochs", type=int, default=None)
+    parser.add_argument("--group-size", type=int, default=None)
+    parser.add_argument("--groups-per-batch", type=int, default=None)
+    parser.add_argument("--learning-rate", type=float, default=4e-5)
+    parser.add_argument("--temperature", type=float, default=1.0)
+    parser.add_argument("--kl-penalty-coef", type=float, default=0.1)
+    parser.add_argument("--lora-rank", type=int, default=32)
+    parser.add_argument("--save-every", type=int, default=2)
+    parser.add_argument("--phase1-max-tokens", type=int, default=26000)
+    parser.add_argument("--num-cpus-per-task", type=int, default=1)
+    parser.add_argument("--eval-timeout", type=int, default=None)
+    parser.add_argument(
+        "--remove-constant-reward-groups",
+        action=argparse.BooleanOptionalAction,
+        default=None,
+    )
+
+    parser.add_argument("--codex-backend", choices=("cli", "responses"), default="cli")
+    parser.add_argument("--codex-model-name", default=None)
+    parser.add_argument("--codex-max-output-tokens", type=int, default=None)
+    parser.add_argument("--codex-temperature", type=float, default=None)
+    parser.add_argument("--codex-api-key-env", default="OPENAI_API_KEY")
+    parser.add_argument("--codex-base-url", default=None)
+    parser.add_argument("--codex-cli-command", default="codex")
+    parser.add_argument(
+        "--codex-cli-sandbox",
+        choices=("read-only", "workspace-write", "danger-full-access"),
+        default=None,
+        help="Defaults to read-only unless --codex-autonomous is set.",
+    )
+    parser.add_argument("--codex-cli-timeout", type=float, default=None)
+    parser.add_argument("--codex-max-concurrent-requests", type=int, default=4)
+    parser.add_argument(
+        "--codex-autonomous",
+        action="store_true",
+        help="AutoEvolve mode: give Codex a writable workspace for deep dives.",
+    )
+
+    parser.add_argument(
+        "--gpu",
+        default=None,
+        help="Set CUDA_VISIBLE_DEVICES for GPU tasks.",
+    )
+    parser.add_argument("--cuda-device-order", default="PCI_BUS_ID")
+    parser.add_argument(
+        "--torch-cuda-arch-list",
+        default=None,
+        help="Optional TORCH_CUDA_ARCH_LIST override, e.g. 8.0 for A800.",
+    )
+    parser.add_argument(
+        "--dry-run",
+        action="store_true",
+        help="Print the resolved configuration without launching discovery.",
+    )
+    return parser.parse_args()
+
+
+def _configure_gpu_env(args: argparse.Namespace, spec: TaskSpec) -> None:
+    if not spec.uses_gpu:
+        return
+    if args.gpu is not None:
+        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
+    os.environ.setdefault("CUDA_DEVICE_ORDER", args.cuda_device_order)
+    if args.torch_cuda_arch_list:
+        os.environ["TORCH_CUDA_ARCH_LIST"] = args.torch_cuda_arch_list
+    os.environ.setdefault("PYTHONUNBUFFERED", "1")
+
+
+def main() -> None:
+    args = parse_args()
+    spec = _task_spec(args)
+    _configure_gpu_env(args, spec)
+
+    env_type = spec.env_loader()
+    experiment_name = args.experiment_name or spec.experiment_name
+    problem_type = args.problem_type if args.problem_type is not None else spec.problem_type
+    wandb_project = (
+        _none_if_empty(args.wandb_project)
+        if args.wandb_project is not None
+        else spec.wandb_project
+    )
+    num_epochs = args.num_epochs if args.num_epochs is not None else spec.num_epochs
+    group_size = args.group_size if args.group_size is not None else spec.group_size
+    groups_per_batch = (
+        args.groups_per_batch
+        if args.groups_per_batch is not None
+        else spec.groups_per_batch
+    )
+    eval_timeout = args.eval_timeout if args.eval_timeout is not None else spec.eval_timeout
+    discovery_cpus = args.num_cpus_per_task
+    if args.runner == "codex_no_finetune":
+        discovery_cpus = 0
+
+    codex_cli_sandbox = args.codex_cli_sandbox
+    if codex_cli_sandbox is None:
+        codex_cli_sandbox = "workspace-write" if args.codex_autonomous else "read-only"
+
+    remove_constant_reward_groups = args.remove_constant_reward_groups
+    if remove_constant_reward_groups is None:
+        remove_constant_reward_groups = group_size > 1
+
+    initial_pools = _append_paths(
+        spec.initial_pools,
+        args.codex_initial_pool,
+        use_defaults=not args.no_default_initial_pool,
+    )
+    if args.dry_run:
+        print(f"task={args.task}")
+        print(f"env_type={env_type.__module__}.{env_type.__name__}")
+        print(f"problem_type={problem_type!r}")
+        print(f"experiment_name={experiment_name!r}")
+        print(f"log_root={args.log_root!r}")
+        print(f"log_path={str(Path(args.log_root) / experiment_name)!r}")
+        print(f"runner={args.runner!r}")
+        print(f"num_epochs={num_epochs}")
+        print(f"group_size={group_size}")
+        print(f"groups_per_batch={groups_per_batch}")
+        print(f"eval_timeout={eval_timeout}")
+        print(f"wandb_project={wandb_project!r}")
+        print(f"codex_autonomous={args.codex_autonomous}")
+        print(f"codex_cli_sandbox={codex_cli_sandbox!r}")
+        print(f"codex_initial_pool_paths={initial_pools!r}")
+        if spec.uses_gpu:
+            print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
+            print(f"CUDA_DEVICE_ORDER={os.environ.get('CUDA_DEVICE_ORDER')!r}")
+            print(f"TORCH_CUDA_ARCH_LIST={os.environ.get('TORCH_CUDA_ARCH_LIST')!r}")
+        return
+
+    if args.runner == "codex_no_finetune":
+        from ttt_discover.rl.codex_no_finetune import (
+            CodexNoFinetuneConfig,
+            main as codex_no_finetune_main,
+        )
+
+        cfg = CodexNoFinetuneConfig(
+            env_type=env_type,
+            problem_type=problem_type,
+            backend=args.codex_backend,
+            model_name=args.codex_model_name,
+            groups_per_batch=groups_per_batch,
+            group_size=group_size,
+            num_cpus_per_task=max(1, int(args.num_cpus_per_task)),
+            eval_timeout=eval_timeout,
+            num_epochs=num_epochs,
+            max_output_tokens=args.codex_max_output_tokens,
+            temperature=args.codex_temperature,
+            api_key_env=args.codex_api_key_env,
+            base_url=args.codex_base_url,
+            cli_command=args.codex_cli_command,
+            cli_sandbox=codex_cli_sandbox,
+            cli_timeout=args.codex_cli_timeout,
+            max_concurrent_requests=args.codex_max_concurrent_requests,
+            initial_program_paths=tuple(args.codex_initial_program or ()),
+            initial_pool_paths=initial_pools,
+            autonomous=args.codex_autonomous,
+            wandb_project=wandb_project,
+            wandb_name=experiment_name,
+            log_path=str(Path(args.log_root) / experiment_name),
+            remove_constant_reward_groups=remove_constant_reward_groups,
+        )
+        asyncio.run(codex_no_finetune_main(cfg))
+        return
+
+    from ttt_discover import DiscoverConfig, discover
+
+    config = DiscoverConfig(
+        env_type=env_type,
+        problem_type=problem_type,
+        model_name=args.model_name,
+        runner=args.runner,
+        lora_rank=args.lora_rank,
+        group_size=group_size,
+        groups_per_batch=groups_per_batch,
+        learning_rate=args.learning_rate,
+        num_epochs=num_epochs,
+        temperature=args.temperature,
+        kl_penalty_coef=args.kl_penalty_coef,
+        phase1_max_tokens=args.phase1_max_tokens,
+        save_every=args.save_every,
+        num_cpus_per_task=discovery_cpus,
+        eval_timeout=eval_timeout,
+        experiment_name=experiment_name,
+        log_root=args.log_root,
+        wandb_project=wandb_project,
+        codex_model_name=args.codex_model_name,
+        codex_backend=args.codex_backend,
+        codex_max_output_tokens=args.codex_max_output_tokens,
+        codex_temperature=args.codex_temperature,
+        codex_api_key_env=args.codex_api_key_env,
+        codex_base_url=args.codex_base_url,
+        codex_cli_command=args.codex_cli_command,
+        codex_cli_sandbox=codex_cli_sandbox,
+        codex_cli_timeout=args.codex_cli_timeout,
+        codex_max_concurrent_requests=args.codex_max_concurrent_requests,
+        codex_initial_program_paths=tuple(args.codex_initial_program or ()),
+        codex_initial_pool_paths=initial_pools,
+        codex_autonomous=args.codex_autonomous,
+        remove_constant_reward_groups=remove_constant_reward_groups,
+    )
+    discover(config)
+
+
+if __name__ == "__main__":
+    main()
````
</details>

