# Recursive AutoEvolve Experiment

This folder contains the protocol, the task description for the outer Codex
agent, and the protected local meta-evaluation launcher. The launcher runs each
committed harness in a detached Git worktree, starts three fresh inner
AutoEvolve tasks, enforces a one-hour wall-time limit for each inner task,
and stores its incumbent ledger outside the repository.

The design is inspired by Weco's
[AIDE² recursive self-improvement experiment](https://www.weco.ai/blog/first-evidence-of-recursive-self-improvement).
In that experiment, an outer autoresearch agent rewrites an inner autoresearch
harness. Each proposed harness is evaluated by running it on a heterogeneous
task suite under a fixed cost budget, and it survives only when it improves on
the incumbent. Weco also separates the score visible to an inner task-solving
agent from a held-out private score used to judge whether the harness actually
generalizes.

Our initial experiment follows the same high-level pattern with Codex and the
TTT-Discover AutoEvolve implementation:

1. A persistent outer Codex task owns one fixed Git branch and worktree.
2. The outer agent edits the AutoEvolve harness and commits one coherent
   candidate change.
3. A protected meta-evaluator checks out that committed candidate and launches
   three fresh inner AutoEvolve runs.
4. Each inner run executes exactly one AutoEvolve epoch with no evaluator-call
   cap and a hard one-hour wall-time limit.
5. The meta-evaluator reports task results, costs, and inner-run locations.
6. The candidate is retained only when it beats the incumbent under the
   protected selection rule. A rejected candidate is reverted on the fixed
   branch.

## Evolvable and protected code

The outer agent may modify:

- `ttt_discover/algorithms/autoevolve/**`
- `ttt_discover/codex_utils/autonomous.py`

This includes the state pool and parent-selection logic, prompt and context
construction, workspace layout, candidate collection, retry behavior, and
checkpoint handling.

The outer agent must not modify task environments, reward evaluators,
public/private data, benchmark manifests, model settings, cost accounting,
meta-evaluation code, or the selection rule. These form the trusted evaluation
boundary. In particular, extending wall time, hiding a failure, changing a task,
or increasing parallelism is not a harness improvement.

## Inner benchmark suite

Every outer candidate is evaluated on all three tasks from a fresh initial
state and a fresh log directory.

| Task | Initial environment | Role in the suite | Phase-one evaluation |
| --- | --- | --- | --- |
| Erdős minimum overlap | `examples.erdos_min_overlap.env:ErdosMinOverlapEnv` | Mathematical/scientific program search | Existing task score |
| ARC WhestBench | `examples.aicrowd_whestbench.env:WhestBenchEnv` | Numerical/ML estimator search | Public 50 for inner search; private 50 for outer evaluation |
| GPU kernel | `examples.gpu_mode.env:GpuModeEnv` | GPU systems and kernel engineering | Existing kernel evaluator, initially `trimul` |

This is the training/meta-selection suite. A separate held-out suite is
deliberately deferred until the recursive loop and accounting are stable.

### ARC WhestBench split

The protected benchmark manifest defines an ordered set of 100 WhestBench
instances. The split is fixed before the experiment begins:

- `public-50`: instances 0 through 49. The inner AutoEvolve run may evaluate
  candidates on these instances and use their scores as search feedback.
- `private-50`: instances 50 through 99. The inner agent cannot inspect these
  instances or query this evaluator.

After the public inner run finishes, the meta-evaluator takes its best
submission and evaluates that exact artifact once on `private-50`. The outer
agent receives the aggregate private result and may compare it with the public
result, but it does not receive private instances or per-instance outputs.

## Cost protocol

For every outer candidate, the three inner runs are configured independently
with:

```yaml
algorithm: autoevolve
num_epochs: 1
max_evaluator_calls: null
cli_timeout: 3000
```

The blackbox evaluator does not impose a call-count cap. Each task instead has
a hard 3600-second process limit. The autonomous Codex search receives up to
3000 seconds, leaving the remainder for candidate collection, final scoring,
and cleanup. The three task clocks are independent and their unused time cannot
be transferred.

Token counts, evaluator calls, elapsed time, and compute usage should still be
recorded for analysis. The acceptance constraint is the fixed one-epoch,
one-hour-per-task protocol.

## What the outer agent receives

The meta-evaluator should return a small machine-readable summary plus a run
reference for each inner task. A suggested shape is:

```json
{
  "candidate_sha": "...",
  "incumbent_sha": "...",
  "accepted": false,
  "selection_score": 0.0,
  "tasks": {
    "erdos": {
      "run_ref": "...",
      "best_score": null,
      "evaluator_calls": 0
    },
    "arc_whestbench": {
      "run_ref": "...",
      "public_best_score": null,
      "private_best_score": null,
      "evaluator_calls": 0
    },
    "kernel": {
      "run_ref": "...",
      "best_score": null,
      "evaluator_calls": 0
    }
  }
}
```

The outer agent may decide which public artifacts to inspect at those run
locations: prompts, responses, candidate code, pool snapshots, score
trajectories, failures, token usage, or timing. The task description does not
force a fixed trajectory summary. Private ARC inputs and per-instance private
outputs remain inaccessible; only their aggregate result is returned.

## Candidate lifecycle on the fixed branch

The branch is an auditable experiment journal:

```text
incumbent
  |
candidate commit
  |
  +-- accepted: continue from the candidate
  |
  +-- rejected: commit a revert, then continue from restored incumbent code
```

Each candidate commit should state its hypothesis. The associated result must
record the candidate SHA, parent incumbent SHA, exact evaluation configuration,
three run references, resource usage, scores, and acceptance decision. Inner
runs must never resume another harness candidate's state pool.

## Files in this folder

- `TASK_DESCRIPTION.md`: the instruction to give the persistent outer Codex
  task.
- `README.md`: the human-facing experiment design and trusted protocol.
- `meta_eval.py`: the protected launcher, private WhestBench scorer, selection
  rule, and external incumbent ledger.

## Protected launcher

The exact command supplied to the outer agent for each committed candidate is:

```bash
python experiments/recursive_autoevolve/meta_eval.py --candidate-sha HEAD
```

Results and the incumbent ledger are written under
`/opt/tiger/recursive_autoevolve_runs` by default. This location is outside the
Git worktree, so generated artifacts cannot be committed as harness changes.
The launcher rejects a candidate unless its tree diff from the incumbent is
limited to the editable paths in `TASK_DESCRIPTION.md`.

An operator may establish the initial baseline before launching the outer loop:

```bash
python experiments/recursive_autoevolve/meta_eval.py \
  --candidate-sha HEAD \
  --bootstrap
```

If the ledger does not exist when the first candidate is evaluated, the
launcher automatically evaluates the candidate's parent commit as the initial
incumbent before evaluating the candidate.

### Fixed phase-one manifest

- Every inner run uses AutoEvolve, one epoch, one autonomous Codex call,
  `gpt-5.5` with the harness's fixed high reasoning effort, no evaluator-call
  cap, and a hard one-hour task wall-time limit.
- Erdős starts from one deterministic protected construction.
- This branch's lightweight NumPy WhestBench environment uses seeds 0–49 as
  `public-50` and seeds 50–99 as `private-50`. Monte Carlo targets are cached
  inside the trusted evaluator. Only the aggregate private score is recorded.
- The kernel task is `trimul`. Each run exclusively reserves physical GPU 7
  and exposes only that device through `CUDA_VISIBLE_DEVICES`.
- The three heterogeneous tasks run concurrently, but each task keeps
  `group_size=1`, `groups_per_batch=1`, and
  `max_concurrent_requests=1`.

All three task scores are treated as losses. A candidate is accepted only if:

1. every task and the private WhestBench evaluation is valid;
2. the geometric mean of incumbent/candidate task-score ratios is at least
   `1.01`;
3. no individual task regresses by more than 2%; and
4. at least two of the three task scores improve.

The candidate result JSON contains the public/private gap, run paths,
evaluator calls, token usage when reported by Codex, task wall time, GPU
allocation, and the authoritative accept/reject decision.

Validate command plumbing without launching any inner agents:

```bash
python experiments/recursive_autoevolve/meta_eval.py \
  --candidate-sha HEAD \
  --validate-only
```
