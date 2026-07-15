# Recursive AutoEvolve Experiment

This folder describes a first recursive AutoEvolve experiment. It intentionally
contains only the experiment protocol and the task description for the outer
Codex agent. The meta-evaluation launcher, the ARC 50/50 split, and remote run
infrastructure are not implemented here.

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
4. Each inner run executes exactly one AutoEvolve epoch with a hard budget of at
   most 25 evaluator calls.
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
boundary. In particular, reducing a budget, hiding a failure, changing a task,
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
max_evaluator_calls: 25
```

The intended maximum is therefore 25 evaluator calls per task and 75 across a
complete three-task candidate evaluation. The meta-evaluator must enforce the
limit rather than trusting candidate code or post-hoc logs. An inner agent may
use fewer calls because of invalid candidates, failure, or timeout, but it may
not exceed the cap or transfer unused calls between tasks.

Token counts, evaluator calls, elapsed time, and compute usage should still be
recorded for analysis. The phase-one acceptance constraint is the fixed
one-epoch/25-evaluation protocol. A later version may replace this proxy with a
token-and-compute cost ledger closer to Weco's dollar-denominated budget.

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
