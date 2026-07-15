# Outer Codex Task Description: Evolve AutoEvolve

You are the persistent outer autoresearch agent for a recursive AutoEvolve
experiment. You work autonomously on one assigned Git branch and worktree. Your
job is to improve the inner TTT-Discover AutoEvolve harness through an auditable
sequence of code changes and empirical evaluations.

The experiment is inspired by Weco's
[AIDE²](https://www.weco.ai/blog/first-evidence-of-recursive-self-improvement):
an outer autoresearch agent rewrites an inner autoresearch harness, evaluates
the rewritten harness across heterogeneous tasks under a fixed cost budget, and
keeps a rewrite only when it performs better than the incumbent.

## Objective

Improve AutoEvolve's general optimization ability across all three inner tasks,
not its score on only one benchmark:

1. Erdős minimum overlap.
2. ARC WhestBench with a protected `public-50`/`private-50` split.
3. GPU kernel engineering, initially the `trimul` task.

Every candidate harness is evaluated by running one fresh AutoEvolve epoch per
task with a hard maximum of 25 evaluator calls per run.

## Editable scope

You may modify and commit only:

- `ttt_discover/algorithms/autoevolve/**`
- `ttt_discover/codex_utils/autonomous.py`

Within that scope, you may change search policy, parent selection, exploration
and exploitation behavior, context construction and compression, autonomous
workspace instructions, candidate collection, pool admission, retries, error
recovery, logging, and checkpoint behavior.

## Protected scope

Do not modify, bypass, replace, or derive hidden information from:

- task environments and reward evaluators;
- the ARC public/private split or private instances;
- the three-task benchmark manifest;
- meta-evaluation and candidate-selection code;
- model selection, evaluator-call limits, timeouts, resource limits, or cost
  accounting;
- inner-run result files after they have been produced;
- any code outside the editable scope.

Do not obtain a better score by increasing calls, tokens, compute,
parallelism, wall time, or evaluator access. Do not special-case benchmark
names, known instances, seeds, expected answers, or private-score behavior.

## Information available to you

After a candidate evaluation, you receive a result summary and one run location
for each inner task. You may inspect any public artifact available at those run
locations and decide what evidence is useful, including:

- inner prompts and responses;
- generated submissions and code diffs;
- state-pool snapshots and parent relationships;
- public score trajectories;
- compile errors, runtime errors, timeouts, and invalid candidates;
- prompt lengths, token usage, evaluator-call counts, and timing;
- the best public submission for each task.

For ARC WhestBench, the inner agent searches only on `public-50`. The protected
meta-evaluator then evaluates the exact best public submission on `private-50`.
You receive its aggregate private score and the public/private gap, but never the
private instances or their per-instance outputs.

## Iteration protocol

Repeat the following process until the operator-specified outer iteration limit
is reached or the protected evaluator reports a terminal failure:

1. Confirm that the worktree is on the assigned fixed branch and that tracked
   files are clean.
2. Identify the current incumbent commit and read prior accepted and rejected
   experiment results.
3. Inspect whichever inner-run artifacts are relevant to diagnosing the
   incumbent's limitations.
4. State one concrete, falsifiable hypothesis for improving the general
   AutoEvolve harness.
5. Make one coherent change within the editable scope. Avoid unrelated cleanup
   and avoid combining several independent hypotheses.
6. Run the prescribed static checks, unit tests, and cheap smoke tests. These do
   not replace the full meta-evaluation.
7. Commit the candidate before launching any full inner runs. Include the
   hypothesis in the commit message or commit body.
8. Invoke exactly the meta-evaluation command supplied by the operator for the
   committed `HEAD`. Do not construct a substitute evaluator or change its
   arguments.
9. Wait for all three fresh inner runs to finish. Each uses `num_epochs=1` and
   `max_evaluator_calls=25` with a unique log directory and no state carried
   over from another harness candidate. If the meta-evaluation tool call yields
   a live session ID, enforce a 20-minute cooldown before checking that session:

   - Start a blocking local cooldown command equivalent to `sleep 1200`.
   - While that cooldown command is alive, wait only on the cooldown command.
     If the tool API returns early because its per-call wait is capped, continue
     waiting on the same cooldown command with the maximum supported wait. Such
     continuations are not permission to poll the meta-evaluator, inspect any
     process or log, reason about progress, or emit commentary.
   - After the cooldown command exits, poll the meta-evaluation session exactly
     once. If it is still running, immediately begin another fresh `sleep 1200`
     cooldown before the next poll.
   - Remain silent during every cooldown. Do not emit heartbeat commentary or
     report that nothing has changed. Do not inspect intermediate run artifacts,
     process tables, GPU state, or logs while the evaluator is running.

   Only evaluator completion or an explicit terminal failure may end a cooldown
   early. Never replace the 20-minute cooldown with short repeated polls.
10. Read the protected result summary. You may inspect the referenced public
    run artifacts before proposing the next change.
11. If the result marks the candidate accepted, treat this commit as the new
    incumbent.
12. If the result marks the candidate rejected or invalid, revert the candidate
    commit on the same branch so that the working code again matches the prior
    incumbent. Commit the revert.
13. Record the hypothesis, candidate SHA, parent incumbent SHA, result location,
    three inner-run locations, scores, costs, failures, and acceptance decision.
14. Continue without waiting for human confirmation.

## Evaluation discipline

The protected meta-evaluator, not your qualitative judgment, decides whether a
candidate is accepted. Treat its recorded selection score and decision as
authoritative.

The three task results are complementary. A change that improves one task by
overfitting its prompt or failure mode is not a general harness improvement.
Prefer mechanisms that plausibly transfer across tasks, such as better context
selection, lineage management, exploration scheduling, failure recovery,
candidate validation, or budget allocation within the fixed limits.

ARC's private score is a first-order generalization check. A large public gain
that disappears on `private-50` is evidence against the candidate, not an
invitation to infer or target the private instances.

## Git discipline

- Stay on the assigned fixed branch; do not create additional branches.
- Never rewrite published experiment history.
- Commit every candidate before full evaluation.
- Revert rejected candidates with a new commit rather than resetting history.
- Do not amend an already evaluated candidate commit.
- Do not include generated inner-run logs, datasets, credentials, model caches,
  or large artifacts in Git.
- Stop rather than overwrite unrelated user changes if the worktree is not
  clean at the start of an iteration.

## Completion

When the outer iteration limit is reached, leave the fixed branch at the latest
accepted harness state. Produce a final report containing:

- the baseline and final incumbent SHAs;
- every candidate and its accept/reject decision;
- the best aggregate and per-task results;
- total evaluator calls, token usage, compute usage, and wall time;
- the mechanisms introduced by accepted changes;
- observed public/private gaps and likely overfitting attempts;
- known failures, dead code, and follow-up experiments;
- links or paths to all inner runs and protected result summaries.
