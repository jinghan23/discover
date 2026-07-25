# WhestBench competitive learned covariance closure

This directory trains an offline estimator for the Phase-1 leaderboard target
`(width=256, depth=32)`.  It is not a signal-only pilot: model selection is
gated against the strongest checked-in analytic and sampling results.

## Estimator

For fixed weights `W`, the supervised target is

```text
y_l(W) = E_x[ReLU(... ReLU(x @ W_0) ... @ W_l)],  x ~ N(0, I).
```

The deterministic base is a dense Gaussian covariance closure:

- the first post-ReLU covariance uses the exact centred Gaussian arc-cosine
  kernel;
- later pairwise ReLU moments use a non-central bivariate Gaussian closure;
- marginal means and variances are propagated exactly under that closure.

The learned model predicts the higher-order non-Gaussian residual.  Its state
is permutation equivariant and receives `W.T @ state`, `(W**2).T @ state`, a
covariance-aware relational message, and invariant analytic features.  A
global head learns stable layer-wise calibration while a local head learns
per-neuron corrections.  The exact first-layer mean is never changed.

The scored final row starts from the reproducible `0.9921` global calibration
of the covariance closure.  Step 0 is evaluated and saved before optimization,
so noisy training can never replace the deterministic baseline with a worse
`best` artifact.  Checkpoint selection and export use an EMA of the weights.

The residual range is `0.5 * pre_std`, chosen from an empirical clipping
diagnostic.  The previous `0.1 * layer_fraction * pre_std` range imposed a raw
final-MSE floor around `4.7e-4`, which could not beat the existing analytic
baselines even with an oracle residual.

## Training labels

Labels are generated online from fresh independent He-initialized MLPs.  The
teacher combines:

- antithetic directions;
- independently randomized rank-1 lattice points mapped through the normal
  inverse CDF;
- exact radial Rao--Blackwellization using `E[chi_width]`;
- the exact first-layer Gaussian mean.

These operations preserve unbiasedness while reducing label variance.  The
fixed synthetic validation target averages two independent estimates and
reports their inferred noise floor.

## Competitive gates

Every validation record includes:

- learned, closure-baseline, and auxiliary MSE;
- optimal global scale and affine diagnostic baselines;
- the oracle MSE allowed by the configured residual bound;
- bound saturation and teacher-noise estimates;
- `qualification_gate_pass`: raw final MSE at most `3.5e-5`, enough to compete
  with the best checked-in analytic adjusted score while remaining below the
  10% compute-multiplier floor;
- `leaderboard_gate_pass`: raw final MSE at most `3.5e-6`, the target band of
  the checked-in 10%-compute sampling estimator.

Checkpoint selection uses only newly generated synthetic networks.  Production
also records local full-100 at step 0, every 25,000 steps, and the final step,
but these reports are monitoring-only and never replace `checkpoint_best.pt`.
Do not use the monitoring curve for hyperparameter or checkpoint selection.

## Configurations

| Config | Hardware | Role |
|---|---|---|
| `configs/smoke.json` | CPU/GPU | contract and serialization wiring only |
| `configs/ablation_1xh100_80gb.json` | 1x H100/H20 80GB+ | full-shape architecture/scale ablation |
| `configs/production_4xh100_80gb.json` | 4x H100/H20 80GB+ | 12-hour leaderboard training run |
| `configs/production_large_4xh100_80gb.json` | 4x H100/H20 80GB+ | 12-hour 661k-parameter leaderboard run |
| `configs/finetune_mc65536_4xh100_80gb.json` | 4x H100/H20 80GB+ | 12-hour high-precision continuation with 65,536 teacher samples per MLP |

`batch_size` is per GPU.  `matmul_precision` is deliberately `highest`: TF32
roundoff is large enough to contaminate a `1e-6` final-MSE target and can also
break parity with the CPU submission path.

The production architecture has 29,890 trainable scalars.  A local official
`flopscope` width-256/depth-32 profile measured `4,662,878,080` tracked FLOPs
(`1.71%` of budget); including measured residual wall time, effective compute
was about `4.0%`, still below the score multiplier's 10% floor.

Run the smoke configuration:

```bash
python3 -m repro_external.aicrowd_whestbench.learned_residual.train \
  --config repro_external/aicrowd_whestbench/learned_residual/configs/smoke.json
```

Run the full-shape ablation:

```bash
python3 -m repro_external.aicrowd_whestbench.learned_residual.train \
  --config repro_external/aicrowd_whestbench/learned_residual/configs/ablation_1xh100_80gb.json
```

Run production DDP training.  Set `WHEST_DEPS_PATH` to the directory containing
the official `flopscope` and `whestbench` packages used by the asynchronous
full-100 monitor:

```bash
WHEST_DEPS_PATH=/tmp/whest-official-deps \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc-per-node=4 \
  -m repro_external.aicrowd_whestbench.learned_residual.train \
  --config repro_external/aicrowd_whestbench/learned_residual/configs/production_4xh100_80gb.json
```

The larger production configuration uses `hidden_dim=256`, six update blocks,
and a four-times-larger validation teacher.  Launch it with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
  -m repro_external.aicrowd_whestbench.learned_residual.train \
  --config repro_external/aicrowd_whestbench/learned_residual/configs/production_large_4xh100_80gb.json
```

Online training data are stateless functions of `(seed, rank, step)`, so a
resumed run does not replay earlier batches.  Checkpoints also persist the best
validation score, elapsed training time, optimizer, and CPU/CUDA RNG states.
The one-million-step value is only a safety ceiling: production stops after 12
hours of cumulative SGD time, including across resumes, and decays the learning
rate against that wall-clock budget.  Full-100 subprocesses run CPU-only beside
training and write reports under `output_dir/full100/`.

## Package an estimator

Copy the frozen best export under the required filename:

```bash
cp outputs/whestbench_learned_residual/production_4xh100_competitive/learned_residual_weights_best.npz \
  repro_external/aicrowd_whestbench/learned_residual/submission/learned_residual_weights.npz

whest validate \
  --estimator repro_external/aicrowd_whestbench/learned_residual/submission
```

Then run the official subprocess evaluator and verify both tracked FLOPs and
residual wall-time utilization before packaging.
