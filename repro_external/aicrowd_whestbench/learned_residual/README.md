# WhestBench learned residual closure

This directory implements an offline-trained black-box estimator while keeping
the official submission path transparent and FLOP-accounted.

## What the model predicts

For weights `W`, the supervised quantity is the per-neuron conditional mean

```text
y_l(W) = E_x[ReLU(... ReLU(x @ W_0) ... @ W_l)],  x ~ N(0, I).
```

The primary loss is final-layer MSE. Training labels are antithetic Monte Carlo
sample means generated from fresh random He-initialized MLPs. They are noisy but
unbiased conditional on `W`; validation uses a separate fixed set with many
more samples.

The network does not directly regress `y`. It predicts a bounded standardized
residual on top of diagonal Gaussian mean/variance propagation:

```text
prediction_l = analytic_mean_l
             + pre_std_l * residual_scale * layer_fraction * tanh(head_l)
```

The first layer remains exact. The final prediction is clamped non-negative.

## Recommended base model

Use `ResidualClosureNet(hidden_dim=32, blocks=2)` for the first real run. It is
a shared, forward message-passing network. At every layer it consumes analytic
moment features and the equivariant messages `W.T @ state` and
`(W**2).T @ state`. This has three useful properties:

- hidden-neuron permutations produce the same permutation in the output;
- parameters are shared across all 32 layers;
- inference is far below the 10% Phase-1 score-multiplier floor.

The 32-wide model has about 5.6k trainable scalars; a bigger generic Transformer
or flattened-weight MLP is not a useful first baseline. Promote to
`hidden_dim=48, blocks=3` only after the pilot beats diagonal propagation on a
fresh synthetic validation set.

For the Phase-1 shape `(depth=32, width=256)`, the 32-wide model was locally
profiled at 5,569 parameters and 392,200,256 tracked FLOPs per MLP, or 0.144% of
the 272B FLOP budget. The official effective-compute score also charges
residual wall time, so the packaged subprocess evaluation remains the final
compute gate.

## GPU configurations

| Stage | Config | Hardware | Purpose |
|---|---|---|---|
| smoke | `configs/smoke.json` | CPU/MPS | wiring only |
| pilot | `configs/pilot_1xa100_40gb.json` | 1x A100 40GB (80GB also fine) | decide whether residual learning has signal |
| production | `configs/production_4xh100_80gb.json` | 4x H100 80GB | larger teacher batches and more independent MLPs |

`batch_size` is per GPU. Training data are generated online, so host RAM and
storage are not bottlenecks. For the pilot, reserve 16 CPU cores, 64 GB RAM,
and roughly 2-4 GPU hours. The production config is intended as a 6-12 hour
DDP job; actual throughput depends heavily on batched 256x256 matmul efficiency.

Run the smoke test:

```bash
.venv/bin/python -m repro_external.aicrowd_whestbench.learned_residual.train \
  --config repro_external/aicrowd_whestbench/learned_residual/configs/smoke.json
```

Run the single-GPU pilot:

```bash
python -m repro_external.aicrowd_whestbench.learned_residual.train \
  --config repro_external/aicrowd_whestbench/learned_residual/configs/pilot_1xa100_40gb.json
```

Run four-GPU production training:

```bash
torchrun --standalone --nproc-per-node=4 \
  -m repro_external.aicrowd_whestbench.learned_residual.train \
  --config repro_external/aicrowd_whestbench/learned_residual/configs/production_4xh100_80gb.json
```

Each run writes JSONL metrics, resumable `.pt` checkpoints, and pickle-free
`learned_residual_weights_{best,last}.npz` exports under `output_dir`.

## Package an estimator

Copy the best export next to the submission entrypoint under the required name:

```bash
cp outputs/whestbench_learned_residual/pilot_1xa100/learned_residual_weights_best.npz \
  repro_external/aicrowd_whestbench/learned_residual/submission/learned_residual_weights.npz

whest validate \
  --estimator repro_external/aicrowd_whestbench/learned_residual/submission
```

Do not select checkpoints on the public mini targets. Use newly generated MLPs,
split by whole MLP, then perform one untouched full-100 evaluation after the
synthetic validation decision is frozen.
