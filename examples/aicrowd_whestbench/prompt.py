WHESTBENCH_PROMPT = """You are solving the ARC White-Box Estimation Challenge 2026
(WhestBench).

Task:
- You are given the weights of a randomly initialized square ReLU MLP.
- Inputs are standard normal vectors.
- Return an array with shape (depth, width), where each row estimates the
  per-neuron post-ReLU activation mean for that layer.
- The main score uses the final row's MSE against a high-sample Monte Carlo
  reference. Lower final-layer MSE is better.

Local interface:
- Write Python code defining `estimate(mlp, budget)`.
- `mlp.width`, `mlp.depth`, `mlp.seed`, and `mlp.weights` are available.
- `mlp.weights` is a list of NumPy arrays with shape (width, width).
- Return a finite NumPy-compatible array of shape (mlp.depth, mlp.width).
- You may also define `predict(mlp, budget)` or `class Estimator` with a
  `predict(self, mlp, budget)` method.

Official submission notes:
- The official AIcrowd starter kit expects an `Estimator` class with
  `predict(self, mlp, budget)` in `estimator.py`.
- In the official harness, use `flopscope.numpy` rather than plain `numpy` for
  FLOP-counted operations.
- The official package is submitted as a tarball or through `whest submit`.

Algorithm directions worth exploring:
- Mean and diagonal-variance propagation through ReLU moments.
- Full or low-rank covariance propagation.
- Hybrid analytic propagation plus limited Monte Carlo or randomized probes.
- Layer-wise corrections for correlation error in deeper networks.
- Compute-aware approximations that spend FLOPs only where final-layer MSE is
  most sensitive.
"""
