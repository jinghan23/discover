"""Evaluation entrypoint for the QMC plus radial Rao--Blackwell ablation."""

from repro_external.aicrowd_whestbench.baselines.monte_carlo import (
    RaoBlackwellEstimator,
)


class Estimator(RaoBlackwellEstimator):
    """Auto-detectable entrypoint for the Rao--Blackwell ablation."""
