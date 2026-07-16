import flopscope as flops
import flopscope.numpy as fnp
from whestbench.domain import MLP

from repro_external.aicrowd_whestbench.baselines.covariance_propagation import (
    Estimator as CovariancePropagationEstimator,
    _compress_third_cumulant,
    _third_cumulant_diagonal,
)
from repro_external.aicrowd_whestbench.baselines.mean_propagation import (
    Estimator as MeanPropagationEstimator,
)
from repro_external.aicrowd_whestbench.baselines.monte_carlo import (
    ControlVariateEstimator,
    Estimator as MonteCarloEstimator,
    ImportanceSamplingEstimator,
    PlainMonteCarloEstimator,
    QuasiMonteCarloEstimator,
    RaoBlackwellEstimator,
    WhitenedAntitheticRaoBlackwell10Estimator,
    WhitenedAntitheticRaoBlackwell15Estimator,
    _angular_whitening_transform,
    sample_count,
    whitened_antithetic_sample_count,
)


def _identity_mlp(width: int = 4, depth: int = 3) -> MLP:
    return MLP(
        width=width,
        depth=depth,
        weights=[fnp.eye(width, dtype=fnp.float32) for _ in range(depth)],
        seed=7,
        name="identity",
    )


def test_starter_baselines_satisfy_shape_and_finite_contract():
    mlp = _identity_mlp()

    for estimator_type in (MeanPropagationEstimator, CovariancePropagationEstimator):
        with flops.BudgetContext(flop_budget=10_000_000, quiet=True):
            prediction = estimator_type().predict(mlp, 10_000_000)

        assert prediction.shape == (mlp.depth, mlp.width)
        assert fnp.all(fnp.isfinite(prediction))


def test_starter_baselines_match_exact_first_layer_standard_normal_mean():
    mlp = _identity_mlp(depth=1)

    with flops.BudgetContext(flop_budget=10_000_000, quiet=True):
        expected = fnp.ones(mlp.width) / fnp.sqrt(2.0 * fnp.pi)
    for estimator_type in (MeanPropagationEstimator, CovariancePropagationEstimator):
        with flops.BudgetContext(flop_budget=10_000_000, quiet=True):
            prediction = estimator_type().predict(mlp, 10_000_000)

        assert fnp.allclose(prediction[0], expected)


def test_factored_k3_rank_compression_preserves_marginal_skewness():
    rng = fnp.random.default_rng(23)
    with flops.BudgetContext(flop_budget=10_000_000, quiet=True):
        factors = tuple(
            rng.standard_normal((3, 520), dtype=fnp.float64)
            for _ in range(3)
        )
        target_diagonal = fnp.array([0.25, -0.5, 0.75])
        compressed = _compress_third_cumulant(factors, target_diagonal)
        recovered_diagonal = _third_cumulant_diagonal(compressed)

    assert compressed[0].shape == (3, 515)
    assert fnp.allclose(recovered_diagonal, target_diagonal, atol=1e-10)


def test_monte_carlo_sample_count_targets_nine_percent_of_phase1_budget():
    assert sample_count(272_000_000_000, width=256, depth=32) == 5_808


def test_monte_carlo_is_seeded_and_satisfies_output_contract():
    mlp = _identity_mlp(width=2, depth=2)

    with flops.BudgetContext(flop_budget=1_000_000, quiet=True):
        first = MonteCarloEstimator().predict(mlp, 1_000_000)
    with flops.BudgetContext(flop_budget=1_000_000, quiet=True):
        second = MonteCarloEstimator().predict(mlp, 1_000_000)

    assert first.shape == (mlp.depth, mlp.width)
    assert fnp.all(fnp.isfinite(first))
    assert fnp.all(first == second)


def test_progressive_monte_carlo_stages_satisfy_output_contract():
    mlp = _identity_mlp(width=2, depth=2)

    estimator_types = (
        PlainMonteCarloEstimator,
        QuasiMonteCarloEstimator,
        ImportanceSamplingEstimator,
        ControlVariateEstimator,
        RaoBlackwellEstimator,
    )
    for estimator_type in estimator_types:
        with flops.BudgetContext(flop_budget=1_000_000, quiet=True):
            prediction = estimator_type().predict(mlp, 1_000_000)

        assert prediction.shape == (mlp.depth, mlp.width)
        assert fnp.all(fnp.isfinite(prediction))


def test_rao_blackwell_stage_reports_exact_first_layer_mean():
    mlp = _identity_mlp(width=4, depth=2)
    with flops.BudgetContext(flop_budget=10_000_000, quiet=True):
        prediction = RaoBlackwellEstimator().predict(mlp, 10_000_000)
        expected = fnp.ones(mlp.width) / fnp.sqrt(2.0 * fnp.pi)

    assert fnp.allclose(prediction[0], expected)


def test_whitened_antithetic_counts_are_even_and_grow_with_budget():
    count_10 = whitened_antithetic_sample_count(
        272_000_000_000,
        width=256,
        depth=32,
        target_fraction=0.10,
    )
    count_15 = whitened_antithetic_sample_count(
        272_000_000_000,
        width=256,
        depth=32,
        target_fraction=0.15,
    )

    assert count_10 % 2 == 0
    assert count_15 % 2 == 0
    assert count_15 > count_10


def test_angular_whitening_matches_identity_second_moment():
    rng = fnp.random.default_rng(19)
    with flops.BudgetContext(flop_budget=10_000_000, quiet=True):
        directions = rng.standard_normal((64, 4), dtype=fnp.float64)
        directions = directions / fnp.sqrt(
            fnp.sum(directions * directions, axis=1)
        )[:, None]
        transform = _angular_whitening_transform(directions)
        whitened = directions @ transform
        second_moment = 4.0 * (whitened.T @ whitened) / 64.0

    assert fnp.allclose(second_moment, fnp.eye(4), atol=1e-8)


def test_whitened_antithetic_rao_blackwell_contract_and_exact_first_layer():
    mlp = _identity_mlp(width=4, depth=2)
    expected = fnp.ones(mlp.width) / fnp.sqrt(2.0 * fnp.pi)

    for estimator_type in (
        WhitenedAntitheticRaoBlackwell10Estimator,
        WhitenedAntitheticRaoBlackwell15Estimator,
    ):
        with flops.BudgetContext(flop_budget=10_000_000, quiet=True):
            prediction = estimator_type().predict(mlp, 10_000_000)

        assert prediction.shape == (mlp.depth, mlp.width)
        assert fnp.all(fnp.isfinite(prediction))
        assert fnp.allclose(prediction[0], expected)
