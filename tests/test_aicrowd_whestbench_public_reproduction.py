import flopscope as flops
import flopscope.numpy as fnp
from whestbench.domain import MLP

from repro_external.aicrowd_whestbench.public_reproductions.evaaaz_rqmc_rao_blackwell import (
    _exact_first_layer_mean,
    _first_primes,
    _lattice_generator,
    _lattice_uniforms,
    _sample_count as rqmc_sample_count,
)
from repro_external.aicrowd_whestbench.public_reproductions.german_alfaro_311690_whitened_antithetic import (
    Estimator,
    _sample_count,
)
from repro_external.aicrowd_whestbench.public_reproductions.pscamillo_314331_scalar_corrected_gaussian_closure import (
    FINAL_LAYER_CORRECTION,
    Estimator as ScalarCorrectedGaussianClosureEstimator,
    UncorrectedEstimator,
)


def test_phase1_sample_count_matches_public_submission():
    assert _sample_count(272_000_000_000, width=256, depth=32) == 9826


def test_estimator_satisfies_official_shape_contract_on_small_mlp():
    width = 2
    depth = 2
    weights = [fnp.eye(width, dtype=fnp.float32) for _ in range(depth)]
    mlp = MLP(width=width, depth=depth, weights=weights, seed=7, name="small")

    with flops.BudgetContext(flop_budget=1_000_000, quiet=True):
        prediction = Estimator().predict(mlp, 1_000_000)

    assert prediction.shape == (depth, width)
    assert prediction.dtype == fnp.float32
    assert fnp.all(prediction[0] == 0.0)
    assert fnp.all(fnp.isfinite(prediction[-1]))


def test_evaaaz_generator_uses_fractional_square_roots_of_primes():
    assert _first_primes(8) == (2, 3, 5, 7, 11, 13, 17, 19)

    with flops.BudgetContext(flop_budget=10_000, quiet=True):
        generator = _lattice_generator(3)
        expected = fnp.sqrt(fnp.array([2.0, 3.0, 5.0]))
        expected = expected - fnp.floor(expected)
        assert fnp.allclose(generator, expected)


def test_evaaaz_phase1_sample_count_matches_compute_target():
    assert rqmc_sample_count(272_000_000_000, width=256, depth=32) == 26_858


def test_evaaaz_lattice_uses_one_cranley_patterson_shift():
    with flops.BudgetContext(flop_budget=100_000, quiet=True):
        generator = fnp.array([0.25, 0.6])
        shift = fnp.array([0.1, 0.3])
        uniforms = _lattice_uniforms(3, generator, shift)
        expected = fnp.array([[0.1, 0.3], [0.35, 0.9], [0.6, 0.5]])
        assert fnp.allclose(uniforms, expected)


def test_evaaaz_rao_blackwell_mean_uses_first_weight_column_norms():
    weight = fnp.array([[3.0, 0.0], [4.0, 2.0]], dtype=fnp.float32)
    with flops.BudgetContext(flop_budget=10_000, quiet=True):
        exact = _exact_first_layer_mean(weight)
        expected = fnp.array([5.0, 2.0]) / fnp.sqrt(2.0 * fnp.pi)
        assert fnp.allclose(exact, expected)


def test_pscamillo_reproduction_only_corrects_final_layer():
    width = 4
    depth = 3
    weights = [fnp.eye(width, dtype=fnp.float32) for _ in range(depth)]
    mlp = MLP(width=width, depth=depth, weights=weights, seed=11, name="small")

    with flops.BudgetContext(flop_budget=10_000_000, quiet=True):
        uncorrected = UncorrectedEstimator().predict(mlp, 10_000_000)
    with flops.BudgetContext(flop_budget=10_000_000, quiet=True):
        corrected = ScalarCorrectedGaussianClosureEstimator().predict(
            mlp, 10_000_000
        )

    assert corrected.shape == (depth, width)
    assert fnp.allclose(corrected[:-1], uncorrected[:-1])
    assert fnp.allclose(
        corrected[-1], uncorrected[-1] * FINAL_LAYER_CORRECTION
    )
    assert fnp.all(fnp.isfinite(corrected))
