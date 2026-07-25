from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import torch
from whestbench import SetupContext
from whestbench.domain import MLP

from repro_external.aicrowd_whestbench.learned_residual.model import (
    ModelConfig,
    ResidualClosureNet,
    export_model,
)
from repro_external.aicrowd_whestbench.learned_residual.submission.estimator import (
    Estimator,
)
from repro_external.aicrowd_whestbench.learned_residual.train import (
    _step_generator,
    monte_carlo_targets,
    sample_he_weights,
)


def test_model_shape_first_layer_and_gradients():
    torch.manual_seed(7)
    config = ModelConfig(width=4, depth=3, hidden_dim=8, blocks=1)
    model = ResidualClosureNet(config)
    weights = torch.randn(2, config.depth, config.width, config.width) * (2 / 4) ** 0.5

    prediction, baseline, correction_limit = model(weights)
    assert prediction.shape == (2, config.depth, config.width)
    assert correction_limit.shape == prediction.shape
    assert torch.equal(prediction[:, 0], baseline[:, 0])
    assert torch.equal(correction_limit[:, 0], torch.zeros_like(correction_limit[:, 0]))

    prediction[:, -1].square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_antithetic_monte_carlo_target_shape():
    generator = torch.Generator().manual_seed(11)
    weights = sample_he_weights(
        2, 3, 4, device=torch.device("cpu"), generator=generator
    )
    targets = monte_carlo_targets(
        weights,
        32,
        8,
        antithetic=True,
        generator=generator,
        randomized_lattice=True,
        radial_rao_blackwell=True,
        exact_first_layer=True,
    )
    assert targets.shape == (2, 3, 4)
    assert torch.isfinite(targets).all()
    assert (targets >= 0).all()
    exact_first = torch.sqrt(weights[:, 0].square().sum(dim=1)) / (2 * torch.pi) ** 0.5
    assert torch.allclose(targets[:, 0], exact_first)


def test_independent_hidden_permutation_equivariance():
    torch.manual_seed(13)
    config = ModelConfig(width=5, depth=4, hidden_dim=7, blocks=1)
    model = ResidualClosureNet(config).eval()
    torch.nn.init.normal_(model.local_residual_head.weight, std=0.1)
    torch.nn.init.normal_(model.global_residual_head.weight, std=0.1)
    weights = torch.randn(1, config.depth, config.width, config.width) * (2 / 5) ** 0.5

    with torch.no_grad():
        expected, _, _ = model(weights)
    perm = torch.tensor([2, 4, 1, 0, 3])
    transformed = weights.clone()
    transformed[:, 1] = transformed[:, 1, :, perm]
    transformed[:, 2] = transformed[:, 2, perm, :]
    with torch.no_grad():
        actual, _, _ = model(transformed)

    assert torch.allclose(actual[:, :1], expected[:, :1], atol=2e-5, rtol=2e-5)
    assert torch.allclose(actual[:, 1], expected[:, 1, perm], atol=2e-5, rtol=2e-5)
    assert torch.allclose(actual[:, 2:], expected[:, 2:], atol=2e-5, rtol=2e-5)


def test_step_generators_make_resumes_stateless():
    device = torch.device("cpu")
    first = torch.randn(8, generator=_step_generator(device, 101, 73))
    second = torch.randn(8, generator=_step_generator(device, 101, 73))
    different_step = torch.randn(8, generator=_step_generator(device, 101, 74))
    assert torch.equal(first, second)
    assert not torch.equal(first, different_step)


def test_exported_flopscope_estimator_matches_torch(tmp_path: Path):
    torch.manual_seed(19)
    config = ModelConfig(width=4, depth=3, hidden_dim=6, blocks=2)
    model = ResidualClosureNet(config).eval()
    # Make parity exercise a non-zero learned correction, not just the baseline.
    torch.nn.init.normal_(model.local_residual_head.weight, std=0.1)
    torch.nn.init.normal_(model.local_residual_head.bias, std=0.02)
    torch.nn.init.normal_(model.global_residual_head.weight, std=0.1)
    torch.nn.init.normal_(model.global_residual_head.bias, std=0.02)
    weights = torch.randn(1, config.depth, config.width, config.width) * (2 / 4) ** 0.5
    with torch.no_grad():
        expected, _, _ = model(weights)

    export_model(model, tmp_path / "learned_residual_weights.npz")
    estimator = Estimator()
    estimator.setup(
        SetupContext(
            width=config.width,
            depth=config.depth,
            flop_budget=100_000_000,
            api_version="1.0",
            submission_dir=str(tmp_path),
        )
    )
    mlp = MLP(
        width=config.width,
        depth=config.depth,
        weights=[fnp.asarray(layer.numpy()) for layer in weights[0]],
        seed=3,
    )
    with flops.BudgetContext(flop_budget=100_000_000, quiet=True):
        actual = estimator.predict(mlp, 100_000_000)

    assert actual.shape == (config.depth, config.width)
    assert fnp.allclose(actual, fnp.asarray(expected[0].numpy()), atol=2e-5, rtol=2e-5)
