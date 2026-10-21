import copy

import numpy as np
import pytest
import torch

from morphostate.models import (
    ConditionalNBModel,
    NBCovariateModel,
    RidgePredictor,
    model_from_config,
    nb_nll,
)
from morphostate.training import fit_conditional_nb, fit_nb_baseline


def test_nb_matches_distribution_and_poisson_like_large_counts():
    y = torch.tensor([[0, 1, 4], [10, 100, 10000]], dtype=torch.float64)
    log_mu = torch.tensor([[-3, 0, 2], [2, 4, 9]], dtype=torch.float64, requires_grad=True)
    theta = torch.tensor([0.1, 3, 20], dtype=torch.float64, requires_grad=True)
    reference = -torch.distributions.NegativeBinomial(
        total_count=theta, logits=log_mu - theta.log()
    ).log_prob(y)
    actual = nb_nll(y, log_mu, theta, "none")
    torch.testing.assert_close(actual, reference, rtol=1e-10, atol=1e-10)
    actual.mean().backward()
    assert torch.isfinite(log_mu.grad).all()
    assert torch.isfinite(theta.grad).all()


def test_nb_extreme_log_means_have_finite_gradients():
    y = torch.tensor([[0.0, 1e6]])
    means = torch.tensor([[-100.0, 100.0]], requires_grad=True)
    dispersion = torch.tensor([0.01, 100.0], requires_grad=True)
    loss = nb_nll(y, means, dispersion)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(means.grad).all()
    assert torch.isfinite(dispersion.grad).all()


def test_conditional_initial_mean_and_frozen_baseline_roundtrip():
    baseline = NBCovariateModel(2, 4)
    original = copy.deepcopy(baseline.state_dict())
    model = ConditionalNBModel(baseline, image_dim=3, rank=2, hidden_dim=5)
    x, c = torch.randn(8, 3), torch.randn(8, 2)
    torch.testing.assert_close(model.log_rate(x, c), baseline.log_rate(None, c))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    nb_nll(torch.ones(8, 4), model.log_rate(x, c), model.dispersion()).backward()
    optimizer.step()
    for key, value in original.items():
        torch.testing.assert_close(baseline.state_dict()[key], value, rtol=0, atol=0)
        torch.testing.assert_close(model.baseline.state_dict()[key], value, rtol=0, atol=0)
    assert all(not p.requires_grad and p.grad is None for p in model.baseline.parameters())
    restored = model_from_config(model.model_config())
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored.log_rate(x, c), model.log_rate(x, c))


@pytest.mark.parametrize("hidden_dim", [0, 5])
def test_zero_image_features_cannot_recalibrate_baseline_intercept(hidden_dim):
    baseline = NBCovariateModel(2, 4)
    model = ConditionalNBModel(baseline, image_dim=3, rank=2, hidden_dim=hidden_dim)
    with torch.no_grad():
        # The claim must hold after the loading has moved away from zero, not
        # merely because its initial value disables the entire image branch.
        model.gene_loading.weight.fill_(1.5)
        if hidden_dim:
            for module in model.encoder:
                if isinstance(module, torch.nn.Linear):
                    module.bias.fill_(0.75)
        else:
            assert model.encoder.bias is None
    x, c = torch.zeros(8, 3), torch.randn(8, 2)
    torch.testing.assert_close(model.state(x), torch.zeros(8, 2), rtol=0, atol=0)
    torch.testing.assert_close(model.log_rate(x, c), baseline.log_rate(None, c), rtol=0, atol=0)


def test_ridge_residual_and_collinear_covariates():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(200, 2))
    c = rng.normal(size=(200, 1))
    c = np.column_stack((c, c, np.ones(200)))
    target = 2 * c[:, :1] + 3 * x[:, :1] + 1
    predictor = RidgePredictor("residual", alpha=0).fit(x, c, target)
    baseline = RidgePredictor("covariate", alpha=0).fit(x, c, target)
    assert np.mean((predictor.predict(x, c) - target) ** 2) < 0.05
    assert np.mean((baseline.predict(x, c) - target) ** 2) > 5
    assert np.isfinite(predictor.predict(x, c)).all()


def test_conditional_learns_signal_and_training_is_reproducible():
    # A deliberately simple plumbing test; it is not a biological benchmark.
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        rng = np.random.default_rng(17)
        x = rng.normal(size=(240, 2)).astype(np.float32)
        c = rng.normal(size=(240, 1)).astype(np.float32)
        loadings = np.array([[0.9, -0.8, 0.7, -0.6]], dtype=np.float32)
        mu = np.exp(1.5 + 0.1 * c + x[:, :1] @ loadings)
        y = rng.poisson(mu).astype(np.float32)
        s = np.ones(240, dtype=np.float32)
        config = {"seed": 9, "epochs": 60, "batch_size": 64, "patience": 12, "lr": 0.03, "rank": 1}
        baseline, _ = fit_nb_baseline(c[:180], y[:180], s[:180], c[180:], y[180:], s[180:], config)
        before = copy.deepcopy(baseline.state_dict())
        args = (x[:180], c[:180], y[:180], s[:180], x[180:], c[180:], y[180:], s[180:], baseline, config)
        model, history = fit_conditional_nb(*args)
        repeat, repeat_history = fit_conditional_nb(*args)
        assert history == repeat_history
        assert model.training_summary["best_val_nll"] < baseline.training_summary["best_val_nll"] - 0.15
        assert model.training_summary["best_val_nll"] == min(row["val_nll"] for row in history if row["is_best"])
        for key, value in before.items():
            torch.testing.assert_close(baseline.state_dict()[key], value, rtol=0, atol=0)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(repeat.state_dict()[key], value, rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("weights", [[1, -1], [1], [0, 0]])
def test_invalid_sample_weights_rejected(weights):
    c = np.ones((2, 1), dtype=np.float32)
    y = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="weights"):
        fit_nb_baseline(c, y, np.ones(2), c, y, np.ones(2), {"epochs": 1, "sample_weights": weights})
