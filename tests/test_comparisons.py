import numpy as np
import pytest

from morphostate.comparisons import WeightedRidge, fit_predict_comparisons


def fixture(seed=3):
    rng = np.random.default_rng(seed)
    x, c = rng.normal(size=(48, 4)), rng.normal(size=(48, 2))
    exposure = rng.uniform(.5, 1.5, 48)
    y = rng.poisson(np.exp(.3 * x[:, :3] + .2 * c[:, :1]) * exposure[:, None])
    return (x[:24], c[:24], y[:24], exposure[:24], x[24:36], c[24:36], y[24:36], exposure[24:36], x[36:], c[36:])


def test_heads_are_reproducible_and_test_target_free():
    args = fixture()
    opts = dict(epochs=3, patience=2, ranks=(1, 2), seed=17, train_donors=np.repeat(["a", "b", "c"], 8))
    a, meta_a = fit_predict_comparisons(*args, **opts)
    b, meta_b = fit_predict_comparisons(*args, **opts)
    assert set(a) == {"train_mean", "reduced_rank_regression", "residual_pca_ridge", "regularized_partial_cca", "poisson_mlp", "donor_crossfit_residual_ridge"}
    for name, rates in a.items():
        assert rates.shape == (12, 3)
        assert np.isfinite(rates).all() and (rates >= 0).all()
        np.testing.assert_array_equal(rates, b[name])
    assert meta_a == meta_b
    for fold in meta_a["models"]["donor_crossfit_residual_ridge"]["inner_folds"]:
        assert not set(fold["held_out_donors"]) & set(fold["fit_donors"])


def test_weighted_mean_uses_exposure_and_oracle_unseen_fallback():
    args = fixture()
    w = np.arange(1, 25)
    rates, meta = fit_predict_comparisons(*args, epochs=1, ranks=(1,), sample_weight=w,
        train_types=np.repeat("T", 24), test_types=np.repeat("unseen", 12), train_donors=np.repeat("one", 24))
    expected = (w @ args[2]) / (w @ args[3])
    np.testing.assert_allclose(rates["train_mean"][0], expected)
    np.testing.assert_allclose(rates["type_mean_oracle"], rates["train_mean"])
    assert meta["models"]["donor_crossfit_residual_ridge"]["status"] == "unavailable"


def test_weighted_ridge_ignores_zero_weight_outlier():
    x = np.arange(10.)[:, None]
    y = 2 * x + 3
    a = WeightedRidge(.001).fit(x, y)
    b = WeightedRidge(.0011).fit(np.vstack((x, [[999]])), np.vstack((y, [[-99999]])), np.r_[np.ones(10), 0])
    np.testing.assert_allclose(a.predict(x), b.predict(x))


@pytest.mark.parametrize("invalid", [{"ranks": ()}, {"epochs": 0}, {"train_types": ["a"]}, {"sample_weight": np.zeros(24)}])
def test_invalid_baseline_configuration(invalid):
    with pytest.raises(ValueError):
        fit_predict_comparisons(*fixture(), **invalid)
