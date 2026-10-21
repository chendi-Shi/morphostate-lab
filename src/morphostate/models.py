"""Small, inspectable baselines for conditional image-to-count prediction.

The conditional model is a research baseline, not a claim of causal or identifiable
biological states. Its image increment is measured against a frozen covariate model.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def nb_nll(
    counts: Tensor,
    log_mu: Tensor,
    dispersion: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """Negative-binomial NLL, with variance ``mu + mu**2 / dispersion``.

    Includes the count-dependent gamma terms. Softplus expressions avoid taking
    ``exp(log_mu)`` and remain finite for large log means. ``none`` returns one
    loss per cell and gene; ``mean`` averages all entries.
    """
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if torch.any(dispersion <= 0):
        raise ValueError("dispersion must be strictly positive")
    y = counts.to(dtype=log_mu.dtype, device=log_mu.device)
    theta = dispersion.to(dtype=log_mu.dtype, device=log_mu.device)
    log_theta = theta.log()
    log_prob = (
        torch.lgamma(y + theta)
        - torch.lgamma(theta)
        - torch.lgamma(y + 1)
        - theta * F.softplus(log_mu - log_theta)
        - y * F.softplus(log_theta - log_mu)
    )
    loss = -log_prob
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def _inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("initial_dispersion must be positive")
    return float(value + np.log(-np.expm1(-value)))


class NBCovariateModel(nn.Module):
    """Log-linear count model conditioned only on provided covariates."""

    def __init__(
        self, covariate_dim: int, n_genes: int, initial_dispersion: float = 10.0
    ) -> None:
        super().__init__()
        if covariate_dim < 1 or n_genes < 1:
            raise ValueError("covariate_dim and n_genes must be positive")
        self.covariate_dim = int(covariate_dim)
        self.n_genes = int(n_genes)
        self.initial_dispersion = float(initial_dispersion)
        self.linear = nn.Linear(self.covariate_dim, self.n_genes)
        self.raw_dispersion = nn.Parameter(
            torch.full((self.n_genes,), _inverse_softplus(initial_dispersion))
        )

    def log_rate(self, x: Tensor | None, c: Tensor) -> Tensor:
        """Return log expected counts at reference exposure, without RNA input."""
        return self.linear(c)

    def dispersion(self) -> Tensor:
        return F.softplus(self.raw_dispersion) + 1e-5

    def model_config(self) -> dict[str, Any]:
        return {
            "type": "nb_covariate",
            "kwargs": {
                "covariate_dim": self.covariate_dim,
                "n_genes": self.n_genes,
                "initial_dispersion": self.initial_dispersion,
            },
        }


class ConditionalNBModel(nn.Module):
    """Frozen covariate baseline plus a low-rank image-dependent increment.

    Gene loadings start at zero, so the initial mean exactly equals the copied
    baseline. Dispersion is initialized from the baseline and then re-estimated.
    A zero hidden dimension uses a bias-free linear image encoder. Nonlinear
    encoders are anchored by subtracting their output at the zero image vector.
    Thus the image increment is zero at the feature origin; this does not center
    nonlinear states over the data distribution or identify a biological state.
    """

    def __init__(
        self,
        baseline: NBCovariateModel,
        image_dim: int,
        rank: int,
        hidden_dim: int = 0,
    ) -> None:
        super().__init__()
        if image_dim < 1 or rank < 1 or hidden_dim < 0:
            raise ValueError("image_dim/rank must be positive and hidden_dim nonnegative")
        self.image_dim = int(image_dim)
        self.rank = int(rank)
        self.hidden_dim = int(hidden_dim)
        self.n_genes = baseline.n_genes
        self.baseline = copy.deepcopy(baseline)
        self.baseline.requires_grad_(False)
        self.baseline.eval()
        if hidden_dim:
            self.encoder = nn.Sequential(
                nn.Linear(image_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, rank),
            )
        else:
            self.encoder = nn.Linear(image_dim, rank, bias=False)
        self.gene_loading = nn.Linear(rank, self.n_genes, bias=False)
        nn.init.zeros_(self.gene_loading.weight)
        self.raw_dispersion = nn.Parameter(
            baseline.raw_dispersion.detach().clone()
        )

    def train(self, mode: bool = True) -> ConditionalNBModel:
        super().train(mode)
        self.baseline.eval()
        return self

    def state(self, x: Tensor) -> Tensor:
        """Return origin-anchored coordinates; their rotation is not identifiable.

        Origin anchoring rules out a standalone image-branch intercept. It does
        not imply zero population mean for nonlinear coordinates or biological
        identifiability of their individual dimensions.
        """
        state = self.encoder(x)
        if self.hidden_dim:
            state = state - self.encoder(torch.zeros_like(x))
        return state

    def log_rate(self, x: Tensor, c: Tensor) -> Tensor:
        return self.baseline.log_rate(None, c) + self.gene_loading(self.state(x))

    def dispersion(self) -> Tensor:
        return F.softplus(self.raw_dispersion) + 1e-5

    def model_config(self) -> dict[str, Any]:
        return {
            "type": "conditional_nb",
            "baseline": self.baseline.model_config(),
            "kwargs": {
                "image_dim": self.image_dim,
                "rank": self.rank,
                "hidden_dim": self.hidden_dim,
            },
        }


def model_from_config(config: dict[str, Any]) -> NBCovariateModel | ConditionalNBModel:
    """Reconstruct architecture; load the saved state_dict separately."""
    if config["type"] == "nb_covariate":
        return NBCovariateModel(**config["kwargs"])
    if config["type"] == "conditional_nb":
        baseline = model_from_config(config["baseline"])
        if not isinstance(baseline, NBCovariateModel):
            raise ValueError("conditional baseline must be NBCovariateModel")
        return ConditionalNBModel(baseline=baseline, **config["kwargs"])
    raise ValueError(f"Unknown model type: {config['type']}")


class RidgePredictor:
    """Ridge baselines with a separate, unpenalized intercept.

    Residual mode regresses both targets and image features on the covariates,
    then fits an image model between their training-set residuals. Prediction
    uses the saved covariate-to-image nuisance fit to residualize new features.
    With alpha=0 this is classical linear partial regression, including for
    collinear covariates. The nuisance fits are *not* cross-fitted.
    """

    def __init__(self, mode: str = "image", alpha: float = 1.0) -> None:
        if mode not in {"image", "covariate", "residual"}:
            raise ValueError("mode must be image, covariate, or residual")
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        self.mode = mode
        self.alpha = float(alpha)
        self.components_: dict[str, dict[str, np.ndarray]] = {}

    @staticmethod
    def _array(value: np.ndarray) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64)
        if result.ndim != 2 or not np.isfinite(result).all():
            raise ValueError("Ridge inputs must be finite two-dimensional arrays")
        return result

    def _fit_component(self, design: np.ndarray, target: np.ndarray) -> dict:
        x_mean, y_mean = design.mean(axis=0), target.mean(axis=0)
        centered = design - x_mean
        response = target - y_mean
        # SVD handles collinear covariates and alpha=0 without an unstable inverse.
        u, singular, vt = np.linalg.svd(centered, full_matrices=False)
        denominator = singular**2 + self.alpha
        cutoff = np.finfo(np.float64).eps * max(centered.shape) * (
            singular.max() if singular.size else 0
        )
        factor = np.divide(
            singular,
            denominator,
            out=np.zeros_like(singular),
            where=(singular > cutoff) if self.alpha == 0 else (denominator > 0),
        )
        coef = (vt.T * factor) @ (u.T @ response)
        return {"coef": coef, "x_mean": x_mean, "y_mean": y_mean}

    @staticmethod
    def _predict_component(design: np.ndarray, component: dict) -> np.ndarray:
        return (design - component["x_mean"]) @ component["coef"] + component["y_mean"]

    def fit(self, x: np.ndarray, c: np.ndarray, log_y: np.ndarray) -> RidgePredictor:
        x, c, target = self._array(x), self._array(c), self._array(log_y)
        if not (len(x) == len(c) == len(target)) or len(target) < 1:
            raise ValueError("Ridge inputs must have the same nonzero row count")
        self.components_ = {}
        if self.mode in {"covariate", "residual"}:
            self.components_["covariate"] = self._fit_component(c, target)
        if self.mode == "residual":
            target = target - self._predict_component(c, self.components_["covariate"])
            self.components_["covariate_to_image"] = self._fit_component(c, x)
            x = x - self._predict_component(c, self.components_["covariate_to_image"])
        if self.mode in {"image", "residual"}:
            self.components_["image"] = self._fit_component(x, target)
        return self

    def predict(self, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        if not self.components_:
            raise RuntimeError("Call fit before predict")
        x, c = self._array(x), self._array(c)
        if len(x) != len(c):
            raise ValueError("Image and covariate row counts differ")
        if self.mode == "covariate":
            return self._predict_component(c, self.components_["covariate"])
        if self.mode == "residual":
            x = x - self._predict_component(c, self.components_["covariate_to_image"])
            return (
                self._predict_component(c, self.components_["covariate"])
                + self._predict_component(x, self.components_["image"])
            )
        return self._predict_component(x, self.components_["image"])
