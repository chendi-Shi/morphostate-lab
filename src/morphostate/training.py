"""Deterministic minibatch training with validation-only early stopping."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch

from .models import ConditionalNBModel, NBCovariateModel, nb_nll


def set_seed(seed: int) -> None:
    """Request deterministic kernels on a fixed software/hardware environment."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def _prepare(x, c, y, s) -> tuple[torch.Tensor, ...]:
    c, y = np.asarray(c, dtype=np.float32), np.asarray(y, dtype=np.float32)
    if c.ndim != 2 or y.ndim != 2 or len(c) != len(y) or not len(y):
        raise ValueError("Covariates/counts must be nonempty 2D arrays with equal rows")
    if not np.isfinite(c).all() or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Covariates/counts must be finite; counts must be nonnegative")
    if not np.equal(y, np.floor(y)).all():
        raise ValueError("Negative-binomial training requires integer counts")
    s = np.asarray(s, dtype=np.float32).reshape(-1)
    if len(s) != len(y) or not np.isfinite(s).all() or (s <= 0).any():
        raise ValueError("Exposure must be finite, strictly positive, and match rows")
    if x is None:
        x = np.zeros((len(y), 1), dtype=np.float32)
    else:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all():
            raise ValueError("Image features must be finite 2D arrays matching count rows")
    return tuple(torch.from_numpy(np.ascontiguousarray(a)) for a in (x, c, y, s))


def _weights(value, rows: int) -> torch.Tensor:
    weights = np.ones(rows, dtype=np.float32) if value is None else np.asarray(value, dtype=np.float32)
    if weights.shape != (rows,) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Sample weights must be a finite positive vector matching rows")
    return torch.from_numpy(np.ascontiguousarray(weights / weights.mean()))


@torch.no_grad()
def _evaluate(model, arrays, weights, batch_size, device) -> float:
    model.eval()
    numerator = 0.0
    denominator = 0.0
    for start in range(0, len(arrays[2]), batch_size):
        x, c, y, s = [a[start : start + batch_size].to(device) for a in arrays]
        w = weights[start : start + batch_size].to(device)
        loss = nb_nll(y, model.log_rate(x, c) + s.log()[:, None], model.dispersion(), "none").mean(1)
        numerator += float((loss * w).sum().cpu())
        denominator += float(w.sum().cpu())
    return numerator / denominator


def _fit(model, train, val, config: dict[str, Any]):
    device_name = config.get("device", "cpu")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use device='cpu' or 'auto'")
    epochs = int(config.get("epochs", 100))
    batch_size = int(config.get("batch_size", 256))
    patience = int(config.get("patience", 15))
    min_delta = float(config.get("min_delta", 1e-4))
    if epochs < 1 or batch_size < 1 or patience < 1 or min_delta < 0:
        raise ValueError("epochs, batch_size, patience must be positive; min_delta >= 0")
    if train[1].shape[1] != val[1].shape[1] or train[2].shape[1] != val[2].shape[1]:
        raise ValueError("Train and validation covariate/gene dimensions differ")
    train_weights = _weights(config.get("sample_weights"), len(train[2]))
    val_weights = _weights(config.get("val_sample_weights"), len(val[2]))
    model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config.get("lr", 0.01)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    generator = np.random.default_rng(int(config.get("seed", 7)))
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_reference = float("inf")
    bad_epochs = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        ordering = generator.permutation(len(train[2]))
        for start in range(0, len(ordering), batch_size):
            index = torch.from_numpy(ordering[start : start + batch_size])
            x, c, y, s = [a[index].to(device) for a in train]
            weights = train_weights[index].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss_per_cell = nb_nll(
                y, model.log_rate(x, c) + s.log()[:, None], model.dispersion(), "none"
            ).mean(1)
            # Weights have global mean one. Batch-local normalization would
            # introduce a ratio-estimator bias when donors have different sizes.
            loss = (loss_per_cell * weights).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss; inspect inputs or lower lr")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=float(config.get("max_grad_norm", 5.0)),
                error_if_nonfinite=True,
            )
            optimizer.step()
        train_loss = _evaluate(model, train, train_weights, batch_size, device)
        val_loss = _evaluate(model, val, val_weights, batch_size, device)
        if not np.isfinite(val_loss) or not np.isfinite(train_loss):
            raise FloatingPointError("Non-finite evaluation NLL")
        # Always retain the lowest observed validation loss. ``min_delta`` only
        # controls patience, not whether a better checkpoint is saved.
        improved = val_loss < best_loss
        history.append({"epoch": epoch, "train_nll": train_loss, "val_nll": val_loss, "is_best": improved})
        if improved:
            best_loss, best_epoch = val_loss, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if val_loss < patience_reference - min_delta:
            patience_reference = val_loss
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break
    model.load_state_dict(best_state)
    model.cpu().eval()
    model.training_summary = {
        "best_epoch": best_epoch,
        "best_val_nll": best_loss,
        "epochs_completed": len(history),
        "restored_best_validation_checkpoint": True,
    }
    return model, history


def fit_nb_baseline(c_train, y_train, s_train, c_val, y_val, s_val, config):
    """Fit a covariate-only NB model and restore its best validation epoch."""
    set_seed(int(config.get("seed", 7)))
    train = _prepare(None, c_train, y_train, s_train)
    val = _prepare(None, c_val, y_val, s_val)
    model = NBCovariateModel(
        train[1].shape[1], train[2].shape[1],
        initial_dispersion=float(config.get("initial_dispersion", 10.0)),
    )
    with torch.no_grad():
        model.linear.weight.zero_()
        weights = _weights(config.get("sample_weights"), len(train[2]))
        mean = (train[2] * weights[:, None]).sum(0) / (train[3] * weights).sum()
        model.linear.bias.copy_(mean.clamp_min(1e-4).log())
    return _fit(model, train, val, config)


def fit_conditional_nb(
    x_train, c_train, y_train, s_train,
    x_val, c_val, y_val, s_val, baseline, config,
):
    """Fit an image increment while preserving an independent frozen baseline."""
    set_seed(int(config.get("seed", 7)))
    train = _prepare(x_train, c_train, y_train, s_train)
    val = _prepare(x_val, c_val, y_val, s_val)
    if train[0].shape[1] != val[0].shape[1]:
        raise ValueError("Train and validation image dimensions differ")
    if baseline.n_genes != train[2].shape[1] or baseline.covariate_dim != train[1].shape[1]:
        raise ValueError("Baseline dimensions do not match the conditional training data")
    model = ConditionalNBModel(
        baseline=baseline,
        image_dim=train[0].shape[1],
        rank=int(config.get("rank", 4)),
        hidden_dim=int(config.get("hidden_dim", 0)),
    )
    return _fit(model, train, val, config)
