"""Train-only, same-feature comparison heads, with validation-only rank selection.

These are independently implemented baselines, not reproductions of named papers.
CCA here is regularized linear CCA, not probabilistic partial CCA. Residual PCA is
supervised residual compression, not an implementation of the ResidPCA paper.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn


def _weights(value, n):
    w = np.ones(n) if value is None else np.asarray(value, dtype=float)
    if w.shape != (n,) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("sample weights must be finite, nonnegative and have positive sum")
    return w / w.sum()


class WeightedRidge:
    """Weighted ridge with unpenalized intercept and SVD solution."""
    def __init__(self, alpha=1.):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha

    def fit(self, x, y, weights=None):
        x, y = np.asarray(x, float), np.asarray(y, float)
        w = _weights(weights, len(x))
        self.x_mean = w @ x
        self.y_mean = w @ y
        a, b = (x - self.x_mean) * np.sqrt(w[:, None]), (y - self.y_mean) * np.sqrt(w[:, None])
        u, s, vt = np.linalg.svd(a, full_matrices=False)
        self.coef = (vt.T * (s / (s * s + self.alpha / len(x)))) @ u.T @ b
        return self

    def predict(self, x):
        return (np.asarray(x) - self.x_mean) @ self.coef + self.y_mean


def _rate(log1p_rate):
    return np.maximum(np.expm1(np.clip(log1p_rate, 0, 20)), 1e-8)


def _score(y, exposure, rate, weights):
    mu = np.maximum(rate * exposure[:, None], 1e-8)
    return float(weights @ np.mean(mu - y * np.log(mu), axis=1))


def _invsqrt(a, ridge):
    values, vectors = np.linalg.eigh(a)
    return (vectors * (1 / np.sqrt(np.maximum(values, 0) + ridge))) @ vectors.T


def fit_predict_comparisons(
    x_train, c_train, y_train, exposure_train,
    x_val, c_val, y_val, exposure_val, x_test, c_test, *,
    train_types=None, test_types=None, train_donors=None, sample_weight=None, val_sample_weight=None,
    seed=7, ranks=(2, 4), epochs=100, patience=15, hidden_dim=64, alpha=1.,
    learning_rate=1e-3, batch_size=256,
):
    """Return ``(test_rates, metadata)``; no test targets are accepted.

    Inputs must use the caller's train-fitted scaling and common reference exposure.
    Type mean is deliberately named an oracle and is omitted unless both type arrays
    are supplied. Validation counts select ranks and MLP epoch, never fit weights.
    """
    arrays = [np.asarray(v, dtype=float) for v in (x_train, c_train, y_train, x_val, c_val, y_val, x_test, c_test)]
    xt, ct, yt, xv, cv, yv, xs, cs = arrays
    if any(a.ndim != 2 or not np.isfinite(a).all() for a in arrays):
        raise ValueError("all feature, covariate and target arrays must be finite matrices")
    for x, c, y in ((xt, ct, yt), (xv, cv, yv)):
        if not len(x) or len(x) != len(c) or len(x) != len(y):
            raise ValueError("training/validation rows must be nonempty and aligned")
    if not len(xs) or len(xs) != len(cs) or min(xt.shape[1], ct.shape[1], yt.shape[1]) < 1:
        raise ValueError("test rows must be aligned and feature dimensions positive")
    if any(a.shape[1] != xt.shape[1] for a in (xv, xs)) or any(a.shape[1] != ct.shape[1] for a in (cv, cs)) or yt.shape[1] != yv.shape[1]:
        raise ValueError("feature, covariate or gene dimensions differ")
    if (yt < 0).any() or (yv < 0).any():
        raise ValueError("counts must be nonnegative")
    et, ev = np.asarray(exposure_train, float), np.asarray(exposure_val, float)
    if et.shape != (len(xt),) or ev.shape != (len(xv),) or not np.isfinite(et).all() or not np.isfinite(ev).all() or min(et.min(), ev.min()) <= 0:
        raise ValueError("exposures must be positive and aligned")
    ranks = sorted(set(int(r) for r in ranks))
    if not ranks or min(ranks) < 1 or epochs < 1 or patience < 1 or hidden_dim < 1 or batch_size < 1:
        raise ValueError("ranks and training parameters must be positive")
    wt, wv = _weights(sample_weight, len(xt)), _weights(val_sample_weight, len(xv))
    log_target = np.log1p(yt / et[:, None])
    # Poisson constant-rate MLE, including varying exposure and donor weights.
    global_mean = (wt @ yt) / (wt @ et)
    rates = {"train_mean": np.tile(global_mean, (len(xs), 1))}
    metadata = {"seed": int(seed), "selection": "validation donor-weighted Poisson score", "models": {"train_mean": {"fit": "train-only weighted count/exposure MLE"}}}
    if (train_types is None) != (test_types is None):
        raise ValueError("both train_types and test_types are required for the oracle")
    if train_types is not None:
        tt, ts = np.asarray(train_types, str), np.asarray(test_types, str)
        if tt.shape != (len(xt),) or ts.shape != (len(xs),):
            raise ValueError("type arrays must match their respective split")
        lookup = {t: (wt[tt == t] @ yt[tt == t]) / (wt[tt == t] @ et[tt == t]) for t in np.unique(tt) if wt[tt == t].sum() > 0}
        rates["type_mean_oracle"] = np.stack([lookup.get(t, global_mean) for t in ts])
        metadata["models"]["type_mean_oracle"] = {"input_mode": "rna_oracle", "unseen_type_fallback": "train_mean"}

    zt, zv, zs = np.concatenate((xt, ct), 1), np.concatenate((xv, cv), 1), np.concatenate((xs, cs), 1)
    ridge = WeightedRidge(alpha).fit(zt, log_target, wt)
    fitted = ridge.predict(zt) - ridge.y_mean
    _, _, vtr = np.linalg.svd(fitted * np.sqrt(wt[:, None]), full_matrices=False)
    cov = WeightedRidge(alpha).fit(ct, log_target, wt)
    residual = log_target - cov.predict(ct)
    image_cov = WeightedRidge(alpha).fit(ct, xt, wt)
    xr, xrv, xrs = xt - image_cov.predict(ct), xv - image_cov.predict(cv), xs - image_cov.predict(cs)
    if train_donors is not None:
        donors = np.asarray(train_donors, str)
        if donors.shape != (len(xt),):
            raise ValueError("train_donors must align with training rows")
        unique = np.unique(donors)
        if len(unique) >= 2:
            # Each nuisance residual is obtained without fitting on its donor.
            # At most five inner folds keep the baseline affordable on large cohorts.
            fold_donors = np.array_split(np.random.default_rng(seed).permutation(unique), min(5, len(unique)))
            oof_x, oof_y = np.empty_like(xt), np.empty_like(log_target)
            folds = []
            for held_donors in fold_donors:
                held = np.isin(donors, held_donors)
                nuisance_x = WeightedRidge(alpha).fit(ct[~held], xt[~held], wt[~held])
                nuisance_y = WeightedRidge(alpha).fit(ct[~held], log_target[~held], wt[~held])
                oof_x[held] = xt[held] - nuisance_x.predict(ct[held])
                oof_y[held] = log_target[held] - nuisance_y.predict(ct[held])
                folds.append({"held_out_donors": held_donors.tolist(), "fit_donors": unique[~np.isin(unique, held_donors)].tolist()})
            crossfit_head = WeightedRidge(alpha).fit(oof_x, oof_y, wt)
            rates["donor_crossfit_residual_ridge"] = _rate(cov.predict(cs) + crossfit_head.predict(xrs))
            metadata["models"]["donor_crossfit_residual_ridge"] = {"alpha": alpha, "inner_folds": folds,
                "note": "Out-of-fold training nuisance residuals; no orthogonality or causal claim."}
        else:
            metadata["models"]["donor_crossfit_residual_ridge"] = {"status": "unavailable", "reason": "requires at least two training donors"}
    _, _, vtp = np.linalg.svd(residual * np.sqrt(wt[:, None]), full_matrices=False)
    xm, ym = wt @ xr, wt @ residual
    xc, yc = xr - xm, residual - ym
    wx = _invsqrt(xc.T @ (wt[:, None] * xc), alpha / len(xt))
    wy = _invsqrt(yc.T @ (wt[:, None] * yc), alpha / len(xt))
    uc, _, _ = np.linalg.svd(wx @ xc.T @ (wt[:, None] * yc) @ wy, full_matrices=False)
    for mode in ("reduced_rank_regression", "residual_pca_ridge", "regularized_partial_cca"):
        candidates = []
        for rank in ranks:
            if mode == "reduced_rank_regression":
                basis = vtr[:rank].T
                pv = (ridge.predict(zv) - ridge.y_mean) @ basis @ basis.T + ridge.y_mean
                ps = (ridge.predict(zs) - ridge.y_mean) @ basis @ basis.T + ridge.y_mean
            elif mode == "residual_pca_ridge":
                basis = vtp[:rank].T
                head = WeightedRidge(alpha).fit(xr, residual @ basis, wt)
                pv, ps = cov.predict(cv) + head.predict(xrv) @ basis.T, cov.predict(cs) + head.predict(xrs) @ basis.T
            else:
                basis = wx @ uc[:, :rank]
                head = WeightedRidge(alpha).fit(xc @ basis, residual, wt)
                pv, ps = cov.predict(cv) + head.predict((xrv - xm) @ basis), cov.predict(cs) + head.predict((xrs - xm) @ basis)
            candidates.append((_score(yv, ev, _rate(pv), wv), rank, _rate(ps), basis.shape[1]))
        best = min(candidates, key=lambda v: (v[0], v[1]))
        rates[mode] = best[2]
        metadata["models"][mode] = {"selected_rank": best[1], "effective_rank": best[3], "alpha": alpha,
            "candidates": [{"rank": r, "validation_score": score} for score, r, _, _ in candidates]}

    # Local RNG context avoids changing the calling experiment's global RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = nn.Sequential(nn.Linear(zt.shape[1], hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, yt.shape[1]))
        with torch.no_grad():
            model[-1].bias.copy_(torch.tensor(np.log(np.maximum(global_mean, 1e-5)), dtype=torch.float32))
            model[-1].weight.mul_(0.01)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        a, b, e, w = [torch.tensor(v, dtype=torch.float32) for v in (zt, yt, et, wt * len(wt))]
        va, vb, ve, vw = [torch.tensor(v, dtype=torch.float32) for v in (zv, yv, ev, wv)]
        generator = torch.Generator().manual_seed(seed)
        best_score, best_epoch, best_state, stale, history = float("inf"), 0, None, 0, []
        for epoch in range(epochs):
            model.train()
            for ids in torch.randperm(len(a), generator=generator).split(batch_size):
                log_mu = model(a[ids]).clamp(-20, 20) + e[ids, None].log()
                loss = ((log_mu.exp() - b[ids] * log_mu).mean(1) * w[ids]).mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 10.)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                lm = model(va).clamp(-20, 20) + ve[:, None].log()
                score = float(((lm.exp() - vb * lm).mean(1) * vw).sum())
            if not np.isfinite(score):
                raise FloatingPointError("Poisson MLP produced non-finite validation loss")
            history.append(score)
            if score < best_score:
                best_score, best_epoch, best_state, stale = score, epoch + 1, copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
            if stale >= patience:
                break
        model.load_state_dict(best_state)
        with torch.no_grad():
            rates["poisson_mlp"] = model(torch.tensor(zs, dtype=torch.float32)).clamp(-20, 20).exp().numpy()
    metadata["models"]["poisson_mlp"] = {"hidden_dim": hidden_dim, "selected_epoch": best_epoch,
        "validation_score": best_score, "validation_history": history, "learning_rate": learning_rate}
    return rates, metadata
