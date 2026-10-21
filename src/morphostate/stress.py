"""A constructive counterexample for pessimistic relative-label learning.

When a null baseline's counts belong to the ambiguity set, relative Poisson
worst-case risk cannot prefer a different mean: choosing that feasible null
label yields the nonnegative Poisson KL from null to the proposed mean. A
positive morphology-visible state may therefore be suppressed by this objective.
This generator constructs that situation; it is a falsification fixture, not a
simulation asserting that real spatial transcriptomics follows these rules.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .budgeted import BudgetedLabelSet
from .data import Dataset, save_dataset
from .geometry import GraphBlock, _csr_from_rows, save_graphs


def make_stress_case(output, seed=7, donors=6, pairs_per_donor=4, state_strength=0.5,
                     state_prevalence=0.25, ambiguity=0.6, allow_null=True):
    """Write a paired-cell state-erasure counterexample and exact null witnesses.

    ``state_strength`` is a fraction of the 20-count state-gene baseline (rounded
    to integer molecules). ``state_prevalence`` is the fraction of ACTIVE PAIRS,
    not positive cells: each active pair contains one positive and one negative
    state direction. At least one pair per donor is active when prevalence > 0;
    the realized fractions are recorded. ``ambiguity`` is the per-cell/gene
    fraction of counts assigned to two-owner rows, rounded down. An insufficient
    ambiguity fraction is rejected rather than silently enlarging its support.

    ``allow_null=False`` retains the same support but imposes zero displacement
    budget, excluding the null whenever state differs. Background is explicit
    with zero mass and no candidate background edges. Counts and rates coincide
    because every cell has the same library (80). Labels come from the latent
    synthetic state and are never consumed as classifier/training covariates.
    """
    if (not isinstance(donors, int) or donors < 3 or not isinstance(pairs_per_donor, int)
            or pairs_per_donor < 2):
        raise ValueError("Need >=3 donors and >=2 pairs per donor")
    if (not np.isfinite(state_strength) or not 0 <= state_strength < 1
            or not np.isfinite(state_prevalence) or not 0 <= state_prevalence <= 1
            or not np.isfinite(ambiguity) or not 0 <= ambiguity <= 1):
        raise ValueError("state_strength must be in [0,1), prevalence/ambiguity in [0,1]")
    if not isinstance(allow_null, (bool, np.bool_)):
        raise ValueError("allow_null must be boolean")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Stress output is not empty: {output}")
    if donors * pairs_per_donor > 100_000:
        raise ValueError("This is a bounded pilot generator; reduce donors*pairs_per_donor")
    shift = int(round(20 * state_strength))
    if state_strength > 0:
        shift = max(1, min(19, shift))
    active_count = int(round(state_prevalence * pairs_per_donor))
    if state_prevalence > 0:
        active_count = max(1, min(pairs_per_donor, active_count))
    rng = np.random.default_rng(seed)
    genes = ["STATE_UP", "STATE_DOWN", "TYPE_A", "TYPE_B"]
    cells, observed_rows, null_rows, feature_rows, covariate_rows = [], [], [], [], []
    state_labels, latent_scores, pair_ids = [], [], []
    blocks, proof_rows, witness_arrays = [], [], {}
    global_start = 0
    for donor in range(donors):
        donor_id, slide_id = f"donor_{donor:03d}", f"slide_{donor:03d}"
        active_pairs = set(rng.choice(pairs_per_donor, active_count, replace=False).tolist())
        block_observed, block_null, block_ids = [], [], []
        for pair in range(pairs_per_donor):
            pair_id = f"{donor_id}_pair_{pair:03d}"
            type_index = pair % 2
            direction = int(rng.choice([-1, 1]))
            active = pair in active_pairs and shift > 0
            for side in range(2):
                baseline = np.array([20, 20, 28, 12] if type_index == 0 else [20, 20, 12, 28], dtype=int)
                state = (1 if side == 0 else -1) * direction if active else 0
                observed = baseline + np.array([state * shift, -state * shift, 0, 0])
                cell_id = f"{slide_id}::pair{pair:03d}_{side}"
                block_ids.append(cell_id)
                cell_type = "type_A" if type_index == 0 else "type_B"
                # Alternate pair axes; the molecular transfer remains within
                # that physical pair and never crosses type or donor boundaries.
                x, y = (10 * pair + side, 0) if pair % 2 == 0 else (10 * pair, side)
                cells.append({"cell_id": cell_id, "source_cell_id": f"pair{pair:03d}_{side}",
                              "donor_id": donor_id, "slide_id": slide_id, "cell_type": cell_type,
                              "x_centroid": x, "y_centroid": y})
                block_observed.append(observed)
                block_null.append(baseline)
                observed_rows.append(observed)
                null_rows.append(baseline)
                onehot = np.eye(2)[type_index]
                covariate_rows.append(onehot)
                # A clean morphology-visible state is intentional: this is a
                # counterexample about the learning objective, not image noise.
                feature_rows.append([*onehot, float(state), direction * (1 if side == 0 else -1),
                                     rng.normal(0, .1), rng.normal(0, .1)])
                latent_scores.append(state)
                pair_ids.append(pair_id)
                state_labels.append({"cell_id": cell_id, "donor_id": donor_id, "cell_type": cell_type,
                                     "state": "positive_state", "label": int(state > 0),
                                     "latent_state_score": state, "pair_id": pair_id,
                                     "neighbor_direction": direction,
                                     "label_source": "independent_synthetic_latent_state"})
        block_observed, block_null = np.asarray(block_observed), np.asarray(block_null)
        owner_rows, cost_rows, mass, gene_index, fixed_owner, null_owner = [], [], [], [], [], []

        def append_row(amount, gene, owner, partner=None, null_target=None):
            if amount <= 0:
                return
            owner_rows.append([owner] if partner is None else [owner, partner])
            cost_rows.append([0.0] if partner is None else [0.0, 1.0])
            mass.append(int(amount))
            gene_index.append(gene)
            fixed_owner.append(owner)
            null_owner.append(owner if null_target is None else null_target)

        for cell in range(len(block_ids)):
            partner = cell ^ 1
            for gene in range(len(genes)):
                count = int(block_observed[cell, gene])
                movable = int(np.floor(ambiguity * count))
                transfer = max(0, int(block_observed[cell, gene] - block_null[cell, gene]))
                if movable < transfer:
                    raise ValueError("ambiguity is too small to represent the requested null ownership; "
                                     "reduce state_strength or increase ambiguity")
                append_row(count - movable, gene, cell)
                append_row(transfer, gene, cell, partner, partner)
                append_row(movable - transfer, gene, cell, partner)
        support, costs = _csr_from_rows(owner_rows, cost_rows, len(block_ids) + 1)
        fixed_owner, null_owner = np.asarray(fixed_owner, dtype=np.int64), np.asarray(null_owner, dtype=np.int64)
        mass = np.asarray(mass, dtype=float)
        null_cost = float(mass[fixed_owner != null_owner].sum())
        budget = null_cost if allow_null else 0.0
        labels = BudgetedLabelSet(support, mass, np.asarray(gene_index, dtype=np.int64), len(genes),
                                  edge_cost=costs, max_total_cost=budget, background_index=len(block_ids),
                                  max_background_mass=0.0)
        np.testing.assert_array_equal(labels.fixed_labels(fixed_owner)[:-1], block_observed)
        # Always verify the null's support/labels. With allow_null=False its
        # declared witness must then fail the displacement budget if distinct.
        np.testing.assert_array_equal(labels.fixed_labels(null_owner, require_feasible=False)[:-1], block_null)
        null_feasible = null_cost <= budget
        if null_feasible:
            labels.fixed_labels(null_owner)
        else:
            try:
                labels.fixed_labels(null_owner)
            except ValueError:
                pass
            else:
                raise AssertionError("Expected the zero movement budget to exclude the distinct null")
        np.testing.assert_array_equal(block_observed.sum(axis=1), 80)
        for pair in range(pairs_per_donor):
            np.testing.assert_array_equal(block_observed[2*pair:2*pair+2].sum(axis=0),
                                          block_null[2*pair:2*pair+2].sum(axis=0))
        coefficients = np.vstack([np.log(block_null) - np.log(block_observed), np.zeros((1, len(genes)))])
        relative_state_loss = float((block_observed - block_null).sum() + labels.support_function(coefficients))
        denominator = block_observed.size
        null_kl = float(np.sum(block_observed - block_null + block_null *
                                (np.log(block_null) - np.log(block_observed))))
        if null_feasible and relative_state_loss + 1e-7 < null_kl:
            raise AssertionError("Null witness lower bound does not hold")
        proof_rows.append({"donor_id": donor_id, "null_feasible": bool(null_feasible),
                           "null_assignment_cost": null_cost, "max_total_cost": budget,
                           "baseline_relative_objective": 0.0,
                           "state_relative_objective": relative_state_loss / denominator,
                           "null_witness_kl_lower_bound": null_kl / denominator,
                           "active_pairs": active_count if shift else 0})
        blocks.append(GraphBlock(slide_id, donor_id, block_ids, genes, labels, fixed_owner,
                                 {"source": "constructive_state_erasure_counterexample",
                                  "null_feasible": bool(null_feasible), "allow_null": bool(allow_null),
                                  "true_state_assignment": "fixed_owner",
                                  "null_assignment_witness": f"truth.npz:null_owner_block{donor:05d}",
                                  "null_assignment_cost": null_cost, "coordinate_units": "synthetic_unit"}))
        witness_arrays[f"null_owner_block{donor:05d}"] = null_owner
        witness_arrays[f"fixed_owner_block{donor:05d}"] = fixed_owner
        witness_arrays[f"cell_rows_block{donor:05d}"] = np.arange(global_start, global_start + len(block_ids))
        global_start += len(block_ids)
    observed, baseline_counts = np.asarray(observed_rows), np.asarray(null_rows)
    dataset = Dataset(sparse.csr_matrix(observed), np.asarray(feature_rows, dtype=np.float32),
                      np.asarray(covariate_rows, dtype=np.float32), pd.DataFrame(cells), genes,
                      {"dataset_id": "constructive-state-erasure", "feature_source": "synthetic",
                       "covariate_source": "synthetic", "counterexample": True,
                       "covariate_columns": ["type_A", "type_B"],
                       "feature_note": "Latent image state signal is explicit; no state labels enter covariates.",
                       "observed_counts_are_synthetic_true_counts": True})
    save_dataset(dataset, output / "data")
    save_graphs(blocks, output / "graphs")
    np.savez_compressed(output / "truth.npz", cell_ids=np.asarray(dataset.cells.cell_id, dtype=str),
                        gene_ids=np.asarray(genes), true_counts=observed, observed_counts=observed,
                        null_counts=baseline_counts, true_rates=observed.astype(float),
                        null_rates=baseline_counts.astype(float), rates=observed.astype(float),
                        latent_state_score=np.asarray(latent_scores), pair_ids=np.asarray(pair_ids),
                        **witness_arrays)
    pd.DataFrame(state_labels).to_csv(output / "state_labels.csv", index=False)
    (output / "programs.json").write_text(json.dumps({"positive_state": {"STATE_UP": 1, "STATE_DOWN": -1}},
                                                    indent=2) + "\n", encoding="utf-8")
    proof = {"seed": seed, "donors": donors, "pairs_per_donor": pairs_per_donor,
             "requested_strength": state_strength, "integer_state_shift": shift,
             "requested_active_pair_prevalence": state_prevalence,
             "realized_active_pair_prevalence": active_count / pairs_per_donor if shift else 0,
             "realized_positive_cell_prevalence": float(np.mean(np.asarray(latent_scores) > 0)),
             "ambiguity_fraction": ambiguity, "allow_null": bool(allow_null), "blocks": proof_rows,
             "statement": "For any candidate mean mu, if null counts b belong to the feasible label set, "
                          "max_Y[L(mu,Y)-L(b,Y)] >= KL_Poisson(b||mu) >= 0. At mu=b the objective is zero. "
                          "A distinct true state therefore cannot be preferred by this fixed finite-label empirical objective.",
             "limitations": "Constructed fixed-count counterexample. This is not evidence about actual biology, "
                            "not an optimization guarantee for a finite neural training run, and not a novel theorem."}
    (output / "proof.json").write_text(json.dumps(proof, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return proof
