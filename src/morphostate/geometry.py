"""Auditable molecule candidate graphs and non-pickle artifact interchange.

Centroid-radius candidates are an explicit geometric proxy, not cell membrane
segmentation. Coordinates must already be registered and use the same units.
No gene aliases, slide aliases, registration or ownership correction is inferred.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

from .budgeted import BudgetedLabelSet
from .data import Dataset, load_dataset, validate_dataset


@dataclass
class GraphBlock:
    block_id: str
    donor_id: str
    cell_ids: list[str]
    gene_ids: list[str]
    label_set: BudgetedLabelSet
    fixed_owner: np.ndarray
    metadata: dict = field(default_factory=dict)

    def validate(self):
        for name, values in (("cell_ids", self.cell_ids), ("gene_ids", self.gene_ids)):
            if (not values or len(set(values)) != len(values)
                    or any(not isinstance(v, str) or not v.strip() or v != v.strip() for v in values)):
                raise ValueError(f"{name} must be nonempty, unique, trimmed strings")
        if any(not isinstance(v, str) or not v.strip() for v in (self.block_id, self.donor_id)):
            raise ValueError("block_id and donor_id must be nonempty strings")
        if (self.label_set.n_cells != len(self.cell_ids) + 1
                or self.label_set.background_index != len(self.cell_ids)
                or self.label_set.n_genes != len(self.gene_ids)):
            raise ValueError("Graph axes must match real cells plus a final background column and genes")
        self.label_set.fixed_labels(self.fixed_owner)
        if not isinstance(self.metadata, dict):
            raise ValueError("Graph metadata must be a dictionary")
        json.dumps(self.metadata, allow_nan=False)


def _validate_blocks(blocks):
    if not blocks:
        raise ValueError("At least one graph block is required")
    seen_cells, seen_blocks = set(), set()
    for block in blocks:
        block.validate()
        if block.block_id in seen_blocks or seen_cells.intersection(block.cell_ids):
            raise ValueError("Graph blocks must have unique IDs and disjoint real cells")
        seen_cells.update(block.cell_ids)
        seen_blocks.add(block.block_id)


def save_graphs(blocks: list[GraphBlock], path):
    """Save validated arrays without pickle; refuse to overwrite any artifact."""
    _validate_blocks(blocks)
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Graph output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    entries = []
    for number, block in enumerate(blocks):
        labels = block.label_set
        filename = f"block{number:05d}.npz"
        np.savez_compressed(path / filename, indptr=labels.candidate_owners.indptr,
                            indices=labels.candidate_owners.indices, mass=labels.mass,
                            gene_index=labels.gene_index, edge_cost=labels.edge_cost,
                            fixed_owner=block.fixed_owner, cell_ids=np.asarray(block.cell_ids, dtype=str),
                            gene_ids=np.asarray(block.gene_ids, dtype=str))
        entries.append({"file": filename, "block_id": block.block_id, "donor_id": block.donor_id,
                        "max_total_cost": labels.max_total_cost,
                        "max_background_mass": labels.max_background_mass,
                        "max_edges": labels.max_edges, "max_label_entries": labels.max_label_entries,
                        "metadata": block.metadata})
    (path / "manifest.json").write_text(json.dumps({"schema_version": 1, "blocks": entries},
                                                  indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                                        encoding="utf-8")


def load_graphs(path) -> list[GraphBlock]:
    """Read graph artifact and revalidate support, budgets, axes and ownership."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("blocks"), list):
        raise ValueError("Unsupported graph manifest schema")
    blocks = []
    for entry in manifest["blocks"]:
        filename = entry["file"]
        if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".npz"):
            raise ValueError("Graph file must be a local NPZ basename")
        with np.load(path / filename, allow_pickle=False) as arrays:
            cells, genes = arrays["cell_ids"].tolist(), arrays["gene_ids"].tolist()
            mass = arrays["mass"].copy()
            graph = sparse.csr_matrix((np.ones(len(arrays["indices"]), dtype=bool),
                                       arrays["indices"].copy(), arrays["indptr"].copy()),
                                      shape=(len(mass), len(cells) + 1))
            labels = BudgetedLabelSet(graph, mass, arrays["gene_index"], len(genes),
                                      edge_cost=arrays["edge_cost"], background_index=len(cells),
                                      max_total_cost=entry.get("max_total_cost"),
                                      max_background_mass=entry.get("max_background_mass"),
                                      max_edges=entry.get("max_edges", 2_000_000),
                                      max_label_entries=entry.get("max_label_entries", 10_000_000))
            blocks.append(GraphBlock(entry["block_id"], entry["donor_id"], cells, genes, labels,
                                     arrays["fixed_owner"].copy(), entry.get("metadata", {})))
    _validate_blocks(blocks)
    return blocks


def without_budgets(block: GraphBlock, *, geometry=True, background=True) -> GraphBlock:
    """Copy a graph, explicitly removing one or both shared budget constraints."""
    block.validate()
    old = block.label_set
    labels = old.copy_with_budgets(max_total_cost=None if geometry else old.max_total_cost,
                                  max_background_mass=None if background else old.max_background_mass)
    metadata = dict(block.metadata, removed_geometry_budget=bool(geometry),
                    removed_background_budget=bool(background))
    return GraphBlock(block.block_id, block.donor_id, list(block.cell_ids), list(block.gene_ids),
                      labels, block.fixed_owner.copy(), metadata)


def subset_genes(block: GraphBlock, gene_ids: list[str]) -> GraphBlock:
    """Select genes, conditioning removed molecules on their observed owners.

    Removed molecules consume their observed geometry/background mass budgets.
    This is an explicit conditional feasible set, NOT an exact projection that
    optimizes the unselected molecules. Use all genes to avoid this conditioning.
    """
    block.validate()
    if not gene_ids or len(set(gene_ids)) != len(gene_ids) or not set(gene_ids) <= set(block.gene_ids):
        raise ValueError("gene_ids must be a nonempty unique subset of graph genes")
    old = block.label_set
    index = {gene: position for position, gene in enumerate(gene_ids)}
    mapping = np.array([index.get(gene, -1) for gene in block.gene_ids])
    keep = mapping[old.gene_index] >= 0
    if not keep.any():
        raise ValueError("Selected genes have no observations in this block")
    edge_keep = keep[old._edge_rows]
    observed_edges = old.candidate_owners.indices == block.fixed_owner[old._edge_rows]
    removed_edges = (~edge_keep) & observed_edges
    removed_cost = float(np.dot(old.edge_cost[removed_edges], old.mass[old._edge_rows[removed_edges]]))
    removed_background = float(old.mass[(~keep) & (block.fixed_owner == old.background_index)].sum())
    max_cost = None if old.max_total_cost is None else max(0.0, old.max_total_cost - removed_cost)
    max_bg = None if old.max_background_mass is None else max(0.0, old.max_background_mass - removed_background)
    labels = BudgetedLabelSet(old.candidate_owners[keep].tocsr(), old.mass[keep],
                              mapping[old.gene_index[keep]], len(gene_ids), edge_cost=old.edge_cost[edge_keep],
                              background_index=old.background_index, max_total_cost=max_cost,
                              max_background_mass=max_bg, max_edges=old.max_edges,
                              max_label_entries=old.max_label_entries)
    result = GraphBlock(block.block_id, block.donor_id, list(block.cell_ids), list(gene_ids), labels,
                        block.fixed_owner[keep], dict(block.metadata,
                        gene_subset_policy="condition_removed_molecules_on_observed_owners",
                        removed_observed_cost=removed_cost, removed_observed_background_mass=removed_background))
    result.validate()
    return result


def _csr_from_rows(owner_rows, cost_rows, n_cells):
    indices, costs, indptr = [], [], [0]
    for owners, row_costs in zip(owner_rows, cost_rows):
        order = np.argsort(owners)
        indices.extend(np.asarray(owners)[order].tolist())
        costs.extend(np.asarray(row_costs)[order].tolist())
        indptr.append(len(indices))
    matrix = sparse.csr_matrix((np.ones(len(indices), dtype=bool), np.asarray(indices), np.asarray(indptr)),
                               shape=(len(owner_rows), n_cells))
    return matrix, np.asarray(costs, dtype=np.float64)


def _block_frames(dataset):
    # Dataset alignment is positional; arbitrary DataFrame index labels are not
    # count-matrix row numbers.
    for slide_id, frame in dataset.cells.reset_index(drop=True).groupby("slide_id", sort=True):
        donors = frame["donor_id"].unique()
        if len(donors) != 1:
            raise ValueError(f"Slide {slide_id} contains multiple donors; explicitly separate donor coordinates")
        yield str(slide_id), str(donors[0]), frame


def construct_geometry_graphs(dataset_path, transcript_csv, output, radius, geometry_budget=None,
                              background_fraction=None, *, coordinate_units=None,
                              max_observations=2_000_000, max_edges=2_000_000,
                              max_cells_per_block=5_000, max_label_entries=10_000_000):
    """Build one patient-pure graph per slide and save it.

    Input columns: slide_id,x,y,gene_id,cell_id; mass is optional (default 1).
    Optional is_core (true/false or 1/0) asserts independently annotated certain
    ownership and restricts that row to its original real cell. It must not be
    inferred from the centroid radius itself.
    Empty cell_id denotes background. Other IDs must match source_cell_id when
    available, otherwise prepared cell_id exactly. ``radius`` and both tables'
    coordinates share caller-declared units. ``geometry_budget`` is an absolute
    per-slide sum of mass*(distance/radius)^2, not a fraction. Background edges
    have cost zero and are separately capped by ``background_fraction`` when set.
    The original owner must be within the radius. Prepared counts must exactly
    equal the original known-gene ownership table for all real cells.
    """
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive and finite")
    if not isinstance(coordinate_units, str) or not coordinate_units.strip():
        raise ValueError("Declare coordinate_units after registering both coordinate tables")
    if background_fraction is not None and (not np.isfinite(background_fraction)
                                             or not 0 <= background_fraction <= 1):
        raise ValueError("background_fraction must lie in [0,1]")
    if not isinstance(max_observations, int) or max_observations < 1:
        raise ValueError("max_observations must be positive")
    dataset = load_dataset(dataset_path)
    if not {"x_centroid", "y_centroid"} <= set(dataset.cells):
        raise ValueError("Prepared cells need registered x_centroid and y_centroid columns")
    table = pd.read_csv(transcript_csv, dtype={"slide_id": str, "gene_id": str, "cell_id": str},
                        keep_default_na=False, nrows=max_observations + 1)
    if len(table) > max_observations:
        raise ValueError("Transcript file exceeds max_observations; create an explicit smaller pilot")
    required = {"slide_id", "x", "y", "gene_id", "cell_id"}
    if not required <= set(table):
        raise ValueError(f"Transcript table lacks columns {sorted(required - set(table))}")
    if table.empty:
        raise ValueError("Transcript table is empty")
    if not set(table.slide_id) <= set(dataset.cells.slide_id):
        raise ValueError("Transcript table contains unknown slide IDs")
    for column in ("x", "y"):
        table[column] = pd.to_numeric(table[column], errors="raise")
    if not np.isfinite(table[["x", "y"]].to_numpy(dtype=float)).all():
        raise ValueError("Transcript coordinates must be finite")
    table["mass"] = pd.to_numeric(table["mass"], errors="raise") if "mass" in table else 1.0
    if not np.isfinite(table["mass"]).all() or (table["mass"] <= 0).any():
        raise ValueError("Transcript mass must be finite and strictly positive")
    if "is_core" in table:
        core_text = table["is_core"].astype(str).str.lower()
        if not core_text.isin(["true", "false", "1", "0"]).all():
            raise ValueError("is_core must explicitly contain true/false or 1/0")
        table["is_core"] = core_text.isin(["true", "1"])
    else:
        table["is_core"] = False
    gene_lookup = {gene: index for index, gene in enumerate(dataset.genes)}
    blocks = []
    for slide_id, donor_id, frame in _block_frames(dataset):
        if len(frame) > max_cells_per_block:
            raise ValueError(f"Slide {slide_id} exceeds max_cells_per_block; define a smaller pilot")
        source_ids = frame["source_cell_id"] if "source_cell_id" in frame else frame["cell_id"]
        if source_ids.duplicated().any() or source_ids.eq("").any():
            raise ValueError("source_cell_id must be nonempty and unique within each slide")
        lookup = {source: index for index, source in enumerate(source_ids)}
        positions = frame[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
        if not np.isfinite(positions).all():
            raise ValueError("Cell centroid coordinates must be finite")
        slide_table = table.loc[table.slide_id == slide_id]
        unknown = ~slide_table.gene_id.isin(gene_lookup)
        filtered = slide_table.loc[~unknown]
        if filtered.empty:
            raise ValueError(f"Slide {slide_id} has no known-gene transcripts")
        tree = cKDTree(positions)
        owner_rows, cost_rows, fixed = [], [], []
        edge_count = 0
        for row in filtered.itertuples(index=False):
            location = np.array([row.x, row.y], dtype=float)
            owners = tree.query_ball_point(location, float(radius))
            original = len(frame) if row.cell_id == "" else lookup.get(row.cell_id)
            if original is None:
                raise ValueError(f"Unknown original cell_id {row.cell_id!r} on slide {slide_id}")
            if original != len(frame) and original not in owners:
                raise ValueError(f"Original owner {row.cell_id!r} is outside radius on slide {slide_id}; "
                                 "verify registration or explicitly enlarge radius")
            costs = ((positions[owners] - location) ** 2).sum(axis=1) / radius ** 2
            if row.is_core:
                if original == len(frame):
                    raise ValueError("An is_core transcript must have an original real-cell owner")
                costs = np.array([costs[owners.index(original)]])
                owners = [original]
            else:
                owners.append(len(frame))
                costs = np.append(costs, 0)
            owner_rows.append(owners)
            cost_rows.append(costs)
            fixed.append(original)
            edge_count += len(owners)
            if edge_count > max_edges:
                raise ValueError("Candidate graph exceeds max_edges; reduce the explicit pilot")
        support, costs = _csr_from_rows(owner_rows, cost_rows, len(frame) + 1)
        masses = filtered["mass"].to_numpy(dtype=float)
        gene_index = np.array([gene_lookup[gene] for gene in filtered.gene_id], dtype=np.int64)
        labels = BudgetedLabelSet(support, masses, gene_index, len(dataset.genes), edge_cost=costs,
                                  max_total_cost=geometry_budget, background_index=len(frame),
                                  max_background_mass=None if background_fraction is None else
                                  float(background_fraction * masses.sum()), max_edges=max_edges,
                                  max_label_entries=max_label_entries)
        fixed = np.asarray(fixed, dtype=np.int64)
        observed = labels.fixed_labels(fixed)
        expected = dataset.counts[frame.index.to_numpy()].toarray()
        if not np.array_equal(observed[:-1], expected):
            discrepancy = float(np.abs(observed[:-1] - expected).sum())
            raise ValueError(f"Fixed transcript labels disagree with prepared counts on {slide_id} "
                             f"(absolute discrepancy={discrepancy:g}); synchronize filtering and IDs")
        blocks.append(GraphBlock(slide_id, donor_id, frame.cell_id.tolist(), list(dataset.genes), labels,
                                 fixed, {"source": "registered_centroid_radius_proxy",
                                         "radius": float(radius), "coordinate_units": coordinate_units,
                                         "radius_rule": "original real owner must already be within radius",
                                         "unknown_gene_rows_filtered": int(unknown.sum()),
                                         "unknown_gene_mass_filtered": float(slide_table.loc[unknown, "mass"].sum()),
                                         "known_gene_rows": len(filtered),
                                         "contains_independent_nuclear_core_annotation": bool(filtered.is_core.any()),
                                         "core_annotation_assumed_correct": bool(filtered.is_core.any()),
                                         "dataset_id": dataset.provenance["dataset_id"]}))
    save_graphs(blocks, output)
    return blocks


def make_synthetic_graphs(dataset_path, output=None, *, seed=7, ambiguous_fraction=0.6,
                           background_fraction=0.1, geometry_fraction=0.25,
                           max_edges=2_000_000) -> list[GraphBlock]:
    """Create declared synthetic ownership ambiguity while preserving counts.

    Each nonzero cell/gene count is one divisible-mass observation, with identical
    candidate owners for its aggregate. Owner-only rows are synthetic certainty,
    not inferred from morphology. Budgets are stress-test knobs, not estimates.
    """
    if any(not np.isfinite(v) or not 0 <= v <= 1 for v in
           (ambiguous_fraction, background_fraction, geometry_fraction)):
        raise ValueError("Synthetic ambiguity and budget fractions must lie in [0,1]")
    dataset = dataset_path if isinstance(dataset_path, Dataset) else load_dataset(dataset_path)
    validate_dataset(dataset)
    rng = np.random.default_rng(seed)
    blocks = []
    for slide_id, donor_id, frame in _block_frames(dataset):
        counts = dataset.counts[frame.index.to_numpy()].tocoo()
        counts.sum_duplicates()
        nonzero = counts.data > 0
        original_cells, original_genes, original_mass = counts.row[nonzero], counts.col[nonzero], counts.data[nonzero]
        cell_indices, genes, mass = [], [], []
        owner_rows, cost_rows = [], []
        edge_count = 0
        for cell, gene, count in zip(original_cells, original_genes, original_mass):
            ambiguous_mass = int(rng.binomial(int(count), ambiguous_fraction)) if len(frame) > 1 else 0
            certain_mass = int(count) - ambiguous_mass
            if certain_mass:
                owner_rows.append([int(cell)])
                cost_rows.append([0.0])
                cell_indices.append(cell)
                genes.append(gene)
                mass.append(certain_mass)
                edge_count += 1
            if ambiguous_mass:
                neighbor = int(rng.integers(len(frame) - 1))
                neighbor += neighbor >= cell
                owner_rows.append([int(cell), neighbor, len(frame)])
                cost_rows.append([0.0, 1.0, 0.0])
                cell_indices.append(cell)
                genes.append(gene)
                mass.append(ambiguous_mass)
                edge_count += 3
            if edge_count > max_edges:
                raise ValueError("Synthetic candidate graph exceeds max_edges")
        cell_indices = np.asarray(cell_indices, dtype=np.int64)
        genes, mass = np.asarray(genes, dtype=np.int64), np.asarray(mass, dtype=float)
        support, costs = _csr_from_rows(owner_rows, cost_rows, len(frame) + 1)
        labels = BudgetedLabelSet(support, mass, genes.astype(np.int64), len(dataset.genes),
                                  edge_cost=costs, background_index=len(frame),
                                  max_total_cost=float(geometry_fraction * mass.sum()),
                                  max_background_mass=float(background_fraction * mass.sum()), max_edges=max_edges)
        blocks.append(GraphBlock(slide_id, donor_id, frame.cell_id.tolist(), list(dataset.genes), labels,
                                 cell_indices.astype(np.int64),
                                 {"source": "synthetic", "seed": int(seed),
                                  "ambiguity_is_simulated": True, "ambiguous_fraction": float(ambiguous_fraction),
                                  "dataset_id": dataset.provenance["dataset_id"]}))
    _validate_blocks(blocks)
    if output is not None:
        save_graphs(blocks, output)
    return blocks
