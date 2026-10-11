"""Generated from authoritative main-project pure helpers; do not edit here."""
from __future__ import annotations
import csv
import itertools
import json
from pathlib import Path
from .v4_evaluators import FEATURES, TARGETS, PACKAGE, design_id, digest, validate_design

DEFAULT = dict(zip(FEATURES, (4., 10., 3., 4.)))


def make_plan(*, method="grid", axes=None, fixed=None, count=None, seed=0, csv_path=None, designs=None):
    """grid axes support one/two/four dimensions. random/LHS sample discrete legal length."""
    axes, fixed = axes or {}, dict(DEFAULT if fixed is None else fixed)
    if set(axes) - set(FEATURES) or set(fixed) - set(FEATURES):
        raise ValueError("Unknown design axis")
    if set(axes) & set(fixed):
        raise ValueError("Varying axes must not also be fixed")
    domain = json.loads((PACKAGE / "configs/domain.json").read_text())
    if method == "grid":
        if len(axes) not in (1, 2, 4) or set(axes) | set(fixed) != set(FEATURES):
            raise ValueError("Grid requires one/two/four varying axes and explicit other fixed values")
        if any(not isinstance(v, list) or not v for v in axes.values()):
            raise ValueError("Nonempty axis lists required")
        names = list(axes)
        points = [dict(fixed, **dict(zip(names, values))) for values in itertools.product(*(axes[k] for k in names))]
    elif method in ("random", "lhs"):
        import numpy as np
        from scipy.stats import qmc
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("Positive integer sample count required")
        if axes or fixed:
            raise ValueError("4D random/LHS requires fixed={} and no axes; length samples legal discrete levels")
        u = np.random.default_rng(seed).random((count, 4)) if method == "random" else qmc.LatinHypercube(4, seed=seed).random(count)
        bounds = domain["bounds"]; points = []
        length_levels = [round(1 + k * .1, 10) for k in range(71)]
        for row in u:
            d = {k: bounds[k][0] + row[j] * (bounds[k][1] - bounds[k][0]) for j, k in enumerate(FEATURES[:3])}
            d["gain_length_m"] = length_levels[min(len(length_levels)-1, int(row[3]*len(length_levels)))]; points.append(d)
    elif method == "csv":
        if csv_path is None:
            raise ValueError("CSV path required")
        with Path(csv_path).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if any(any(row.get(k, "").strip() for k in TARGETS) for row in rows):
            raise ValueError("Candidate import must be X-only, never prelabel unknown solver outputs")
        points = [{k: float(row[k]) for k in FEATURES} for row in rows]
        axes, fixed = {}, {}
    elif method == "candidates":
        if designs is None:
            raise ValueError("Explicit candidate input list required")
        points = list(designs); axes, fixed = {}, {}
    else:
        raise ValueError("Supported plan methods: grid, random, lhs, csv, candidates")
    unique, invalid = {}, []
    for index, value in enumerate(points):
        try:
            d = validate_design(value); unique.setdefault(design_id(d), d)
        except (ValueError, TypeError) as exc:
            invalid.append({"input_index": index, "reason": str(exc), "submitted_repr": repr(value)})
    plan = {"schema_version": "mamyshev_scan_v4", "method": method, "axes": axes, "fixed": fixed,
            "seed": seed, "designs": [{"design_id": k, "design_config": v} for k, v in unique.items()],
            "input_count": len(points), "unique_designs": len(unique), "duplicates_removed": len(points)-len(invalid)-len(unique),
            "invalid": invalid, "domain_status": domain["status"], "default_action": "plan_only",
            "max_new_real_attempts": 0, "length_sampling": "explicit legal 0.1 m levels; no submitted length rounding"}
    plan["plan_hash"] = digest(plan)
    return plan


def verify_plan(plan):
    if plan.get("plan_hash") != digest({k: v for k, v in plan.items() if k != "plan_hash"}):
        raise ValueError("Plan hash mismatch")
    if plan["invalid"]:
        raise ValueError("Plan contains invalid inputs; execute refused")
    ids = []
    for row in plan["designs"]:
        if row["design_id"] != design_id(row["design_config"]):
            raise ValueError("Plan design identity mismatch")
        ids.append(row["design_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate designs in executable plan")
