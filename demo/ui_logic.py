"""Pure prediction UI plumbing, reusing the existing plan constructor."""
from __future__ import annotations
import csv
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import numpy as np
from mamyshev_ml.v4_evaluators import FEATURES, TARGETS, validate_design, canonical
from mamyshev_ml.v4_scan import make_plan, verify_plan
from mamyshev_ml.demo_inference import require_current_model

ASSETS = Path(__file__).resolve().parent / "assets"
DEFAULT = dict(zip(FEATURES, (4., 10., 3., 4.)))
LABELS = dict(zip(FEATURES, ("Pump power (W)", "SF1 − SF2 offset (nm)", "SF2 bandwidth (nm)", "Gain length (m)")))
TARGET_LABELS = dict(zip(TARGETS, ("Compressed energy (nJ)", "Compressed duration (fs)", "Compressed peak power (MW)")))
INPUT_UNITS = dict(zip(FEATURES, ("W", "nm", "nm", "m")))
UNITS = dict(zip(TARGETS, ("nJ", "fs", "MW")))
DOMAIN = json.loads((Path(__file__).resolve().parents[1] / "mamyshev_accuracy_v4/configs/domain.json").read_text())
BOUNDS = DOMAIN["bounds"]


def read_summary(name):
    if Path(name).name != name:
        raise ValueError("Fixed summary name required")
    return json.loads((ASSETS / name).read_text())


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _ranges(ranges):
    """Validate submitted endpoints; never clip or round a user's length."""
    if ranges is None:
        ranges = {}
    if not isinstance(ranges, dict) or set(ranges) - set(FEATURES):
        raise ValueError("Ranges must name only the four design inputs")
    result = {name: list(limits) for name, limits in BOUNDS.items()}
    for name, pair in ranges.items():
        if not isinstance(pair, (list, tuple)) or len(pair) != 2 or not all(_finite(x) for x in pair):
            raise ValueError("Each range needs two finite endpoints")
        lo, hi = pair
        if not BOUNDS[name][0] <= lo <= hi <= BOUNDS[name][1]:
            raise ValueError(name + ": range outside exploratory domain or reversed")
        if name == "gain_length_m":
            for value in pair:
                validate_design({**DEFAULT, name: value})
        result[name] = [float(lo), float(hi)]
    return result


def _length_levels(lo, hi):
    # Endpoints have already been validated as legal levels. This enumerates
    # their grid; it does not repair an arbitrary user-submitted length.
    step = DOMAIN["gain_length_step_m"]
    first, last = int(round(lo / step)), int(round(hi / step))
    return [round(index * step, 10) for index in range(first, last + 1)]


def _validate_scenario(scenario):
    if not isinstance(scenario, dict):
        raise ValueError("Scenario must be an explicit object")
    objective = scenario.get("objective", {})
    if (set(objective) != {"target", "direction"} or objective["target"] not in TARGETS
            or objective["direction"] not in ("maximize", "minimize")):
        raise ValueError("Known target and maximize/minimize objective required")
    fixed = scenario.get("fixed_inputs", {})
    if not isinstance(fixed, dict) or set(fixed) - set(FEATURES):
        raise ValueError("Unknown scenario fixed input")
    validate_design({**DEFAULT, **fixed})
    for group, names in (("input_constraints", FEATURES), ("output_constraints", TARGETS)):
        constraints = scenario.get(group, {})
        if not isinstance(constraints, dict) or set(constraints) - set(names):
            raise ValueError("Unknown constraint field")
        for limits in constraints.values():
            if (not isinstance(limits, dict) or not limits or set(limits) - {"lte", "gte"}
                    or not all(_finite(x) for x in limits.values())):
                raise ValueError("Finite lte/gte constraint boundaries required")
            if "gte" in limits and "lte" in limits and limits["gte"] > limits["lte"]:
                raise ValueError("Contradictory constraint bounds")


def scan_plan(fixed, axis1, axis2=None, count=9, ranges=None, counts=None):
    fixed = validate_design(fixed)
    if isinstance(count, bool) or not isinstance(count, int) or not 2 <= count <= 21:
        raise ValueError("Each scan axis must have 2–21 points")
    names = [axis1] if axis2 is None else [axis1, axis2]
    if any(n not in FEATURES for n in names) or len(set(names)) != len(names):
        raise ValueError("Select distinct known axes")
    limits = _ranges(ranges)
    if counts is None:
        counts = {}
    if not isinstance(counts, dict) or set(counts) - set(names):
        raise ValueError("Counts must name the selected axes only")
    if ranges is not None and set(ranges) - set(names):
        raise ValueError("Scan ranges must name the selected axes only")
    axes = {}
    for name in names:
        n = counts.get(name, count)
        if isinstance(n, bool) or not isinstance(n, int) or not 2 <= n <= 21:
            raise ValueError("Each scan axis must have 2–21 points")
        lo, hi = limits[name]
        if lo >= hi:
            raise ValueError("Scan axis endpoints must be increasing")
        if name == "gain_length_m":
            levels = _length_levels(lo, hi)
            if n > len(levels):
                raise ValueError("Requested length points exceed distinct legal 0.1 m levels")
            values = [levels[i * (len(levels) - 1) // (n - 1)] for i in range(n)]
        else:
            values = np.linspace(lo, hi, n).tolist()
        if len(set(values)) != n:
            raise ValueError("Scan axis must contain distinct values")
        axes[name] = values
    plan = make_plan(axes=axes, fixed={k: v for k, v in fixed.items() if k not in names})
    verify_plan(plan)
    if plan["unique_designs"] > 441:
        raise ValueError("UI scan budget exceeded")
    return plan


def _satisfies(values, constraints):
    for key, limits in constraints.items():
        for op, boundary in limits.items():
            if op not in ("lte", "gte"):
                raise ValueError("Unknown comparison")
            if not (values[key] <= boundary if op == "lte" else values[key] >= boundary):
                return False
    return True


def candidate_plan(scenario, count, max_pump, extra_fixed, seed=72001, ranges=None):
    """X-only sampling using the existing seeded sampler and canonical plans.

    Continuous samples map affinely into submitted ranges. Length samples select
    legal levels from the existing sampler's discrete ranks; no design is rounded.
    Fixed inputs and constraints can reduce the final unique candidate count.
    """
    _validate_scenario(scenario)
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 4096:
        raise ValueError("Candidate limit is 4096")
    if isinstance(max_pump, bool) or not isinstance(max_pump, (int, float)) or not math.isfinite(max_pump) or not 2 <= max_pump <= 10:
        raise ValueError("Pump cap must be finite and inside [2, 10] W")
    if not isinstance(extra_fixed, dict) or set(extra_fixed) - set(FEATURES):
        raise ValueError("Only four named fixed inputs are accepted")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    validate_design({**DEFAULT, **extra_fixed})
    fixed = dict(scenario.get("fixed_inputs", {}))
    if any(k in fixed and fixed[k] != v for k, v in extra_fixed.items()):
        raise ValueError("Extra fixed value conflicts with the scenario")
    fixed.update(extra_fixed)
    limits = _ranges(ranges)
    if any(not limits[k][0] <= value <= limits[k][1] for k, value in fixed.items()):
        raise ValueError("A fixed input is outside the submitted candidate range")
    plan = make_plan(method="random", fixed={}, count=count, seed=seed)
    rows = []
    original_levels = _length_levels(*BOUNDS["gain_length_m"])
    selected_levels = _length_levels(*limits["gain_length_m"])
    for item in plan["designs"]:
        d = dict(item["design_config"])
        for name in FEATURES[:3]:
            lo, hi = limits[name]
            domain_lo, domain_hi = BOUNDS[name]
            if (lo, hi) != (domain_lo, domain_hi):
                d[name] = lo + (d[name] - domain_lo) / (domain_hi - domain_lo) * (hi - lo)
        index = original_levels.index(d["gain_length_m"])
        d["gain_length_m"] = selected_levels[index * len(selected_levels) // len(original_levels)]
        d.update(fixed)
        if d["pump_power_W"] <= max_pump and _satisfies(d, scenario.get("input_constraints", {})):
            rows.append(d)
    if not rows:
        raise ValueError("No candidates satisfy the submitted input constraints")
    plan = make_plan(method="candidates", designs=rows, seed=seed)
    verify_plan(plan)
    return plan


def rank_predictions(result, scenario):
    _validate_scenario(scenario)
    feasible = [r for r in result["predictions"] if all(item["point_pass"] for item in constraint_rows(r, scenario))]
    target = scenario["objective"]["target"]
    sign = -1 if scenario["objective"]["direction"] == "maximize" else 1
    return sorted(feasible, key=lambda r: (sign * r["targets"][target], canonical(r["design"])))


def pending_requests(result, ranked, scenario):
    """Export at most three actual selected predictions, never evaluated targets."""
    require_current_model(result.get("model_id"), result.get("model_sha256"))
    if not 1 <= len(ranked) <= 3:
        raise ValueError("Select one to three candidates for a pending request")
    known = {canonical(row) for row in result["predictions"]}
    if len({canonical(row["design"]) for row in ranked}) != len(ranked):
        raise ValueError("Selected candidates must be distinct")
    for row in ranked:
        if canonical(row) not in known or row.get("kind", "Prediction") != "Prediction":
            raise ValueError("Selected row is not an unchanged prediction from this model result")
        validate_design(row["design"])
        if set(row["targets"]) != set(TARGETS) or not all(_finite(x) and x > 0 for x in row["targets"].values()):
            raise ValueError("Three finite positive predictions required")
    if isinstance(scenario, dict) and any(not all(c["point_pass"] for c in constraint_rows(row, scenario)) for row in ranked):
        raise ValueError("Selected candidates must satisfy the submitted point-prediction constraints")
    return {"schema_version": "mamyshev_demo_pending_v1", "status": "pending_manual_approval",
        "source_model_id": result["model_id"], "source_model_sha256": result["model_sha256"],
        "source_kind": "surrogate_prediction", "scenario": copy.deepcopy(scenario), "max_new_real_attempts": 0,
        "input_units": INPUT_UNITS, "prediction_units": UNITS,
        "requests": [{"design_config": dict(r["design"]), "prediction": dict(r["targets"]),
            "prediction_status": "predicted_not_verified", "real_targets": None,
            "evaluation_id": None, "status": "not_run"} for r in ranked]}


def constraint_rows(row, scenario):
    """Point decisions and limited-interval diagnostics stay separate."""
    _validate_scenario(scenario)
    values = validate_design(row["design"])
    targets = row["targets"]
    if set(targets) != set(TARGETS) or not all(_finite(x) for x in targets.values()):
        raise ValueError("Constraint evaluation needs three finite predictions")
    rows = []
    for key, boundary in scenario.get("fixed_inputs", {}).items():
        passed = values[key] == boundary
        rows.append({"kind": "fixed_input", "field": key, "label": LABELS[key], "operator": "eq",
            "boundary": boundary, "value": values[key], "point_pass": passed,
            "point_status": "satisfied" if passed else "not_satisfied",
            "interval_status": "not_applicable", "lower": None, "upper": None})
    for kind, group, source, labels in (("input", "input_constraints", values, LABELS),
                                        ("output", "output_constraints", targets, TARGET_LABELS)):
        for key, limits in scenario.get(group, {}).items():
            for operator, boundary in limits.items():
                value = source[key]
                passed = value <= boundary if operator == "lte" else value >= boundary
                low, high, diagnostic = None, None, "not_applicable"
                if kind == "output":
                    low, high = (row.get("lower") or {}).get(key), (row.get("upper") or {}).get(key)
                    if low is None or high is None:
                        diagnostic, low, high = "not_available", None, None
                    else:
                        if not _finite(low) or not _finite(high) or not low <= value <= high:
                            raise ValueError("Diagnostic interval must be finite, ordered and contain its point")
                        all_pass = high <= boundary if operator == "lte" else low >= boundary
                        all_fail = low > boundary if operator == "lte" else high < boundary
                        diagnostic = "within_threshold" if all_pass else "outside_threshold" if all_fail else "crosses_threshold"
                rows.append({"kind": kind, "field": key, "label": labels[key], "operator": operator,
                    "boundary": boundary, "value": value, "point_pass": passed,
                    "point_status": "satisfied" if passed else "not_satisfied",
                    "interval_status": diagnostic, "lower": low, "upper": high})
    return rows


def constraint_status(row, scenario):
    checks = constraint_rows(row, scenario)
    if any(not check["point_pass"] for check in checks):
        return "point_not_satisfied"
    if any(check["interval_status"] == "crosses_threshold" for check in checks):
        return "point_satisfied_interval_crosses_threshold"
    return "point_satisfied" if checks else "no_constraints"


def compare_predictions(baseline_row, candidate_row, objective):
    if objective.get("target") not in TARGETS or objective.get("direction") not in ("maximize", "minimize"):
        raise ValueError("Known comparison objective required")
    result = {}
    for target in TARGETS:
        base, candidate = baseline_row["targets"].get(target), candidate_row["targets"].get(target)
        base = base if _finite(base) and base >= 0 else None
        candidate = candidate if _finite(candidate) and candidate >= 0 else None
        delta = candidate - base if base is not None and candidate is not None else None
        relative = delta / base if delta is not None and base != 0 else None
        direction = objective["direction"] if target == objective["target"] else ("minimize" if target == "compressed_duration_fs" else "maximize")
        sign = 1 if direction == "maximize" else -1
        result[target] = {"baseline": base, "candidate": candidate, "absolute_change": delta,
            "relative_change": relative, "relative_change_percent": None if relative is None else relative * 100,
            "directional_improvement": None if delta is None else sign * delta,
            "directional_improvement_percent": None if relative is None else sign * relative * 100,
            "is_objective": target == objective["target"], "preferred_direction": direction, "unit": UNITS[target]}
    return result


def request_key(model_id, model_sha256, payload):
    """Identity plus the complete canonical submitted controls, not display values."""
    require_current_model(model_id, model_sha256)
    if not isinstance(model_id, str) or not model_id or not isinstance(model_sha256, str) or not model_sha256:
        raise ValueError("Model ID and content identity required")
    return hashlib.sha256(canonical({"model_id": model_id, "model_sha256": model_sha256, "request": payload}).encode()).hexdigest()


def prediction_csv(result):
    require_current_model(result.get("model_id"), result.get("model_sha256"))
    stream = io.StringIO()
    fields = [*FEATURES, *TARGETS, "kind", "model_id", "n_train", "interval_status"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in result["predictions"]:
        writer.writerow({**row["design"], **row["targets"], "kind": "Prediction",
            **{k: result[k] for k in ("model_id", "n_train", "interval_status")}})
    return stream.getvalue()
