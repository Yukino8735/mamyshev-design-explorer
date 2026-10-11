"""Exact Prediction charts: numeric cells, original values and explicit markers.

Cell boundaries are neighboring-coordinate midpoints with half-spacing at the
ends. This is display geometry, not interpolation or additional predictions.
"""
from __future__ import annotations

import math

from ui_logic import (DEFAULT, FEATURES, TARGETS, LABELS, TARGET_LABELS, UNITS,
                      constraint_status, rank_predictions, validate_design)


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def normalize_streamlit_padding(spec):
    """Preserve Vega-Lite margins while avoiding Streamlit 1.64's numeric write.

    Its frontend may assign padding.bottom after JSON.parse. Normalize only the
    top-level property; do not alter layer data or add padding to child charts.
    Missing padding keeps frontend defaults; explicit zero keeps four zero sides.
    """
    result = dict(spec)
    if "padding" not in spec:
        return result
    padding = spec["padding"]
    sides = ("left", "right", "top", "bottom")
    if _finite(padding) and padding >= 0:
        result["padding"] = dict.fromkeys(sides, padding)
    elif (isinstance(padding, dict) and not set(padding) - set(sides)
          and all(_finite(value) and value >= 0 for value in padding.values())):
        result["padding"] = dict(padding)
    else:
        raise ValueError("Padding must be a finite nonnegative number or per-side object")
    return result


def _prepare(result, axes, target, scenario):
    names = list(axes)
    if len(names) not in (1, 2) or len(set(names)) != len(names) or any(name not in FEATURES for name in names):
        raise ValueError("One or two distinct known chart axes required")
    if target not in TARGETS:
        raise ValueError("Known plot target required")
    rows = result["predictions"]
    if not rows or len(rows) > (21 if len(names) == 1 else 441):
        raise ValueError("Chart prediction count is outside the UI budget")
    cells, seen = [], set()
    for row in rows:
        design = validate_design(row["design"])
        if set(row["targets"]) != set(TARGETS) or not all(_finite(x) for x in row["targets"].values()):
            raise ValueError("Nonfinite or missing predictions cannot be painted as zero")
        coordinate = tuple(design[name] for name in names)
        if coordinate in seen:
            raise ValueError("Multiple predictions at one chart coordinate")
        seen.add(coordinate)
        cells.append({**design, **row["targets"], "constraint_status": constraint_status(row, scenario)})
    levels = {}
    for name in names:
        supplied = axes[name] if isinstance(axes, dict) else [row["design"][name] for row in rows]
        if not supplied or not all(_finite(value) for value in supplied):
            raise ValueError("Finite numeric chart coordinates required")
        values = sorted(set(supplied))
        for value in values:
            validate_design({**DEFAULT, name: value})
        if not 2 <= len(values) <= 21:
            raise ValueError("Each chart axis needs 2–21 distinct coordinates")
        if isinstance(axes, dict) and len(values) != len(supplied):
            raise ValueError("Repeated chart axis coordinates")
        if any(row["design"][name] not in values for row in rows):
            raise ValueError("Prediction coordinate is not on the submitted grid")
        levels[name] = values
    if len(names) == 1 and len(rows) != len(levels[names[0]]):
        raise ValueError("Missing 1D predictions: refuse to bridge the gap with a line")
    fixed = {}
    for name in FEATURES:
        if name not in names:
            values = {row["design"][name] for row in rows}
            if len(values) != 1:
                raise ValueError("Non-axis inputs must really be fixed")
            fixed[name] = next(iter(values))
    ranked = rank_predictions(result, scenario)
    return names, levels, fixed, cells, ranked


def _baseline_marker(baseline_row, names, levels, fixed):
    if baseline_row is None:
        return {"status": "omitted", "reason": "No separately predicted baseline supplied"}
    design = validate_design(baseline_row["design"])
    if set(baseline_row["targets"]) != set(TARGETS) or not all(_finite(value) for value in baseline_row["targets"].values()):
        raise ValueError("Baseline marker requires actual finite predictions")
    differing = [name for name, value in fixed.items() if design[name] != value]
    if differing:
        return {"status": "omitted", "reason": "Baseline differs from fixed slice inputs: " + ", ".join(differing)}
    outside = [name for name in names if not levels[name][0] <= design[name] <= levels[name][-1]]
    if outside:
        return {"status": "omitted", "reason": "Baseline outside the submitted range: " + ", ".join(outside)}
    return {"status": "shown", "reason": "Actual separately predicted baseline coordinates; no nearest-grid substitution",
            "coordinates": {name: design[name] for name in names},
            "on_sampled_grid": all(design[name] in levels[name] for name in names)}


def response_summary(result, axes, target, scenario, baseline_row=None):
    names, levels, fixed, cells, ranked = _prepare(result, axes, target, scenario)
    marker = _baseline_marker(baseline_row, names, levels, fixed)
    objective = scenario["objective"]
    fixed_text = ", ".join(f"{LABELS[key]} = {value:g}" for key, value in fixed.items())
    text = f"Fixed {fixed_text}. {len(cells)} model predictions; {len(ranked)} meet the point-prediction constraints."
    if ranked:
        best = ranked[0]
        text += (f" Best feasible in this finite set by {objective['direction']} "
                 f"{TARGET_LABELS[objective['target']]}: "
                 + ", ".join(f"{LABELS[name]} = {best['design'][name]:.6g}" for name in names)
                 + f"; objective prediction = {best['targets'][objective['target']]:.6g} {UNITS[objective['target']]}. ")
    else:
        text += " No predicted-feasible candidate in this submitted set; constraints were not relaxed."
    if marker["status"] == "omitted":
        text += " Baseline marker omitted: " + marker["reason"] + "."
    return {"n_total": len(cells), "n_feasible": len(ranked), "best": ranked[0] if ranked else None,
            "objective": dict(objective), "plot_target": target, "fixed": fixed,
            "baseline_marker": marker, "text": text,
            "missing_cells": math.prod(len(v) for v in levels.values()) - len(cells)}


def _edges(values):
    middle = [(left + right) / 2 for left, right in zip(values, values[1:])]
    return [values[0] - (values[1] - values[0]) / 2, *middle,
            values[-1] + (values[-1] - values[-2]) / 2]


def _tooltips():
    return ([{"field": name, "type": "quantitative", "title": LABELS[name]} for name in FEATURES]
            + [{"field": name, "type": "quantitative", "title": TARGET_LABELS[name]} for name in TARGETS]
            + [{"field": "constraint_status", "type": "nominal", "title": "Point / diagnostic interval"}])


def _marker_data(row, scenario, label):
    return [{**row["design"], **row["targets"], "constraint_status": constraint_status(row, scenario), "marker": label}]


def response_chart(result, axes, target, scenario, baseline_row=None):
    """Return a Vega-Lite dict; every colored cell is one actual Prediction."""
    names, levels, fixed, cells, ranked = _prepare(result, axes, target, scenario)
    summary = response_summary(result, axes, target, scenario, baseline_row)
    x = names[0]
    if len(names) == 2:
        y = names[1]
        x_edges, y_edges = _edges(levels[x]), _edges(levels[y])
        for cell in cells:
            ix, iy = levels[x].index(cell[x]), levels[y].index(cell[y])
            cell.update(x_lo=x_edges[ix], x_hi=x_edges[ix + 1], y_lo=y_edges[iy], y_hi=y_edges[iy + 1])
        x_scale, y_scale = {"domain": [x_edges[0], x_edges[-1]], "nice": False}, {"domain": [y_edges[0], y_edges[-1]], "nice": False}
        layer = {"name": "prediction_cells", "data": {"values": cells}, "mark": {"type": "rect"},
            "encoding": {"x": {"field": "x_lo", "type": "quantitative", "title": LABELS[x], "scale": x_scale},
                "x2": {"field": "x_hi"}, "y": {"field": "y_lo", "type": "quantitative", "title": LABELS[y], "scale": y_scale},
                "y2": {"field": "y_hi"}, "color": {"field": target, "type": "quantitative",
                    "title": TARGET_LABELS[target], "scale": {"scheme": "viridis"},
                    "legend": {"title": TARGET_LABELS[target], "orient": "right"}}, "tooltip": _tooltips()}}
        marker_encoding = {"x": {"field": x, "type": "quantitative", "scale": x_scale},
                           "y": {"field": y, "type": "quantitative", "scale": y_scale}}
    else:
        span = levels[x][-1] - levels[x][0]
        x_scale = {"domain": [levels[x][0] - .04 * span, levels[x][-1] + .04 * span], "nice": False}
        cells.sort(key=lambda row: row[x])
        layer = {"name": "prediction_curve", "data": {"values": cells}, "mark": {"type": "line", "point": True, "color": "#226f84"},
            "encoding": {"x": {"field": x, "type": "quantitative", "title": LABELS[x], "scale": x_scale},
                "y": {"field": target, "type": "quantitative", "title": TARGET_LABELS[target], "scale": {"zero": False}},
                "tooltip": _tooltips()}}
        marker_encoding = {"x": {"field": x, "type": "quantitative", "scale": x_scale},
                           "y": {"field": target, "type": "quantitative", "scale": {"zero": False}}}
    layers = [layer]
    for name, row, label, shape, color in (
            ("baseline_marker", baseline_row if summary["baseline_marker"]["status"] == "shown" else None,
             "Session baseline", "diamond", "#d04a35"),
            ("best_feasible_marker", ranked[0] if ranked else None,
             "Best feasible by scenario objective", "cross", "#111827")):
        if row is None:
            continue
        layers.append({"name": name, "data": {"values": _marker_data(row, scenario, label)},
            "mark": {"type": "point", "shape": shape, "filled": False, "strokeWidth": 2.5,
                     "size": 180, "color": color, "clip": False},
            "encoding": {**marker_encoding, "tooltip": [{"field": "marker", "type": "nominal"}, *_tooltips()]}})
    return normalize_streamlit_padding({"$schema": "https://vega.github.io/schema/vega-lite/v5.json", "width": "container", "height": 380,
        "padding": 16, "layer": layers,
        "config": {"axis": {"labelFontSize": 12, "titleFontSize": 12}, "view": {"stroke": None}},
        "usermeta": {"response_summary": summary, "model_id": result["model_id"],
            "model_sha256": result["model_sha256"], "actual_prediction_rows": len(cells),
            "cell_geometry": "Numeric midpoint boundaries, half-spacing ends; no interpolation or new labels",
            "not_interpolated_prediction_nodes": True,
            "line_geometry": "1D line connects actual predicted nodes only; connections are not new labels",
            "marker_key": "Red diamond: session baseline; dark cross: best feasible by scenario objective",
            "physical_state": "undetermined", "source_kind": "predicted_not_verified"}})
