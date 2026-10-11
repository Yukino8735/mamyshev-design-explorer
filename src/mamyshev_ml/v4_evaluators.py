"""Generated from authoritative main-project pure helpers; do not edit here."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


PACKAGE = ROOT / "mamyshev_accuracy_v4"


FEATURES = ("pump_power_W", "filter_offset_nm", "filter2_bw_nm", "gain_length_m")


TARGETS = ("compressed_energy_nJ", "compressed_duration_fs", "compressed_peak_power_MW")


UNITS = dict(zip(TARGETS, ("nJ", "fs", "MW")))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def validate_design(design):
    if set(design) != set(FEATURES):
        raise ValueError("Exactly four design fields required; no aliases or silently ignored fields.")
    domain = json.loads((PACKAGE / "configs/domain.json").read_text())
    result = {}
    for key in FEATURES:
        x = design[key]
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            raise ValueError(f"{key}: finite real required")
        lo, hi = domain["bounds"][key]
        if not lo <= x <= hi:
            raise ValueError(f"{key}: outside declared exploratory domain [{lo}, {hi}]")
        result[key] = float(x)
    grid = result["gain_length_m"] / domain["gain_length_step_m"]
    if abs(grid - round(grid)) > 1e-9:
        raise ValueError("Gain length must already be on the 0.1 m save grid; no implicit rounding.")
    return result


def design_id(design):
    return "d_" + digest(validate_design(design))[:24]
