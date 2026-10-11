"""Data-free prediction for the single trusted 10k model.

This module has no training/evaluator/scan imports. Hashes check integrity of a
locally trusted catalog, not authenticity or sandboxing of arbitrary pickle.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import sys

FEATURES = ("pump_power_W", "filter_offset_nm", "filter2_bw_nm", "gain_length_m")
TARGETS = ("compressed_energy_nJ", "compressed_duration_fs", "compressed_peak_power_MW")
INPUT_UNITS = dict(zip(FEATURES, ("W", "nm", "nm", "m")))
UNITS = dict(zip(TARGETS, ("nJ", "fs", "MW")))
BOUNDS = dict(zip(FEATURES, ([2, 10], [2, 16], [.5, 8], [1, 8])))
MODEL_ID = "demo_10k_v1"
MODEL_SHA = "9885e8f258fa10de0a69844b70fa72242fcc64267d9db1e6c07dd4bd4609b6d0"
MODEL_MANIFEST_SHA = "9908e22c30358d51f0f8993a39da5c2e8fbc76710fbf08213ab7199e1ae9ce7c"
MODELS = {MODEL_ID: (MODEL_ID, 10000)}
CLASS_SHA = "46f9779b878d195b601930446cce3ec5335f694ef219f28c2ae728a353590b11"
DOMAIN_SHA = "66802d60ab841a6ca5f7530147c80cbd465936ccef8b82b411f38d7a126ba668"
ASSETS = {"model.joblib", "models_v4.py", "domain.json", "calibration.json", "metrics.json", "model_card.json", "dependencies.json"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def _json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_pairs)


def load_catalog(assets_root):
    path = Path(assets_root).resolve()
    if path.is_symlink() or (path / "catalog.json").is_symlink():
        raise ValueError("Redirected catalog is not supported")
    catalog = _json(path / "catalog.json")
    if (catalog.get("schema_version") != "mamyshev_demo_catalog_v1"
            or catalog.get("default_model_id") != MODEL_ID
            or set(catalog.get("models", {})) != set(MODELS)):
        raise ValueError("Expected local fixed-version catalog; no arbitrary model uploads")
    for model_id, entry in catalog["models"].items():
        if entry.get("directory") != MODELS[model_id][0]:
            raise ValueError("Untrusted model directory")
        if entry.get("manifest_sha256") != MODEL_MANIFEST_SHA or entry.get("model_sha256") != MODEL_SHA:
            raise ValueError("Unexpected frozen model or manifest identity")
    return catalog


def require_current_model(model_id, model_sha256):
    """Reject foreign/stale result identities; never migrate old predictions."""
    if model_id != MODEL_ID or model_sha256 != MODEL_SHA:
        raise ValueError("Only the current 10k model is accepted; submit a fresh request")


def verify(assets_root, model_id):
    root = Path(assets_root).resolve(); catalog = load_catalog(root)
    if model_id not in catalog["models"]:
        raise ValueError("Model is not an accepted member of the trusted local catalog")
    folder = root / catalog["models"][model_id]["directory"]
    if folder.is_symlink() or not folder.is_dir(): raise ValueError("Missing or redirected model directory")
    manifest_path = folder / "manifest.json"
    if manifest_path.is_symlink() or sha(manifest_path) != catalog["models"][model_id]["manifest_sha256"]:
        raise ValueError("Inference manifest identity differs from trusted catalog")
    manifest = _json(manifest_path)
    if (manifest.get("schema_version") != "mamyshev_demo_inference_v1" or manifest.get("model_id") != model_id
            or set(manifest.get("files", {})) != ASSETS):
        raise ValueError("Unexpected standalone inference manifest")
    for name, identity in manifest["files"].items():
        path = folder / name
        if path.is_symlink() or not path.is_file() or sha(path) != identity:
            raise ValueError("Missing or changed inference asset: " + name)
    if manifest["files"]["models_v4.py"] != CLASS_SHA or manifest["files"]["domain.json"] != DOMAIN_SHA:
        raise ValueError("Original model implementation/domain identity changed")
    require_current_model(model_id, manifest["files"]["model.joblib"])
    card = _json(folder / "model_card.json"); cal = _json(folder / "calibration.json")
    domain = _json(folder / "domain.json"); metrics = _json(folder / "metrics.json")
    if (domain["bounds"] != BOUNDS or domain["gain_length_step_m"] != .1
            or card["model_id"] != model_id or card["n_train"] != MODELS[model_id][1]
            or card["n_unique_fit"] != card["n_train"] or card["available_for_demo"] is not True
            or card["model_sha256"] != manifest["files"]["model.joblib"]
            or card["features"] != list(FEATURES) or card["targets"] != list(TARGETS)
            or card["input_units"] != INPUT_UNITS or card["units"] != UNITS
            or cal["model_sha256"] != card["model_sha256"]):
        raise ValueError("Model/units/training/calibration metadata do not agree")
    q = cal.get("q_joint_log")
    if isinstance(q, bool) or not isinstance(q, (int, float)) or not math.isfinite(q) or q < 0:
        raise ValueError("Accepted model requires its own finite calibration q")
    if cal["n_calibration"] != 1024 or cal["rank"] != 974 or cal["alpha"] != .05:
        raise ValueError("Calibration contract differs")
    versions = _json(folder / "dependencies.json")
    for name, version in versions["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Numerical dependency version changed: " + name)
    return {**card, "model_card": card, "metrics": metrics, "calibration": cal,
        "manifest_sha256": sha(manifest_path), "status": "verified", "files_verified": len(ASSETS),
        "hashes_are_security_certification": False}


def validate_design(value):
    if not isinstance(value, dict) or set(value) != set(FEATURES):
        raise ValueError("Exactly four design fields required")
    result = {}
    for key in FEATURES:
        x = value[key]
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            raise ValueError(key + ": finite real input required")
        if not BOUNDS[key][0] <= x <= BOUNDS[key][1]: raise ValueError(key + ": outside exploratory domain")
        result[key] = float(x)
    if abs(result["gain_length_m"] * 10 - round(result["gain_length_m"] * 10)) > 1e-8:
        raise ValueError("Gain length must already be on the 0.1 m grid")
    return result


def load_model(assets_root, model_id):
    metadata = verify(assets_root, model_id)
    folder = Path(assets_root).resolve() / MODELS[model_id][0]; source = folder / "models_v4.py"
    loaded = sys.modules.get("models_v4")
    if loaded is not None:
        actual = Path(getattr(loaded, "__file__", ""))
        if not actual.is_file() or sha(actual) != CLASS_SHA:
            raise ValueError("Another untrusted models_v4 module is already loaded")
    else:
        spec = importlib.util.spec_from_file_location("models_v4", source)
        loaded = importlib.util.module_from_spec(spec)
        sys.modules["models_v4"] = loaded
        exec(compile(source.read_bytes(), str(source), "exec"), loaded.__dict__)
    import joblib
    model = joblib.load(folder / "model.joblib")
    if (type(model) is not loaded.LogSurrogate or model.family != "spline_interactions"
            or model.parameters != metadata["parameters"] or model.domain != BOUNDS or len(model.models) != 3):
        raise ValueError("Loaded object is not the expected frozen fitted model")
    return model, metadata


def predict(assets_root, model_id, designs):
    designs = [validate_design(d) for d in designs]
    if not 1 <= len(designs) <= 4096: raise ValueError("Prediction batch must contain 1..4096 rows")
    model, metadata = load_model(assets_root, model_id)
    import numpy as np
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        values = np.asarray(model.predict([[d[k] for k in FEATURES] for d in designs]))
    if values.shape != (len(designs), 3) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Invalid three-target model output")
    q = metadata["calibration"]["q_joint_log"]
    rows = [{"kind": "Prediction", "source_kind": "surrogate_prediction", "design": d,
        "targets": dict(zip(TARGETS, y.tolist())), "lower": dict(zip(TARGETS, (y * math.exp(-q)).tolist())),
        "upper": dict(zip(TARGETS, (y * math.exp(q)).tolist()))} for d, y in zip(designs, values)]
    return {"kind": "Prediction", "model_id": model_id, "model_sha256": metadata["model_sha256"],
        "n_train": metadata["n_train"], "training_target_source": "repository_guided_synthetic",
        "interval_status": "limited", "q_joint_log": q, "input_units": INPUT_UNITS, "units": UNITS,
        "n_predictions": len(rows), "predictions": rows, "physical_state": "undetermined",
        "numerical_validation": "not_verified", "eligible_as_validated_training_truth": False,
        "Evaluation_created": 0, "fit_calls": 0, "oracle_calls": 0, "MATLAB_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--model", default=MODEL_ID, choices=tuple(MODELS))
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.input)
    if path.suffix.lower() == ".csv":
        with path.open() as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(FEATURES): raise ValueError("Expected exact ordered four-column CSV")
            values = [{k: float(r[k]) for k in FEATURES} for r in reader]
    else:
        values = _json(path); values = values if isinstance(values, list) else [values]
    output = predict(args.assets, args.model, values)
    with Path(args.output).open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False); stream.write("\n")


if __name__ == "__main__": main()
