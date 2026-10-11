#!/usr/bin/env python3
"""Bounded single-model checks; run serially under the existing external monitor.

References and results must be private paths, never deployment assets. AppTest
checks component output, not JavaScript/browser rendering or cloud multiuser use.
"""
from contextlib import contextmanager
import argparse
import copy
import csv
import hashlib
import io
import json
import runpy
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MODEL = "demo_10k_v1"
RETIRED = "v42_T2_K64_A0001"  # Negative migration fixture, never loaded.
WEIGHT_SHA = "9885e8f258fa10de0a69844b70fa72242fcc64267d9db1e6c07dd4bd4609b6d0"
ATOL = RTOL = 1e-10
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "demo"))


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def runtime_guard(counts):
    old_profile, old_thread_profile = sys.getprofile(), threading.getprofile()
    original_popen = subprocess.Popen

    def guard(frame, event, arg):
        if event != "call":
            return guard
        name, file = frame.f_code.co_name, frame.f_code.co_filename
        key = None
        if name in {"fit", "partial_fit", "fit_transform", "calibrate", "run_calibration"}:
            key = "fit"
        if file.endswith("oracle.py"):
            key = "oracle"
        if name in {"evaluate", "execute_authorized_design", "run_supervised"} and any(
                s in file for s in ("v4_evaluators", "authorization", "supervisor")):
            key = "Evaluation"
        if name == "predict" and file.endswith("models_v4.py"):
            counts["predict"] += 1
        if key:
            counts[key] += 1
            raise AssertionError("Forbidden computation: " + key)
        return guard

    def popen(*args, **kwargs):
        command = args[0] if args else kwargs.get("args", "")
        if any(word in str(command).lower() for word in ("matlab", "run_mamyshev")):
            counts["MATLAB"] += 1
            raise AssertionError("Forbidden MATLAB launch")
        return original_popen(*args, **kwargs)

    sys.setprofile(guard)
    threading.setprofile(guard)
    subprocess.Popen = popen
    try:
        yield
    finally:
        sys.setprofile(old_profile)
        threading.setprofile(old_thread_profile)
        subprocess.Popen = original_popen


def clean(at):
    assert not at.exception, str(at.exception)


def missing(at, *keys):
    for key in keys:
        assert key not in at.session_state, "Stale state: " + key


def numbers(actual, expected):
    import numpy as np
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    np.testing.assert_allclose(a, b, atol=ATOL, rtol=RTOL)
    return float(np.max(np.abs(a - b))) if a.size else 0.


def compare_prediction(actual, expected):
    from mamyshev_ml.demo_inference import FEATURES, TARGETS
    assert {k: v for k, v in actual.items() if k != "predictions"} == {
        k: v for k, v in expected.items() if k != "predictions"}
    assert len(actual["predictions"]) == len(expected["predictions"])
    errors = {field: 0. for field in ("design", "targets", "lower", "upper")}
    for a, b in zip(actual["predictions"], expected["predictions"]):
        assert {k: v for k, v in a.items() if k not in errors} == {
            k: v for k, v in b.items() if k not in errors}
        for field in errors:
            keys = FEATURES if field == "design" else TARGETS
            errors[field] = max(errors[field], numbers([a[field][k] for k in keys], [b[field][k] for k in keys]))
    return errors


@contextmanager
def capture_downloads():
    import streamlit as st
    from streamlit.delta_generator import DeltaGenerator
    original = DeltaGenerator.download_button
    global_original = st.download_button
    downloads = {}

    def download(self, label, data, *args, **kwargs):
        downloads[kwargs.get("key", label)] = data
        return original(self, label, data, *args, **kwargs)

    def global_download(label, data, *args, **kwargs):
        # Streamlit binds this module alias at import time, before the class
        # patch. Observe both entry points; keep each real implementation.
        downloads[kwargs.get("key", label)] = data
        return global_original(label, data, *args, **kwargs)

    with patch.object(DeltaGenerator, "download_button", download), patch.object(st, "download_button", global_download):
        yield downloads


def chart_record(at, downloads):
    import altair as alt
    import pyarrow as pa
    from mamyshev_ml.demo_inference import FEATURES, TARGETS
    from ui_logic import DEFAULT, prediction_csv, rank_predictions
    saved = at.session_state["scan_result"]
    prediction = saved["prediction"]
    chart = at.get("vega_lite_chart")[0]
    final = json.loads(chart.proto.spec)
    alt.LayerChart.from_dict(final, validate=True)
    assert final["padding"] == dict.fromkeys(("left", "right", "top", "bottom"), 16)
    assert all("padding" not in layer for layer in final["layer"])
    datasets = {item.name: pa.ipc.open_stream(io.BytesIO(item.data.data)).read_all().to_pylist()
                for item in chart.proto.datasets}
    original = at.session_state["latest_chart_spec"]
    for layer in final["layer"]:
        data = layer["data"]
        cells = data.get("values", datasets.get(data.get("name")))
        wanted = next(v for v in original["layer"] if v["name"] == layer["name"])["data"]["values"]
        assert cells == wanted
    main = final["layer"][0]
    is_2d = saved["request"]["mode"] == "2D"
    assert main["mark"]["type"] == ("rect" if is_2d else "line")
    assert len(main["encoding"]["tooltip"]) >= len(FEATURES) + len(TARGETS)
    if is_2d:
        assert {"x", "x2", "y", "y2", "color"} <= set(main["encoding"])
    markers = {layer["name"]: layer["data"]["values"][0] for layer in original["layer"][1:]}
    assert all(markers["baseline_marker"][key] == DEFAULT[key] for key in FEATURES)
    ranked = rank_predictions(prediction, saved["scenario"])
    if ranked:
        assert all(markers["best_feasible_marker"][key] == ranked[0]["targets"][key] for key in TARGETS)
    assert json.loads(downloads["response_prediction_json"]) == prediction
    assert downloads["response_prediction_csv"] == prediction_csv(prediction)
    table = [{**row["design"], **row["targets"]} for row in prediction["predictions"]]
    assert any(item.value.to_dict("records") == table for item in at.dataframe)
    return {"request": saved["request"], "prediction": prediction, "final_spec": final,
            "datasets": datasets, "summary": at.session_state["latest_chart_summary"],
            "exports": {"json": downloads["response_prediction_json"], "csv": downloads["response_prediction_csv"]}}


def chart_checks(reference, report, before=False):
    from streamlit.testing.v1 import AppTest
    from mamyshev_ml.demo_inference import TARGETS, predict
    from ui_logic import ASSETS, DEFAULT, prediction_csv
    import ui_charts
    records = {"charts": {}}
    records["single"] = predict(ASSETS / "models", MODEL, [DEFAULT])
    records["single_csv"] = prediction_csv(records["single"])
    records["trusted_hashes"] = {p.name: digest(p) for p in (ASSETS / "models" / MODEL).iterdir() if p.is_file()}
    assert records["trusted_hashes"]["model.joblib"] == WEIGHT_SHA
    with capture_downloads() as downloads:
        at = AppTest.from_file(str(ROOT / "demo/app.py"), default_timeout=10).run()
        clean(at)
        at.button(key="load_example").click().run(); clean(at)
        assert at.session_state["model_id"] == MODEL
        at.radio(key="page").set_value("Parameter response").run(); clean(at)
        for mode, count in (("1D", 9), ("2D", 81)):
            if mode == "2D":
                at.selectbox(key="scan_mode").set_value(mode).run(); clean(at)
                missing(at, "scan_result", "latest_chart_spec")
            for target in TARGETS:
                at.selectbox(key="scan_target").set_value(target).run(); clean(at)
                missing(at, "scan_result", "latest_chart_spec")
                downloads.clear()
                at.button(key="run_scan").click().run(); clean(at)
                assert not at.error, str(at.error)
                record = chart_record(at, downloads)
                assert record["prediction"]["n_predictions"] == count
                name = mode + "_" + target
                if not before:
                    report["cases"].append({"case": name, "max_abs_error": compare_prediction(
                        record["prediction"], reference["charts"][name]["prediction"]),
                        "exact_chart_and_exports_equal": record == reference["charts"][name]})
                    assert record == reference["charts"][name], "Chart geometry/markers/tooltips/exports changed"
                records["charts"][name] = record
        if not before:
            wanted = at.session_state["scan_result"]["prediction"]
            with patch.object(ui_charts, "response_chart", side_effect=ValueError("deliberate chart-only test rejection")):
                downloads.clear()
                at.run(); clean(at)
                assert at.error
                missing(at, "latest_chart_spec", "latest_chart_summary")
                assert json.loads(downloads["response_prediction_json"]) == wanted
                assert downloads["response_prediction_csv"] == prediction_csv(wanted)
                assert any(len(item.value) == 81 for item in at.dataframe)
            at.selectbox(key="axis2").set_value("filter2_bw_nm").run(); clean(at)
            missing(at, "scan_result", "latest_chart_spec")
            at.button(key="run_scan").click().run(); clean(at)
            assert at.error
            missing(at, "scan_result", "latest_chart_spec")
            report["cases"].append({"case": "chart_failure_keeps_data_exports_and_invalid_axis_clears_stale", "passed": True})
    return records


def inference_checks(reference, report):
    from mamyshev_ml.demo_inference import MODELS, predict, verify, load_catalog
    from ui_logic import ASSETS, prediction_csv, pending_requests, rank_predictions, read_summary
    assets = ASSETS / "models"
    assert set(MODELS) == {MODEL}
    catalog = load_catalog(assets)
    assert catalog["default_model_id"] == MODEL and set(catalog["models"]) == {MODEL}
    # The full research repository retains historical assets outside the active catalog.
    assert "v42_T2_K64_A0001" not in catalog["models"]
    actual_hashes = {p.name: digest(p) for p in (assets / MODEL).iterdir() if p.is_file()}
    assert actual_hashes == reference["trusted_hashes"]
    assert actual_hashes["model.joblib"] == WEIGHT_SHA
    cases = {"single": reference["single"]}
    for mode in ("1D", "2D"):
        cases[mode] = reference["charts"][mode + "_compressed_peak_power_MW"]["prediction"]
    for name, expected in cases.items():
        actual = predict(assets, MODEL, [row["design"] for row in expected["predictions"]])
        error = compare_prediction(actual, expected)
        assert prediction_csv(actual) == prediction_csv(expected)
        assert actual["training_target_source"] == "repository_guided_synthetic"
        assert actual["Evaluation_created"] == actual["fit_calls"] == actual["oracle_calls"] == actual["MATLAB_calls"] == 0
        report["cases"].append({"case": name + "_numeric_and_export", "max_abs_error": error, "q": actual["q_joint_log"]})
    def reject(name, operation):
        try:
            operation()
        except (ValueError, FileNotFoundError) as exc:
            report["cases"].append({"case": name, "passed": True, "rejection": str(exc)})
        else:
            raise AssertionError("Invalid operation accepted: " + name)
    design = reference["single"]["predictions"][0]["design"]
    reject("retired_id", lambda: predict(assets, RETIRED, [design]))
    reject("unknown_id", lambda: predict(assets, "unknown", [design]))
    reject("off_grid_length", lambda: predict(assets, MODEL, [{**design, "gain_length_m": 4.05}]))
    reject("extra_field", lambda: predict(assets, MODEL, [{**design, "cw2": 1030}]))
    result = cases["2D"]
    scenario = next(s for s in read_summary("scenarios.json")["scenarios"] if s["id"] == "S3_fixed_hardware")
    ranked = rank_predictions(result, scenario)
    assert len(ranked) >= 4
    request = pending_requests(result, ranked[:3], scenario)
    assert len(request["requests"]) == 3 and request["max_new_real_attempts"] == 0
    assert request["source_model_id"] == MODEL and request["source_kind"] == "surrogate_prediction"
    assert all(row["real_targets"] is None and row["status"] == "not_run" for row in request["requests"])
    reject("four_requests", lambda: pending_requests(result, ranked[:4], scenario))
    retired = copy.deepcopy(result); retired["model_id"] = RETIRED
    reject("retired_result_export", lambda: pending_requests(retired, retired["predictions"][:1], scenario))
    with tempfile.TemporaryDirectory(prefix="presentation_asset_fixture_") as tmp:
        copied = Path(tmp) / "models"
        shutil.copytree(assets, copied)
        weight = copied / MODEL / "model.joblib"
        raw = weight.read_bytes()
        weight.unlink()
        reject("missing_asset", lambda: verify(copied, MODEL))
        weight.write_bytes(raw + b"tamper fixture")
        reject("tampered_asset", lambda: verify(copied, MODEL))
    with tempfile.TemporaryDirectory(prefix="presentation_cli_") as tmp:
        tmp = Path(tmp)
        input_path = tmp / "input.json"
        write(input_path, design)
        argv = sys.argv
        try:
            for route, script, arguments in (
                    ("demo", ROOT / "demo" / "predict.py", ["--input", str(input_path), "--out"]),
                    ("loader", ROOT / "src" / "mamyshev_ml" / "demo_inference.py",
                     ["--assets", str(assets), "--input", str(input_path), "--output"])):
                output_path = tmp / (route + ".json")
                sys.argv = [str(script), *arguments, str(output_path)]
                runpy.run_path(str(script), run_name="__main__")
                report["cases"].append({"case": route + "_CLI_default",
                    "max_abs_error": compare_prediction(json.loads(output_path.read_text()), reference["single"])})
        finally:
            sys.argv = argv
    from ui_presentation import session_model_conflicts
    metadata = verify(assets, MODEL)
    for key in ("baseline_result", "current_result", "candidate_result", "scan_result"):
        stale = {**reference["single"], "model_id": RETIRED}
        state = {key: stale if key == "baseline_result" else {"prediction": stale}}
        assert key in session_model_conflicts(state, MODEL, metadata["model_sha256"])
    assert "pending_request" in session_model_conflicts({"pending_request": {
        "source_model_id": RETIRED, "source_model_sha256": metadata["model_sha256"]}}, MODEL, metadata["model_sha256"])
    report["cases"].append({"case": "all_trusted_model_assets_byte_identical", "hashes": actual_hashes})


def visible_text(at):
    return "\n".join(str(item.value) for kind in ("title", "header", "subheader", "markdown", "caption", "info", "warning", "error", "metric") for item in at.get(kind))


def ui_checks(report):
    from streamlit.testing.v1 import AppTest
    from mamyshev_ml.demo_inference import FEATURES, TARGETS
    from ui_logic import DEFAULT
    with capture_downloads() as downloads:
        at = AppTest.from_file(str(ROOT / "demo/app.py"), default_timeout=10).run(); clean(at)
        assert at.session_state["model_id"] == MODEL
        assert all(box.key != "model_id" and box.label != "Frozen model" for box in at.selectbox)
        assert at.button(key="load_example").label == "Load fixed-hardware example"
        at.button(key="load_example").click().run(); clean(at)
        assert at.session_state["baseline_design"] == DEFAULT
        at.button(key="predict_current").click().run(); clean(at)
        current = at.session_state["current_result"]["prediction"]
        assert current["model_id"] == at.session_state["baseline_result"]["model_id"] == MODEL
        at.number_input(key="current_pump_power_W").set_value(4.1).run(); clean(at)
        missing(at, "current_result")
        at.button(key="load_example").click().run(); clean(at)
        at.button(key="rank").click().run(); clean(at)
        saved = at.session_state["candidate_result"]
        assert saved["prediction"]["model_id"] == at.session_state["baseline_result"]["model_id"] == MODEL
        assert len(saved["ranked"]) >= 3
        at.selectbox(key="candidate_focus").set_value(1).run(); clean(at)
        at.multiselect(key="selected_candidates").set_value([0, 1, 2]).run(); clean(at)
        pending = at.session_state["pending_request"]
        assert json.loads(downloads["pending_download"]) == pending
        assert len(pending["requests"]) == 3 and pending["max_new_real_attempts"] == 0
        assert all(row["real_targets"] is None and row["evaluation_id"] is None for row in pending["requests"])
        assert pending["source_kind"] == "surrogate_prediction" and pending["source_model_id"] == MODEL
        assert all(not {"Model", "Source"} & set(table.value.columns) for table in at.dataframe)
        ranked_table = next(table.value for table in at.dataframe if set(FEATURES) <= set(table.value.columns))
        for index, (_, row) in enumerate(ranked_table.iterrows()):
            expected = saved["ranked"][index]
            numbers([row[key] for key in FEATURES], [expected["design"][key] for key in FEATURES])
            numbers([row[key] for key in TARGETS], [expected["targets"][key] for key in TARGETS])
        comparison = next(table.value for table in at.dataframe if "Selected − baseline" in table.value.columns)
        selected = saved["ranked"][1]
        numbers(comparison["Selected design"].to_numpy(), [selected["targets"][key] for key in TARGETS])
        numbers(comparison["Baseline"].to_numpy(), [saved["baseline"]["targets"][key] for key in TARGETS])
        numbers(comparison["Selected − baseline"].to_numpy(), [selected["targets"][key] - saved["baseline"]["targets"][key] for key in TARGETS])
        report["cases"].append({"case": "single_model_example_baseline_ranking_and_three_zero_authority_exports", "passed": True,
                                "predicted_candidates": saved["prediction"]["n_predictions"], "requests": len(pending["requests"])})
        key = "constraint_S3_fixed_hardware_compressed_duration_fs_lte"
        at.number_input(key=key).set_value(1.).run(); clean(at)
        missing(at, "candidate_result", "pending_request")
        at.button(key="rank").click().run(); clean(at)
        assert not at.session_state["candidate_result"]["ranked"]
        missing(at, "pending_request")
        pages = {}
        for page in ("Design workbench", "Parameter response", "Method & data", "MATLAB pipeline"):
            at.radio(key="page").set_value(page).run(); clean(at)
            text = visible_text(at)
            for old in ("Prediction ≠ SyntheticEvaluation", "Synthetic surrogate ·", "Historical SyntheticEvaluation", "actual Prediction rows", "real_targets=null", "max_new_real_attempts = 0"):
                assert old not in text, old
            assert text.count("Model predictions — verify selected designs with the solver.") == 1
            pages[page] = text
        method = pages["Method & data"]
        assert "Data & validation" in method
        assert "repository-guided synthetic" in method.lower()
        assert "matlab produced data" not in method.lower()
        assert "2,048" in method or any(item.value == "2,048" for item in at.metric)
        assert "Intervals are residual-based diagnostics." in method
        assert not any("run" in button.label.lower() or "launch" in button.label.lower() for button in at.button)
        pipeline = json.loads(downloads["pipeline_download"])
        assert pipeline["plan_configuration"]["max_new_real_attempts"] == 0
        assert all(value is None for value in pipeline["real_targets"].values())
        report["visible_page_text"] = pages
        report["cases"].append({"case": "stale_input_constraint_no_feasible_source_concentration_pipeline", "passed": True})


def session_checks(report):
    from streamlit.testing.v1 import AppTest, app_test
    from mamyshev_ml import demo_inference
    from ui_logic import DEFAULT
    original, original_runner = demo_inference.predict, app_test.LocalScriptRunner
    calls = []
    def observe(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)
    def distinct_session(*args, **kwargs):
        runner = original_runner(*args, **kwargs)
        # AppTest uses one hardcoded ID. This changes test identity only; actual
        # session-scope cache, app code, model and outputs are not mocked.
        runner._session_id = "presentation-smoke-" + str(id(args[1]))
        return runner
    with patch.object(demo_inference, "predict", observe), patch.object(app_test, "LocalScriptRunner", distinct_session):
        first = AppTest.from_file(str(ROOT / "demo/app.py"), default_timeout=10).run(); clean(first)
        first.button(key="predict_current").click().run(); clean(first)
        first.number_input(key="current_pump_power_W").set_value(4.1).run(); clean(first)
        first.button(key="set_baseline").click().run(); clean(first)
        assert first.session_state["baseline_design"]["pump_power_W"] == 4.1
        n = len(calls)
        second = AppTest.from_file(str(ROOT / "demo/app.py"), default_timeout=10).run(); clean(second)
        assert second.session_state["baseline_design"] == DEFAULT
        assert second.session_state["model_id"] == MODEL
        missing(second, "current_result", "pending_request")
        assert len(calls) > n, "Session cache leaked across distinct sessions"
        assert first.session_state["baseline_design"]["pump_power_W"] == 4.1
        second.session_state["model_id"] = RETIRED
        n = len(calls)
        second.run(); clean(second)
        assert second.error and len(calls) == n, "Retired session silently reused or predicted"
        second.button(key="load_example").click().run(); clean(second)
        assert not second.error
        assert second.session_state["model_id"] == MODEL
        assert second.session_state["baseline_design"] == DEFAULT
        assert second.session_state["baseline_result"]["model_id"] == MODEL
        report["cases"].append({"case": "distinct_session_state_cache_and_retired_session_explicit_resubmit", "passed": True,
            "session_test_ids": "distinct per SafeSessionState; AppTest default shared ID replaced in harness only",
            "browser_multiuser_acceptance": "not_run"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capture", "inference", "charts", "ui", "session"))
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "result.json").exists():
        raise FileExistsError("Refusing to overwrite test evidence")
    if args.reference is None and args.mode in ("inference", "charts"):
        args.reference = args.out / "references" / "reference.json"
    report = {"status": "running", "mode": args.mode, "atol": ATOL, "rtol": RTOL, "cases": [],
              "counts": dict(fit=0, oracle=0, Evaluation=0, MATLAB=0, predict=0),
              "browser_visual_check": "not_run", "server_started": False}
    started = time.monotonic(); error = None
    try:
        with runtime_guard(report["counts"]):
            reference = json.loads(args.reference.read_text()) if args.reference else None
            if args.mode == "capture":
                (args.out / "references").mkdir(exist_ok=False)
                write(args.out / "references" / "reference.json", chart_checks(None, report, before=True))
            elif args.mode == "inference":
                inference_checks(reference, report)
            elif args.mode == "charts":
                write(args.out / "actual.json", chart_checks(reference, report))
            elif args.mode == "ui":
                ui_checks(report)
            else:
                session_checks(report)
        assert all(report["counts"][key] == 0 for key in ("fit", "oracle", "Evaluation", "MATLAB"))
        assert not any(name.endswith("oracle") or "synthetic_generator" in name for name in sys.modules)
        report["status"] = "passed"
    except BaseException as exc:
        error = exc
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    finally:
        report["wall_s"] = time.monotonic() - started
        write(args.out / "result.json", report)
    if error:
        raise error
    print(json.dumps({"status": report["status"], "mode": args.mode, "wall_s": report["wall_s"], "counts": report["counts"]}))


if __name__ == "__main__":
    main()
