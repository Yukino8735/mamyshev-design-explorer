"""Task-oriented local design comparison using trusted frozen predictions only."""
from copy import deepcopy
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st
from mamyshev_ml.demo_inference import load_catalog, verify, predict, BOUNDS, validate_design, MODEL_ID, MODEL_SHA
from ui_logic import (ASSETS, DEFAULT, FEATURES, LABELS, TARGETS, TARGET_LABELS,
                      scan_plan, candidate_plan, rank_predictions, pending_requests,
                      prediction_csv, read_summary, canonical, constraint_rows,
                      compare_predictions, request_key)
from ui_charts import response_chart, response_summary
from ui_presentation import session_model_conflicts, predicted_constraint_label, method_view

st.set_page_config(page_title="Mamyshev Design Explorer", page_icon="🔬", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1440px; padding-top: 2rem; padding-bottom: 2rem;}
h1 {letter-spacing: -.035em;} h2 {letter-spacing: -.02em;}
[data-testid="stMetricValue"] {font-size: 1.8rem;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {font-size: .85rem;}
</style>""", unsafe_allow_html=True)

SCENARIO_NAMES = {
    "S1_peak_under_40fs": "Higher peak power · duration constrained",
    "S2_low_pump": "Lower pump budget · higher peak power",
    "S3_fixed_hardware": "Fixed hardware · tune the two filters",
    "S4_short_pulse": "Shorter duration · energy constrained",
}
PAGES = ["Design workbench", "Parameter response", "Method & data", "MATLAB pipeline"]
DECIMALS = dict(zip(TARGETS, (2, 2, 3)))
SHORT_TARGETS = dict(zip(TARGETS, ("Energy (nJ)", "Duration (fs)", "Peak power (MW)")))


def clear_results():
    for key in ("current_result", "candidate_result", "scan_result", "pending_request",
                "candidate_focus", "selected_candidates", "latest_chart_spec", "latest_chart_summary"):
        st.session_state.pop(key, None)


def load_example():
    # An explicit reset starts a fresh baseline/result context for this session.
    st.session_state.model_id = MODEL_ID
    st.session_state.pop("baseline_result", None)
    st.session_state.pop("active_model_sha", None)
    st.session_state.page = "Design workbench"
    st.session_state.scenario = "S3_fixed_hardware"
    st.session_state.baseline_design = dict(DEFAULT)
    for name in FEATURES:
        st.session_state["current_" + name] = DEFAULT[name]
    prefix = "cand_S3_fixed_hardware_"
    for name in FEATURES:
        for suffix in ("vary", "fixed", "lo", "hi"):
            st.session_state.pop(prefix + name + "_" + suffix, None)
    st.session_state["constraint_S3_fixed_hardware_compressed_duration_fs_lte"] = 40.
    st.session_state.candidate_n = 512
    st.session_state.candidate_seed = 72001
    clear_results()


try:
    catalog = load_catalog(ASSETS / "models")
    st.session_state.setdefault("model_id", catalog["default_model_id"])
    st.session_state.setdefault("baseline_design", dict(DEFAULT))
    st.session_state.setdefault("scenario", "S3_fixed_hardware")
    model_id = MODEL_ID
    metadata = verify(ASSETS / "models", model_id)
    scenarios = read_summary("scenarios.json")["scenarios"]
except Exception as exc:
    st.error("Trusted model assets unavailable: " + str(exc))
    st.info("Restore the reviewed model assets before making predictions.")
    st.stop()

conflicts = session_model_conflicts(st.session_state, MODEL_ID, MODEL_SHA)
if conflicts:
    st.error("Saved results belong to a different model or lack its identity. They cannot be reused here.")
    st.button("Load fixed-hardware example", key="load_example", on_click=load_example, type="primary")
    st.caption("This action resets the baseline and saved requests for the current model.")
    st.stop()
if st.session_state.get("active_model_sha") != metadata["model_sha256"]:
    clear_results()
    st.session_state.active_model_sha = metadata["model_sha256"]
if st.session_state.get("page") not in PAGES:
    st.session_state.page = PAGES[0]
page = st.sidebar.radio("View", PAGES, key="page")
st.sidebar.caption("Model: 10k surrogate")
st.title("Mamyshev Design Explorer")
st.write("Compare designs, explore parameter trade-offs, and prepare candidates for solver validation.")
st.caption(f"{metadata['n_train']:,} training designs · {len(FEATURES)} inputs · {len(TARGETS)} predicted outputs")
st.caption("Model predictions — verify selected designs with the solver.")


@st.cache_data(max_entries=32, show_spinner=False, scope="session")
def frozen_prediction(selected_id, model_sha256, complete_designs_json):
    result = predict(ASSETS / "models", selected_id, json.loads(complete_designs_json))
    if result["model_sha256"] != model_sha256:
        raise ValueError("Model identity changed during cached request")
    return result


def execute(designs):
    return frozen_prediction(model_id, metadata["model_sha256"], canonical(designs))


def identity(payload):
    return request_key(model_id, metadata["model_sha256"], payload)


def active_result(name, key):
    saved = st.session_state.get(name)
    if saved and saved["key"] != key:
        st.session_state.pop(name, None)
        if name == "candidate_result":
            for item in ("pending_request", "candidate_focus", "selected_candidates"):
                st.session_state.pop(item, None)
        if name == "scan_result":
            st.session_state.pop("latest_chart_spec", None)
            st.session_state.pop("latest_chart_summary", None)
        st.caption("Inputs changed. Previous results were cleared; submit again to compare this request.")
        return None
    return saved


def input_editor(name, key, value=None, disabled=False):
    lo, hi = BOUNDS[name]
    return st.number_input(LABELS[name], min_value=float(lo), max_value=float(hi),
        value=float(DEFAULT[name] if value is None else value), step=.1, key=key, disabled=disabled)


def design_caption(design):
    return " · ".join(f"{LABELS[name]}: {design[name]:g}" for name in FEATURES)


def metrics(row):
    for column, target in zip(st.columns(3), TARGETS):
        column.metric(SHORT_TARGETS[target], f"{row['targets'][target]:.{DECIMALS[target]}f}")


def table_config():
    config = {name: st.column_config.NumberColumn(LABELS[name], format="%.3f") for name in FEATURES}
    config.update({name: st.column_config.NumberColumn(SHORT_TARGETS[name],
        format="%." + str(DECIMALS[name]) + "f") for name in TARGETS})
    return config


def show_constraints(row, scenario, title):
    details = constraint_rows(row, scenario)
    st.caption(title)
    for entry in details:
        operator = {"lte": "≤", "gte": "≥", "eq": "="}[entry["operator"]]
        point = "Predicted feasible" if entry["point_pass"] else "Predicted infeasible"
        st.write(f"**{entry['label']} {operator} {entry['boundary']:g}** — {point}.")
    return details


def editable_scenario(prefix, key):
    ids = [s["id"] for s in scenarios]
    chosen = st.selectbox("Design task", ids, index=ids.index("S3_fixed_hardware"),
        format_func=SCENARIO_NAMES.get, key=key)
    scenario = deepcopy(next(s for s in scenarios if s["id"] == chosen))
    objective = scenario["objective"]
    st.write(f"**Objective:** {objective['direction']} {TARGET_LABELS[objective['target']].lower()}.")
    entries = [(kind, field, op, value) for kind in ("input_constraints", "output_constraints")
               for field, limits in scenario[kind].items() for op, value in limits.items()]
    columns = st.columns(max(1, len(entries)))
    for column, (kind, field, op, value) in zip(columns, entries):
        label = (LABELS if field in FEATURES else TARGET_LABELS)[field]
        suffix = "upper limit" if op == "lte" else "lower limit"
        limits = {"min_value": 0.}
        if field in FEATURES:
            limits = {"min_value": float(BOUNDS[field][0]), "max_value": float(BOUNDS[field][1])}
        scenario[kind][field][op] = column.number_input(f"{label} · {suffix}", value=float(value), step=.1,
            key=f"{prefix}constraint_{chosen}_{field}_{op}", **limits)
    fixed = scenario["fixed_inputs"]
    if fixed:
        st.caption("Scenario fixed inputs: " + " · ".join(f"{LABELS[k]} = {v:g}" for k, v in fixed.items()))
    else:
        st.caption("No scenario-fixed hardware. Fix any input below or explore its declared range.")
    return scenario


def raw_downloads(result, name):
    a, b = st.columns(2)
    a.download_button("Download full-precision predictions (CSV)", prediction_csv(result),
        name + ".csv", "text/csv", key=name + "_csv")
    b.download_button("Download raw prediction record (JSON)", json.dumps(result, indent=2),
        name + ".json", "application/json", key=name + "_json")


def candidate_map(result, ranked, ranges, scenario):
    """A scatter of sampled X-only candidates, distinct from a response grid."""
    variable = [name for name in FEATURES if name in ranges and ranges[name][0] < ranges[name][1]]
    if len(variable) < 2:
        return
    x, y = variable[:2]
    target = scenario["objective"]["target"]
    feasible = {canonical(row["design"]) for row in ranked}
    values = [{**row["design"], **row["targets"], "point_constraints":
        "satisfied" if canonical(row["design"]) in feasible else "not_satisfied"}
        for row in result["predictions"]]
    tooltip = ([{"field": name, "type": "quantitative", "title": LABELS[name]} for name in FEATURES]
        + [{"field": name, "type": "quantitative", "title": TARGET_LABELS[name]} for name in TARGETS]
        + [{"field": "point_constraints", "title": "Point constraints"}])
    encoding = {"x": {"field": x, "type": "quantitative", "title": LABELS[x], "scale": {"zero": False}},
        "y": {"field": y, "type": "quantitative", "title": LABELS[y], "scale": {"zero": False}},
        "tooltip": tooltip}
    spec = {"height": 260, "data": {"values": values}, "layer": [
        {"transform": [{"filter": "datum.point_constraints === 'not_satisfied'"}],
            "mark": {"type": "circle", "size": 28, "color": "#bfc5cc", "opacity": .65}, "encoding": encoding},
        {"transform": [{"filter": "datum.point_constraints === 'satisfied'"}],
            "mark": {"type": "circle", "size": 38}, "encoding": {**encoding,
                "color": {"field": target, "type": "quantitative", "title": TARGET_LABELS[target],
                    "scale": {"scheme": "viridis"}, "legend": {"orient": "right"}}}}]}
    st.vega_lite_chart(spec=spec, width="stretch")
    st.caption("Sampled candidates: colored points meet the prediction constraints; gray points do not.")


def comparison(baseline, candidate, scenario, title):
    st.subheader(title)
    st.caption("Predicted performance")
    changes = compare_predictions(baseline, candidate, scenario["objective"])
    left, right = st.columns(2)
    with left:
        st.markdown("**Baseline**")
        st.caption(design_caption(baseline["design"]))
        metrics(baseline)
    with right:
        st.markdown("**Selected design**")
        st.caption(design_caption(candidate["design"]))
        metrics(candidate)
    objective = scenario["objective"]["target"]
    change = changes[objective]
    percent = change["directional_improvement_percent"]
    percent_text = "unavailable (invalid or zero baseline)" if percent is None else f"{percent:+.2f}%"
    st.write(f"**Predicted change vs baseline: {percent_text}** in the objective direction.")
    rows = []
    for target in TARGETS:
        item = changes[target]
        rows.append({"Output": SHORT_TARGETS[target], "Baseline": item["baseline"],
            "Selected design": item["candidate"], "Selected − baseline": item["absolute_change"],
            "Relative change (%)": item["relative_change_percent"],
            "Preferred direction": item["preferred_direction"],
            "Directional change (%)": item["directional_improvement_percent"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        field: st.column_config.NumberColumn(format="%.3f") for field in
        ("Baseline", "Selected design", "Selected − baseline", "Relative change (%)", "Directional change (%)")})
    st.caption("Changes use the displayed baseline at full precision.")
    c1, c2 = st.columns(2)
    with c1:
        base_constraints = show_constraints(baseline, scenario, "Baseline constraints")
    with c2:
        candidate_constraints = show_constraints(candidate, scenario, "Selected-design constraints")
    with st.expander("Diagnostic prediction intervals"):
        interval_rows = []
        for label, row in (("Baseline", baseline), ("Selected design", candidate)):
            for target in TARGETS:
                interval_rows.append({"Design": label, "Output": SHORT_TARGETS[target],
                    "Lower": row["lower"][target], "Prediction": row["targets"][target],
                    "Upper": row["upper"][target]})
        st.dataframe(pd.DataFrame(interval_rows), hide_index=True, width="stretch")
    return {"differences": changes, "baseline_constraints": base_constraints,
            "candidate_constraints": candidate_constraints}


try:
    baseline_result = execute([st.session_state.baseline_design])
    st.session_state.baseline_result = baseline_result
    baseline = baseline_result["predictions"][0]
except Exception as exc:
    st.error("Baseline prediction rejected: " + str(exc))
    st.stop()

technical = {"model": metadata, "baseline": baseline_result}

if page == "Design workbench":
    st.button("Load fixed-hardware example", key="load_example", on_click=load_example, type="primary")
    st.caption("Example: hold pump at 4 W and gain length at 4 m; tune filter offset and bandwidth for higher predicted peak power under a 40 fs duration limit.")
    st.subheader("1 · Set the task and compare a baseline")
    scenario = editable_scenario("", "scenario")
    with st.container(border=True):
        st.markdown("**Baseline · Predicted performance**")
        st.caption(design_caption(baseline["design"]))
        metrics(baseline)
    with st.expander("Edit a single design or set a different session baseline"):
        columns = st.columns(4)
        current = {}
        for column, name in zip(columns, FEATURES):
            with column:
                current[name] = input_editor(name, "current_" + name)
        a, b = st.columns(2)
        submit_current = a.button("Predict current design", key="predict_current")
        set_baseline = b.button("Use current design as baseline", key="set_baseline")
        current_key = identity({"design": current, "scenario": scenario})
        current_saved = active_result("current_result", current_key)
        if set_baseline:
            try:
                new_baseline_design = validate_design(current)
                new_baseline_result = execute([new_baseline_design])
                st.session_state.baseline_design = new_baseline_design
                st.session_state.baseline_result = new_baseline_result
                clear_results()
                st.rerun()
            except Exception as exc:
                st.error("Baseline rejected: " + str(exc))
        if submit_current:
            st.session_state.pop("current_result", None)
            current_saved = None
            try:
                current_saved = {"key": current_key, "prediction": execute([current])}
                st.session_state.current_result = current_saved
            except Exception as exc:
                st.error("Current design rejected: " + str(exc))
        if current_saved:
            technical["current_comparison"] = comparison(baseline,
                current_saved["prediction"]["predictions"][0], scenario, "Current design vs baseline")
            technical["current_prediction"] = current_saved["prediction"]

    st.subheader("2 · Explore candidate designs")
    st.caption("Choose which inputs to vary and set their ranges.")
    fixed, ranges = {}, {}
    for column, name in zip(st.columns(4), FEATURES):
        with column:
            st.markdown("**" + LABELS[name] + "**")
            prefix = f"cand_{scenario['id']}_{name}_"
            if name in scenario["fixed_inputs"]:
                fixed[name] = input_editor(name, prefix + "fixed", scenario["fixed_inputs"][name], disabled=True)
                st.caption("Fixed by this scenario")
            elif st.checkbox("Vary this input", value=True, key=prefix + "vary"):
                low, high = BOUNDS[name]
                lo = st.number_input("From", float(low), float(high), float(low), .1, key=prefix + "lo")
                hi = st.number_input("To", float(low), float(high), float(high), .1, key=prefix + "hi")
                ranges[name] = [lo, hi]
            else:
                fixed[name] = input_editor(name, prefix + "fixed", baseline["design"][name])
    c1, c2 = st.columns(2)
    count = int(c1.number_input("Candidate count", 1, 4096, 512, 1, key="candidate_n"))
    seed = int(c2.number_input("Candidate seed", 0, 2**31 - 1, 72001, 1, key="candidate_seed"))
    st.caption(f"Explore up to {count:,} candidate designs using the trained model.")
    max_pump = min(10., scenario["input_constraints"].get("pump_power_W", {}).get("lte", 10.))
    candidate_scenario = deepcopy(scenario)
    candidate_scenario["fixed_inputs"].update(fixed)
    payload = {"scenario": candidate_scenario, "scenario_preset": scenario,
        "count": count, "seed": seed, "fixed": fixed,
        "ranges": ranges, "max_pump": max_pump, "baseline": baseline["design"], "current_input": current}
    candidate_key = identity(payload)
    saved = active_result("candidate_result", candidate_key)
    if st.button("Predict and rank candidates", key="rank", type="primary"):
        for item in ("candidate_result", "pending_request", "candidate_focus", "selected_candidates"):
            st.session_state.pop(item, None)
        try:
            plan = candidate_plan(candidate_scenario, count, max_pump, fixed, seed=seed, ranges=ranges)
            result = execute([r["design_config"] for r in plan["designs"]])
            saved = {"key": candidate_key, "prediction": result,
                "ranked": rank_predictions(result, candidate_scenario), "scenario": candidate_scenario,
                "plan": plan, "baseline": baseline, "request": payload}
            st.session_state.candidate_result = saved
        except Exception as exc:
            saved = None
            st.error("Candidate request rejected: " + str(exc))
    if saved:
        result, ranked = saved["prediction"], saved["ranked"]
        st.subheader("Top predicted candidates in this search")
        st.write(f"**{len(ranked):,} of {result['n_predictions']:,} predicted designs** satisfy every point-prediction condition.")
        st.caption("Selected candidates still require solver validation.")
        candidate_map(result, ranked, ranges, candidate_scenario)
        if not ranked:
            st.info("No predicted-feasible candidates in this set. Constraints have not been relaxed; no recommendation or request is produced.")
        else:
            table = []
            for rank, row in enumerate(ranked[:12], 1):
                conditions = constraint_rows(row, candidate_scenario)
                table.append({"Rank": rank, **row["design"], **row["targets"],
                    "Prediction status": predicted_constraint_label(conditions)})
            st.dataframe(pd.DataFrame(table), column_config=table_config(), hide_index=True, width="stretch")
            focus = st.selectbox("Candidate to compare with baseline", list(range(min(12, len(ranked)))),
                format_func=lambda i: f"Rank {i + 1}", key="candidate_focus")
            details = comparison(baseline, ranked[focus], candidate_scenario, "Selected design vs baseline")
            saved["comparison"] = details
            selected = st.multiselect("Candidates to export (maximum 3)", list(range(min(12, len(ranked)))),
                default=[0], format_func=lambda i: f"Rank {i + 1}", max_selections=3, key="selected_candidates")
            if selected:
                request = pending_requests(result, [ranked[i] for i in selected], candidate_scenario)
                st.session_state.pending_request = request
                st.download_button("Export for solver validation", json.dumps(request, indent=2),
                    "pending_matlab_requests.json", "application/json", key="pending_download")
                st.caption("Creates request files; no solver run is started.")
            else:
                st.session_state.pop("pending_request", None)
                st.caption("Select up to three candidates to prepare a request.")
        technical["candidate_request"] = saved["request"]
        technical["candidate_result"] = result
        technical["pending_request"] = st.session_state.get("pending_request")

elif page == "Parameter response":
    st.header("Parameter response")
    st.write("Explore how predicted performance changes as design parameters vary.")
    scenario = editable_scenario("response_", "response_scenario")
    mode = st.selectbox("Dimensions", ["1D", "2D"], key="scan_mode")
    c1, c2 = st.columns(2)
    axis1 = c1.selectbox("First axis", FEATURES, index=2, format_func=LABELS.get, key="axis1")
    axis2 = c2.selectbox("Second axis", FEATURES, index=1, format_func=LABELS.get, key="axis2") if mode == "2D" else None
    names = [axis1] if axis2 is None else [axis1, axis2]
    ranges, counts = {}, {}
    for index, (column, name) in enumerate(zip(st.columns(len(names)), names), 1):
        with column:
            st.markdown("**" + LABELS[name] + " range**")
            lo, hi = BOUNDS[name]
            # Distinct widget keys even for the deliberately rejected duplicate-axis case.
            prefix = f"scan_axis{index}_{name}_"
            start = st.number_input("From", float(lo), float(hi), float(lo), .1, key=prefix + "lo")
            end = st.number_input("To", float(lo), float(hi), float(hi), .1, key=prefix + "hi")
            n = int(st.number_input("Points on this axis", 2, 21, 9, 1, key="scan_n" + str(index)))
            ranges[name] = [start, end]
            counts[name] = n
            if name == "gain_length_m":
                st.caption("Endpoints must lie on the 0.1 m grid; requested levels must be distinct. Unsupported requests are rejected.")
    fixed = dict(baseline["design"])
    st.markdown("**Fixed inputs**")
    remaining = [name for name in FEATURES if name not in names]
    for column, name in zip(st.columns(len(remaining)), remaining):
        with column:
            fixed[name] = input_editor(name, "scan_fixed_" + name, baseline["design"][name])
    target = st.selectbox("Plot target", TARGETS, index=2, format_func=TARGET_LABELS.get, key="scan_target")
    st.caption("Session baseline: " + design_caption(baseline["design"]))
    payload = {"mode": mode, "axes": names, "ranges": ranges, "counts": counts,
        "fixed": fixed, "target": target, "scenario": scenario, "baseline": baseline["design"]}
    scan_key = identity(payload)
    saved = active_result("scan_result", scan_key)
    if st.button("Run prediction response", key="run_scan", type="primary"):
        st.session_state.pop("scan_result", None)
        st.session_state.pop("latest_chart_spec", None)
        st.session_state.pop("latest_chart_summary", None)
        try:
            plan = scan_plan(fixed, axis1, axis2, count=counts[axis1], ranges=ranges, counts=counts)
            result = execute([r["design_config"] for r in plan["designs"]])
            saved = {"key": scan_key, "prediction": result, "plan": plan, "axes": plan["axes"],
                "target": target, "scenario": scenario, "baseline": baseline, "request": payload}
            st.session_state.scan_result = saved
        except Exception as exc:
            saved = None
            st.error("Response request rejected: " + str(exc))
    if saved:
        result = saved["prediction"]
        # Current data/export are independent of chart construction. A browser
        # JS render failure is asynchronous and cannot be caught by this Python try.
        technical["response_request"] = saved["request"]
        technical["response_prediction"] = result
        with st.expander("Prediction table · full grid"):
            st.dataframe(pd.DataFrame([{**row["design"], **row["targets"]} for row in result["predictions"]]),
                column_config=table_config(), hide_index=True, width="stretch")
        try:
            spec = response_chart(result, saved["axes"], target, scenario, baseline)
            summary = response_summary(result, saved["axes"], target, scenario, baseline)
            st.session_state.latest_chart_spec = spec
            st.session_state.latest_chart_summary = summary
            st.subheader(f"{result['n_predictions']} predicted designs")
            st.caption(" · ".join(f"{LABELS[n]} = {fixed[n]:g}" for n in remaining))
            st.vega_lite_chart(spec=spec, width="stretch")
            st.caption("Red diamond: actual session baseline. Dark cross: best feasible by "
                f"{scenario['objective']['direction']} {TARGET_LABELS[scenario['objective']['target']]}. "
                "The ranking objective can differ from the plotted output.")
            st.write(summary["text"])
            marker = summary["baseline_marker"]
            if marker["status"] != "shown":
                st.caption("Baseline marker omitted: " + marker["reason"] + ". Its separate same-model prediction is retained.")
            st.caption("Model predictions at sampled parameter settings.")
            technical["response_summary"] = summary
        except Exception as exc:
            st.session_state.pop("latest_chart_spec", None)
            st.session_state.pop("latest_chart_summary", None)
            st.error("Response chart rejected: " + str(exc))

elif page == "Method & data":
    st.header("Method & data")
    view = method_view(metadata, TARGETS, SHORT_TARGETS)
    st.subheader("Data & validation")
    st.write(view["training_text"])
    st.write(view["source_detail"])
    st.markdown("**Four inputs → spline and interaction features → fitted regression model → three predictions.**")
    st.caption(f"{view['n_features']} fitted features · stored scaling and coefficients · energy (nJ), duration (fs), peak power (MW)")
    a, b, c = st.columns(3)
    a.metric("Training designs", f"{view['n_train']:,}")
    b.metric("Held-out designs", f"{view['n_held_out']:,}")
    c.metric("Features", str(view["n_features"]))
    st.dataframe(pd.DataFrame(view["rows"]), hide_index=True, width="stretch", column_config={
        "MAE": st.column_config.NumberColumn(format="%.6f"),
        "Absolute-error P95": st.column_config.NumberColumn(format="%.6f")})
    if view["all_held_out_passed"]:
        st.write("All held-out designs met the stated error tolerances.")
    st.caption("Saved aggregate test metrics for the training-label source described above.")
    st.subheader("Diagnostic prediction intervals")
    st.write("Intervals are residual-based diagnostics. Coverage varies across regions; they are not physical safety guarantees.")
    st.caption(f"Saved joint held-out coverage: {100 * view['joint_interval_coverage']:.2f}%.")
    technical["inference_code_path"] = ["demo/app.py:frozen_prediction", "src/mamyshev_ml/demo_inference.py:predict",
        "demo/assets/models/demo_10k_v1/models_v4.py:LogSurrogate._x", "LogSurrogate.predict"]

else:
    st.header("MATLAB pipeline")
    summary = read_summary("matlab_plan_summary.json")
    st.markdown("**Plan → Simulate → Save → Resume → Store / export**")
    st.write("This online demo prepares parameter requests for the full MATLAB/GMMNLSE workflow.")
    st.write("Execution, per-point saving, resume and database / export are provided by the full research project.")
    example = {"example_kind": "handwritten_schema_example_not_train_row",
        "plan_configuration": {"N": summary.get("n_unique_designs"), "serial": True, "max_new_real_attempts": 0},
        "design_config": DEFAULT, "input_units": dict(zip(FEATURES, ["W", "nm", "nm", "m"])),
        "real_targets": dict.fromkeys(TARGETS), "status": "not_run"}
    st.subheader("Example parameter request")
    st.json({"design_config": example["design_config"], "input_units": example["input_units"]})
    st.download_button("Export request example for solver validation", json.dumps(example, indent=2),
        "pending_matlab_schema_example.json", "application/json", key="pipeline_download")
    st.caption("Creates request files; no solver run is started.")
    technical["pipeline_summary"] = summary
    technical["schema_example"] = example

with st.expander("Technical details · model identity, interval calibration and raw requests"):
    st.caption("Read-only model identity, provenance, calibration and request values.")
    st.json(technical)
    for label, field in (("current", "current_prediction"), ("candidates", "candidate_result"), ("response", "response_prediction")):
        if field in technical:
            raw_downloads(technical[field], label + "_prediction")
    st.download_button("Download current technical record", json.dumps(technical, indent=2),
        "design_comparison_record.json", "application/json", key="technical_download")
