"""Display-only views of the trusted model facts and existing predictions."""


def session_model_conflicts(state, model_id, model_sha256):
    """Reject another model's saved session; do not relabel its baseline/results."""
    conflicts = []
    if state.get("model_id", model_id) != model_id:
        conflicts.append("model_id")
    if state.get("active_model_sha", model_sha256) != model_sha256:
        conflicts.append("active_model_sha")
    for key in ("baseline_result", "current_result", "candidate_result", "scan_result"):
        saved = state.get(key)
        if saved is None:
            continue
        if not isinstance(saved, dict):
            conflicts.append(key)
            continue
        result = saved if key == "baseline_result" else saved.get("prediction", {})
        if (not isinstance(result, dict) or result.get("model_id") != model_id
                or result.get("model_sha256") != model_sha256):
            conflicts.append(key)
    pending = state.get("pending_request")
    if pending is not None and (not isinstance(pending, dict) or pending.get("source_model_id") != model_id
            or pending.get("source_model_sha256") != model_sha256):
        conflicts.append("pending_request")
    return conflicts


def predicted_constraint_label(checks):
    """Keep point decisions separate from limited diagnostic interval status."""
    if any(not item["point_pass"] for item in checks):
        return "Predicted infeasible"
    if any(item["interval_status"] in {"crosses_threshold", "not_available"} for item in checks):
        return "Uncertain"
    return "Predicted feasible"


def method_view(metadata, targets, labels):
    """Format saved aggregate evidence; never fit, score, or rewrite its facts."""
    card = metadata["model_card"]
    evidence = metadata["metrics"]["metrics"]
    if card["training_target_source"] != "repository_guided_synthetic":
        raise ValueError("Unrecognized training provenance; review source explanation")
    rows = []
    for target in targets:
        value = evidence["targets"][target]
        rows.append({"Output": labels[target], "MAE": value["mae"],
            "Absolute-error P95": value["absolute_error_quantiles"]["P95"],
            "Within stated tolerance": f"{value['passed']:,}/{value['n']:,}"})
    all_passed = (evidence.get("finite_valid") is True and evidence["n"] > 0
        and evidence.get("joint_passed") == evidence["n"]
        and all(evidence["targets"][target]["passed"] == evidence["n"]
                and evidence["targets"][target]["n"] == evidence["n"] for target in targets))
    return {"training_text": f"Trained on {card['n_train']:,} repository-guided synthetic designs.",
        "source_detail": "These labels are not MATLAB/GMMNLSE simulation outputs. Full-solver validation of these predictions has not been established.",
        "n_train": card["n_train"], "n_held_out": evidence["n"],
        "n_features": card["actual_feature_count"], "rows": rows,
        "all_held_out_passed": all_passed,
        "joint_interval_coverage": metadata["metrics"]["intervals"]["joint_coverage"]}
