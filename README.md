# Mamyshev Design Explorer

Compare designs, explore parameter trade-offs, and prepare candidates for solver validation.

**10,000 training designs · 4 inputs · 3 predicted outputs**

This generated snapshot exposes only `demo_10k_v1`. GitHub code publication is
separate from a running website: a live app URL, Cloud/Linux acceptance and
real-browser visual review remain pending.

## Explore a design

- **Design workbench:** load the fixed-hardware example, compare a selected
  design with a baseline, rank predicted candidates and export up to three
  requests for solver validation.
- **Parameter response:** sampled 1D curves and 2D rect heatmaps retain numeric
  coordinates, fixed inputs, baseline and best predicted-feasible markers.
- **Method & data:** directly read the model's source, structure, saved aggregate
  test metrics and diagnostic interval interpretation.
- **MATLAB data pipeline:** prepare a small request for the complete project's
  Plan → Simulate → Save → Resume → Store / export workflow.

Inputs are pump power (W), SF1 − SF2 offset (nm), SF2 bandwidth (nm), and gain
length (m). Outputs are predicted compressed energy (nJ), duration (fs), and
peak power (MW). Gain length must already lie on the supported 0.1 m grid.
Tables and CSV/JSON retain full precision and source metadata. Changed inputs
invalidate old results; sessions have separate baselines, requests and caches.
Limits remain 21 curve points, 441 heatmap cells and 4,096 candidates. Exports
carry zero execution allowance and null real targets.

## Method & data

The model was trained on **10,000 repository-guided synthetic designs**.
These labels are not MATLAB/GMMNLSE simulation outputs; real-solver accuracy has
not been measured. The saved evaluation uses 2,048 held-out designs from that
same source. Original model-card facts and aggregate metrics are preserved.
Intervals are residual-based diagnostics. Coverage varies across regions;
they are not physical safety guarantees. The domain remains exploratory.

The fitted model, scaler, knots, calibration and custom `models_v4.py` retain
their frozen bytes. The loader verifies trusted weights and the inner manifest;
missing/altered assets and other model IDs fail closed. There is no training,
calibration, oracle or solver fallback.

## Run and maintain

From this snapshot's root, use an already prepared compatible Python 3.11
environment matching root `requirements.txt`:

```sh
python -m streamlit run demo/app.py
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for review and deployment settings.
`deploy_manifest.json` binds exact payload and source hashes. This snapshot is
generated from the [complete Mamyshev project](https://github.com/Yukino8735/Mamyshev-Oscillator-Simulation)'s
`scripts/export_demo.py` and public profile. **Change the main project, test, then export a new directory.**
Do not maintain this generated tree as a second source of truth.
The manifest records the actual source bytes; `source_dirty=true` means they
cannot be identified by the source commit alone. Local work and remote main
may differ; this snapshot does not claim to update the complete project's main.

The complete project retains MATLAB/GMMNLSE, controlled execution, scans,
between-design resume, database and CSV/MAT export, research/training tools and
historical models. This lightweight clone contains none of those execution
backends and cannot propagate a field. Solver validation belongs in the full
project with its existing authorization and budget controls.

The owner authorized this minimal code/model snapshot for GitHub publication.
That authorization does not grant a blanket license for third-party material
or authorize a cloud deployment. Browser review is still pending; no live link
is claimed. Historical model-card review metadata is retained unchanged.
