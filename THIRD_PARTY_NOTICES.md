# Attribution and publication review

This notice records provenance, not a grant of rights or a new blanket license.
The owner authorized publication of this reviewed project code/model snapshot.
This is not a blanket license for third-party material. Existing model-card
historical review metadata remains unchanged.

## Project and model

The interface, trusted inference loader, input validation and request planning
come from the complete Mamyshev project. Selected pure helper functions and
their required constants/imports are generated from its authoritative source;
the complete scientific backend remains in that project. Rights for project
contributions have not been independently established by this engineering task.

The sole trusted model, `demo_10k_v1`, is one frozen fit on 10,000
repository-guided synthetic designs. Model SHA-256:
`9885e8f258fa10de0a69844b70fa72242fcc64267d9db1e6c07dd4bd4609b6d0`.
The required byte-identical `models_v4.py` class SHA-256 is
`46f9779b878d195b601930446cce3ec5335f694ef219f28c2ae728a353590b11`.
Calibration, original model card, aggregated metrics and internal manifest are
preserved. Any publication approval must identify the outer snapshot separately;
it must not erase historical pending-review metadata or change calibrated values.

The label source is not measured or newly executed GMMNLSE truth. Raw training,
test and historical candidate rows are excluded. Integrity hashes identify the
trusted local assets; they are neither arbitrary-pickle safety certification
nor permission to redistribute underlying material.

## Runtime dependencies

Streamlit, pandas, NumPy, SciPy, scikit-learn, joblib, threadpoolctl, Altair and
Apache Arrow/PyArrow are separately distributed dependencies. Versions appear
in root `requirements.txt`; packages and transitive dependencies retain their
maintainers' authorship and license notices. They are not vendored or relicensed
here. The custom model's use of scikit-learn does not establish rights to the
model, training recipe, historical references or outputs.

## Exclusions

No original GMMNLSE tree, MMTools, MATLAB auxiliary/material databases,
third-party solver license bundles, papers, theses, historical images, raw
waveforms, old model weights or individual candidate confirmations are included.
This snapshot does not resolve their separate publication review. Runtime plots
are current frozen-model predictions with source and limitations in the UI.

The exact manifest identifies the reviewed project code and necessary 10k
asset set. Local export and successful component checks are not themselves
publication approval or physical validation; owner authorization is separate.
