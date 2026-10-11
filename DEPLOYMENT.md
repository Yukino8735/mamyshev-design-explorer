# Generated single-model Demo snapshot

This ordinary-file snapshot was exported from the main project's public profile.
It contains only `demo_10k_v1`. The manifest records
the actual source-file hashes, profile version and source Git identity; HEAD
alone does not identify uncommitted source. The current owner instruction
authorizes GitHub code publication of the checked minimal snapshot; it does
not authorize Community Cloud deployment or certify browser behavior.

## Runtime

Run from this snapshot's root with the existing compatible Python 3.11
environment and the exact root `requirements.txt`:

```sh
python -m streamlit run demo/app.py
```

Use the project owner's verified `conda run -n mamyshev-ml --no-capture-output
python` prefix on their Mac. Do not install or bypass trusted dependency/model
checks as part of a review. A manually started server stops with Ctrl-C in its
own terminal; export and component checking do not start a service.

The current app and CLI accept only `demo_10k_v1`. Foreign session results and
model IDs are rejected; explicitly load the fixed-hardware example to reset.
`st.cache_data(scope="session")` remains enabled. The complete main project
separately retains historical loaders, models and research records.

## Browser review still required

1. In **Design workbench**, load **Load fixed-hardware example**. Confirm no
   version selector, then compare a selected design with a different baseline.
2. Rank bounded predicted candidates, select at most three and **Export for
   solver validation**. Inspect the real model ID, zero allowance and null
   real targets in the downloaded request.
3. In **Parameter response**, use bandwidth 0.5–8 nm, nine points, pump 4 W,
   offset 10 nm, gain length 4 m. Then add offset 2–16 nm as a second nine-point
   axis. Inspect line/rect geometry, numeric axes, tooltips, baseline and best
   predicted-feasible markers for all three targets; no readonly-property error.
4. Change inputs or constraints without submitting: old results/downloads
   disappear. Resubmit and compare downloads with the current table; include
   a no-feasible case.
5. **Method & data** visibly presents Data & validation before technical details.
   **MATLAB data pipeline** offers a request download, not a propagation button.

AppTest establishes component behavior, not browser rendering or Cloud multiuser
acceptance. Cloud/Linux build and anonymous access remain pending.

## Maintenance and future publication

Modify the complete main project, run affected tests, then execute its
`scripts/export_demo.py --output <new-directory>`. The source contains the full
public profile/templates, trusted model assets and authoritative pure helpers.
The exporter does not require this snapshot or private historical data. It
refuses an existing target and emits no Git metadata. Edit neither generated
thin modules nor the frozen internal model manifest.

The [complete source project](https://github.com/Yukino8735/Mamyshev-Oscillator-Simulation)
remains the maintenance source. Repository publication is performed separately
from export and verified against its actual commit. A cloud app still requires
its own authorization; the live URL remains pending.

| Platform setting | Intended value, pending actual review |
|---|---|
| Main file | `demo/app.py` |
| Python | Request 3.11; actual Advanced settings availability unverified |
| Dependencies | Root `requirements.txt` only |
| Secrets | Empty |
| Port/lifecycle | Managed by the deployment platform |

Account login, OAuth and 2FA belong in the platform UI, never source or logs.
Model historical pending-review metadata is preserved and is not publication
approval. Full MATLAB execution requires the main project and separate budget.

## Component checks

`checks/presentation_smoke.py` is a reusable single-model checker. Reference
captures/results are supplied through private paths and are not distributed.
Run it serially under the main project's existing process monitor and the
current task's explicit budget. No checks run automatically at app startup.
