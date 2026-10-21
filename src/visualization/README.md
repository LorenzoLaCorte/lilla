# Local Pauli–ADC explorer

This implements the [MVP brief](../../visualization-tool.md) using the existing
Pauli–ADC backend. Scientific context is in [analysis.md](../../analysis.md)
and the Pauli–ADC section of the Overleaf notes. Neither file is needed at runtime.
Core physics is unchanged.

## Setup and launch

Run these commands from the repository root. For a fresh environment, first run:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-viz.txt
```

For the existing environment, activate it and install only the optional requirements.
Use `.venv` in place of `venv` if that is your environment directory.
`requirements-viz.txt` pins Streamlit 1.41.1, compatible with the repository's
Python 3.10 environment, NumPy 2.0.0, pandas 2.2.2 and Plotly 5.22.0.
It does not upgrade the scientific dependencies.

```bash
source venv/bin/activate
python -m streamlit run src/visualization/app.py --server.address 127.0.0.1
```

Open `http://127.0.0.1:8501`. Repository `.streamlit/config.toml` disables usage
statistics and defaults to localhost. Evaluation, plotting, persistence and HTML
exports work offline after installation. No cloud account, external assets or
Overleaf access is used. Stop the app with Ctrl-C.

## Working example

The initial form is `presets/default.json`: four links, balanced ordered tree,
generation probability 0.2, swap hardware success 0.9, Werner fidelity 0.98,
T1=2000 and T2=1000 slots, all corrected Bell outcomes, no cutoff, and T=1024.
Submit **Preview configuration** to inspect the compiled operations and derived
node rates, then **Evaluate**. The operations are `s0,s2,s1`, with IDs
`swap:0:1`, `swap:2:3`, `swap:0:3`. Each node has
`(GammaX,GammaY,GammaZ)=(0.00075,0.00075,0)` and kappa=0.0005 per slot.

Reproduce it through the same service from the CLI:

```bash
source venv/bin/activate
python -m src.visualization.cli --config src/visualization/presets/default.json --output results/visualization
```

Measured on this workspace on 2026-10-05 with Python 3.10.12:

| Quantity | Default result |
|---|---:|
| Covered probability | 0.999999999999732 |
| Repository finite-horizon mean | 12.274101616471224 slots |
| Observed conditional mean | 12.274101616196784 slots |
| Pooled secret fraction | 0.33668136114768227 |
| Pooled key rate | 0.02743022435922104 bits/slot |
| Average Phi+ fidelity | 0.9067255182834015 |
| eX / eZ | 0.06435087812049628 / 0.05784720719220443 |
| Service wall time, one uncached CLI evaluation | 0.0267 seconds |

This is a smoke example, not a runtime guarantee or optimization result.
The CLI prints the exact newly created bundle path and runtime. Bundles under
`results/visualization/` are ignored by the existing repository ignore rules.
Each bundle includes `run.json`, `config.json`, compressed `arrays.npz`, and
the CLI additionally writes `per-time.csv`, `summary.csv`, and `figures.html`.
The HTML embeds Plotly, so it can be opened without network access.
This workspace's smoke bundle is
`results/visualization/run-500dddfa212c4d5cb774248a6ea5584c-fe5ebf51/`;
use **Reload result** to inspect it in the app. The CLI generates a fresh bundle
when reproducing the example.

## Controls and interpretation

Structural controls refresh the available form fields without simulating:
choose 1–8 elementary links, an ordered tree, homogeneous or per-link/per-node
tables, initial-state mode, and memory mode. This small-workbench link limit
bounds the ordered-tree dropdown; mirror trees are retained. All simulation
inputs are submitted with the form. Only **Evaluate** and explicit re-evaluation
actions invoke the backend. Chart zoom, log-axis changes, exports, reloads and
ordinary reruns reuse saved results. Failed evaluations retain the last success.
The [Streamlit form API](https://docs.streamlit.io/develop/api-reference/execution-flow/st.form)
documents batching; charts use its
[Plotly integration](https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart).

Choose a scenario preset: default, no memory noise, pure dephasing, one slow
link, or one poor memory. Protocol variants provide root distillation, a shared
cutoff of 10 slots, both together, Phi pair, and Psi pair. **Custom** uses the
form's selected recurrence locations, cutoff and outcomes. These variants
describe candidates, with no promised improvement.

All physical inputs follow the version-1 JSON schema in the brief and default
preset. JSON `null` means an infinite lifetime or no cutoff. In node tables,
blank lifetime cells mean infinite lifetime. Advanced controls accept Bell
populations in `(phi+,psi+,psi-,phi-)` order or normalized, physical real-X
coordinates in `(II,IZ,ZI,ZZ,XX,YY)` order. Advanced Pauli rates are mode decay
rates constrained by the CP triangle inequalities.

Distillation consumes two independent copies of the selected subtree, at most
one round at each location, including leaves and root. Operation cutoffs are
attached to stable IDs, including distillation. For example, this protocol
object adds root distillation and assigns two overrides:

```json
{
  "tree": [[0, 1], [2, 3]],
  "distillation_rounds": {"0:3": 1},
  "memory_cutoff_slots": null,
  "cutoff_overrides": {"swap:0:3": 10, "dist:0:3": null},
  "bell_outcomes": "phi_pair"
}
```

The operation sequence becomes `s0,s2,s1,d3`; no-cutoff entries are compiled
to the current horizon, while `null` remains in the high-level config.
Cutoffs accept arrival differences `abs(t1-t2) <= cutoff`; they do not bound
total duration. Hardware success and quantum outcome acceptance remain
separate, and the backend includes destructive retries. No extra acceptance
factor is applied to the rate.

Times are generation-attempt slots. Optional `slot_duration_seconds` converts
times to seconds and rates to bits/second, without changing dynamics. Rates
use the repository's BB84 convention without an extra basis-sifting factor.
T1/T2 mapping is kappa=1/T1 and gamma_phi=1/T2−kappa/2, requiring T2≤2T1.
The memory channel applies AD followed by Pauli noise over each waiting interval;
general Pauli rates need not represent simultaneous generator evolution.
The pure Z dephasing and AD channels in the T1/T2 mapping commute.

The two prediction models share the real-X evaluator and all protocol maps:
full dynamics preserves the physical rates, and local channel twirling changes
node rates to `(GammaX+kappa/2,GammaY+kappa/2,GammaZ+kappa)` with damping zero.
The original physical config is always saved. **Re-evaluate in full model**
starts from it. General real-X initial states can retain local biases under
the channel approximation; this is not a Bell/Werner state-projection baseline.

The headline objective uses completion-mass-weighted pooled coordinates. The
returned backend rows are already normalized; they are not divided by the PMF.
The State tab shows correlators, local biases, Bell populations, the pooled or
completion-time-selected density matrix, trace and minimum eigenvalue.
Time-separated processing is an ideal asymptotic diagnostic of the same run;
the headline rate remains pooled.

Quality plots mask `pmf <= 1e-12` for display. All positive bins remain in
metrics, NPZ and CSV exports; empty rows have no conditional state values in CSV.
Material probability/physicality errors and nonfinite outputs fail evaluation.
Cleanup only permits roundoff at tolerance `1e-10`: bin-0 residue, negative mass,
unit-mass excess, and normalized-state endpoint/PSD roundoff. Any cleanup is
recorded under run diagnostics. Backend normalization also applies its existing
physicality checks. Undefined no-completion metrics are JSON `null`, with status
`no_completion`, and rate/coverage zero.

Low-coverage results are provisional. **Re-evaluate at 2× horizon** saves another
run, rebuilds no-cutoff entries and reports rate/coverage changes. The repository
mean includes an unresolved-tail finite-horizon estimate; the observed mean is
conditional on completion inside the horizon. Neither certifies an infinite-
horizon mean. High coverage alone does not establish a reliable close ranking.

## Saved runs and manual comparison

Every successful explicit evaluation saves a new atomic JSON/NPZ bundle.
Display names never overwrite another run. **Reload result** restores the
submitted physical configuration, numerical status and arrays, without
simulation. Export the draft or submitted config, per-time CSV, comparison
summary CSV, or embedded HTML figures using buttons outside the form.
The CLI also exports these files. NumPy loading disables pickle.

Select up to 12 saved runs in **Compare**. The table shows model, tree,
distillation, cutoff, outcome policy, rate, waiting time, fidelity, coverage,
horizon and status; a second table highlights configuration differences.
Time overlays use stable run colors and descriptive labels. Mixed unit overlays
use slots. Different hardware is allowed for exploratory overlays.

For controlled selection-loss calculations, selected runs must share the same
expanded physical scenario, units and horizon. Press **Evaluate candidates in
both models** to evaluate the distinct manual candidates under both models.
The app shows pending backend calls first, advances progress per result, saves
partial successes, and refuses a loss calculation for an incomplete matrix.
An eight-result in-memory cache uses expanded inputs, horizon, prediction
model, evaluator options and the relevant source fingerprint. Names, units
metadata and coverage targets are recomputed on cache hits. Disk runs are
history and are never silently inserted into the cache; historical source
differences are shown.

Prediction error is twirled minus full rate for each candidate. Selection loss
is the full-model performance lost by choosing the twirled-model winner within
these candidates. Relative loss is undefined when the best full-model rate is
zero. All near-tied winners are retained with ranges of physical rates/losses,
using `abs(a-b) <= 1e-12 + 1e-3*max(abs(a),abs(b))` bits/slot. This describes
ranking ambiguity, not a certified numerical error bound. Coverage deficiencies
make the comparison provisional.

The MVP includes no optimizer, automatic sweep, Bell/Werner projection schedule,
intermediate-operation traces, adaptive outcomes or cutoffs, online scheduler,
or syndrome/finite-key processing. Those remain the later extensions in the brief.

## Verification

Run the new tests and the relevant short backend checks:

```bash
source venv/bin/activate
python -m pytest -q src/tests/visualization src/tests/pauli_adc/test_numeric.py src/tests/pauli_adc/test_metrics.py src/tests/pauli_adc/test_protocol_metamorphic.py src/tests/pauli_adc/test_sequential_efficient_backend.py -k 'not avoids_quadratic'
python -m compileall -q src/visualization
python -m pip check
```

The efficient-backend enumeration/scaling check is explicitly deselected;
the optimizer and unrelated campaigns are not run. Tests include physical
mapping/rejection, ordered operation replay, cutoff alignment, direct/service/CLI
parity under heterogeneous noise and postselection, channel/state-twirl controls,
deterministic and zero-completion controls, biased-input corrected swapping,
horizon/cache behavior, storage/exports, 12-candidate comparisons, near ties,
zero denominators and incomplete matrices. Streamlit's test harness exercises
submission, previews, display reruns, failures, JSON import, exports, reload,
heterogeneous controls, full-model conversion and comparison partial failures.
The UI tests skip when the optional Streamlit dependency is absent.

The combined test command above passed **105 tests with 1 deselected** in
5.41 seconds on this workspace. Python compilation, `pip check`, and
`git diff --check` also passed.

Local startup was checked using the documented command and
`http://127.0.0.1:8501/_stcore/health` returned HTTP 200 with `ok`.
The CLI smoke example produced the values above; dependency compatibility and
Python compilation checks passed.
