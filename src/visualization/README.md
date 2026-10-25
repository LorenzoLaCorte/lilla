# Local repeater explorer

A local Streamlit/Plotly workbench with single runs, fixed-tree parameter grids,
and bounded Bayesian or enumeration searches. Physics comes from the existing
Pauli–ADC backend; [the brief](../../visualization-tool.md) describes the interface.

**Deploy online:** use the [Render deploy button and three-step setup](DEPLOY.md).
The deployment installs only the CPU app dependencies; a persistent results
directory can be configured with `LILLA_RESULTS_DIR`.

From the repository root:

```bash
source venv/bin/activate
# For a fresh environment, install requirements.txt first.
python -m pip install -r requirements-viz.txt
python -m streamlit run src/visualization/app.py --server.address 127.0.0.1
```

These local commands bind to localhost; telemetry is disabled in `.streamlit/config.toml`.
After installation no network access is needed. Use `.venv` if that is your
Python environment's directory.

## Use

Load a scenario/configuration or start with the four-link balanced preset.
Choose the initial-state mode, link count and **Tree shape** slider: −1 is the
fully left-unbalanced tree, 0 is balanced, and +1 is fully right-unbalanced.
The discrete positions retain all ordered trees and their mirrors. For sizes
with tied balanced shapes, the center is a deterministic representative; a
few repeated balanced positions keep the midpoint selectable. Search uses only
unique physical trees, and saved configurations retain the actual tree.

Every node shows its distillation count inside a circle, following the protocol
tree notation: leaves are elementary links, parents are swap outputs, and the
root is the end-to-end link. Click a number to edit it directly; Tab and arrow
keys also work. Link spans appear on hover. The tree has no zoom, pan or toolbar,
and no separate counter fields. Edits update the draft without running a simulation.
The editor uses bundled local HTML/JavaScript; no frontend build or CDN is needed.
The tree space is limited to 1–8 elementary links to keep enumeration practical.

The main controls cover fidelity, hardware probabilities, T1/T2, shared cutoff,
horizon and global Bell acceptance. Full real-X dynamics and T1/T2 memory
controls are the defaults. Per-link/node tables, direct rates, channel twirling,
per-operation cutoffs and search settings are inside **Advanced settings**.
Each additional distillation round gets a separate operation ID: `dist:0:3`,
`dist:0:3:2`, etc., so cutoff overrides remain attached to individual rounds.

The app's starting values live in `presets/default.json`: F0=0.98, p_swap=0.5,
p_gen=0.02, T1=2000, T2=1000 and horizon=10000. `config.DEFAULT` retains the
older seed used by existing analysis scripts; imported version-1 configurations
keep their submitted values.

- **Single run:** press **Run**. Editing controls, selecting distributions or
  inspecting a result never calls the simulator.
- **Parameter sweep:** choose one to three parameters, bounds and point counts.
  The tree stays fixed. One axis produces curves, two produce heatmaps, and
  three produce 3D scatter plots colored by each metric. The default is 15
  points per axis with a 500-evaluation ceiling; reduce points or explicitly
  change the ceiling in Advanced settings. Invalid physical points remain in
  the history. The optional T2=2T1 constraint varies both lifetimes together.
- **Bayesian optimization:** search tree, shared cutoff, distillation placement
  and/or Bell acceptance at fixed hardware and memory. Defaults are 20 trials,
  including five initial trials, over tree and cutoff. Advanced settings expose
  bounds, seed, maximum rounds and bounded enumeration. The GP uses the existing
  left-balanced-right tree ordering and `skopt` LCB machinery with the real-X objective. Budget
  counts trials including duplicate proposals, which reuse cached results.

Sweeps/searches perform one evaluation per UI rerun. **Cancel remaining trials**
stops between evaluations and keeps completed results. Best-found selection uses
only covered runs. Use **Inspect trial** for any completed point, including
under-covered results; 1D/2D plots also support point selection. For 3D views use
this selector. The selected configuration stays attached to its plots even
when the editor changes.

The main result window shows SKR, SKF, mean waiting time, mean Φ+ fidelity and
coverage, followed by completion PMF and Φ+ fidelity. The fidelity histogram
weights states by completion probability. Diagnostics and additional plots are
collapsed under **More visualizations**; overlays and exports are under
**Saved runs / Compare**.

## CLI examples

All modes use the same evaluator as the UI. These small examples are quicker
than the default scenario:

```bash
python -m src.visualization.cli --config src/visualization/examples/smoke.json
python -m src.visualization.cli --config src/visualization/examples/smoke.json --study src/visualization/examples/sweep.json
python -m src.visualization.cli --config src/visualization/examples/smoke.json --study src/visualization/examples/optimize.json
```

Also supplied: `sweep-2d.json`, `sweep-3d.json` and `enumerate.json`. Edit these
ordinary JSON requests to change bounds, point counts, trial budget or seed.
`--output PATH` changes the save directory. Ctrl+C preserves completed study
trials and exits with status 130.

Results go to **`results/visualization/`**, or **`LILLA_RESULTS_DIR`** when set. Each immutable `run-*` bundle holds
configuration, source metadata, diagnostics and full-resolution NPZ arrays.
Each `study-*` directory holds a checkpointed `history.json` with the actual
candidate configurations/run paths, plus `trials.csv`. The CLI also writes
self-contained HTML plots. The app offers JSON, CSV, NPZ and HTML downloads.
Reload a run through **Load a scenario or configuration**. Arrays load without
pickle. No earlier run is overwritten.

## Numerical conventions and checks

T2≤2T1; equality means pure relaxation. Times are attempt slots and rates are
bits/slot under the existing pooled BB84 convention. No extra sifting or Bell
acceptance factor is applied. The mean is the repository's finite-horizon
estimate. Low coverage is provisional; **Run at 2× horizon** saves a separate
check. Horizon growth is always explicit. Material state/probability failures
are reported, never broadly clipped; this includes numerical tail failures that
can arise for extreme near-deterministic repeated distillation.

```bash
source venv/bin/activate
python -m pytest -q src/tests/visualization src/tests/analysis src/tests/pauli_adc/test_numeric.py src/tests/pauli_adc/test_metrics.py src/tests/pauli_adc/test_protocol_metamorphic.py
python -m compileall -q src/visualization
node --check src/visualization/tree_component/editor.js
git diff --check
```

Checks cover backend parity, physical mapping, centered tree ordering/mirrors,
multiple-round cutoff alignment, all three grid dimensions, seeded search,
failed/provisional trials, cancellation checkpoints, weighted fidelity plots,
CLI examples and Streamlit submission/export/reload without automatic simulation.
Tree-editor checks cover inline edits reaching Run and saved configurations,
nonnegative integer validation, and stale events after changing tree shape.

Verified locally: **117 tests passed** (9.69 s). The GP emitted one ordinary
duplicate-proposal warning and chose another point. Compilation and diff checks
passed. An isolated headless Chrome check exercised the actual local app:
seven inline node inputs, edits 2 → 0 → 3 acknowledged by Streamlit, negative
counts rejected, and no tree toolbar. The localhost health endpoint returned 200.

The requested default at horizon 10000 took 0.049 s in the evaluator: coverage
0.999999999998, mean wait 431.542 slots, mean Φ+ fidelity 0.726959, SKR=0.
The small single-run example gave SKR=0.215706 bits/slot in 0.027 s. The supplied
1D/2D/3D examples completed 3/4/8 points in 0.089/0.120/0.239 s; the four-trial
Bayesian and enumeration examples took 0.394/0.121 s. These are measured local
smoke runs, not performance guarantees. Their bundles are under
`results/visualization/working-examples/`; the default bundle is directly under
`results/visualization/`. Other hardware/protocols can take substantially longer.
