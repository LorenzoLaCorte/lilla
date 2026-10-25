# Local repeater explorer: concise build brief

Build a fresh local **Streamlit + Plotly** app using the existing Pauli–ADC backend. Make it quick to explore: protocol first, a few physical controls, clear results. Keep detailed controls and diagnostics collapsed. Use `analysis.md`, the existing backend/tests, and `~/Research/lilla-overleaf/main.tex` as context. Preserve unrelated changes and research notes.

## Main interface, in order

1. Keep **Load a scenario or configuration**: presets, JSON upload/paste, and saved-run reload.
2. Choose **Single run**, **Parameter sweep**, or **Bayesian optimization**.
3. Put **State mode** at the top of the editor: initial-state input as **Werner fidelity** (default), **Bell weights**, or **Real-X coordinates**. This selects the input representation; dynamics default to full real-X in every mode.
4. Make the **protocol tree** the first substantial interactive area. Fill elementary-link counter (default is 4), then move a horizontal **Tree shape** slider: fully left-unbalanced at −1, balanced at 0, fully right-unbalanced at +1. Update the diagram immediately. Edit distillation-round counts directly inside the nodes; no separate counter fields. Default: zero rounds everywhere.
5. Show the basic physical controls below the tree, in this order:

   | Control | Default |
   |---|---|
   | Initial fidelity F0 | 0.98 |
   | Swap hardware success | 0.5 |
   | Distillation hardware success | 1 |
   | Generation success | 0.02 |
   | T1 (slots) | 2000 |
   | T2 (slots) | 1000 |
   | Shared memory cutoff (slots) | No cutoff |
   | Horizon | 10000 |
   | Global Bell-outcome acceptance | All corrected |

   Fidelity is editable for Werner inputs; other state modes reveal their own inputs instead. Keep ordinary controls numeric, with explicit No cutoff / Infinite lifetime options, rather than requiring JSON.
6. One primary action: **Run**, **Run sweep**, or **Optimize**. Show progress, elapsed time, and the latest successful result. Editing inputs or interacting with plots must never launch a simulation.

Use a compact layout with minimal explanatory text. Avoid operation tables, raw JSON, backend names, and long captions on the main screen. Keep the existing four-link balanced scenario as the starting point.

### Tree shape slider and notation

Enumerate ordered trees using the existing tree library. Arrange a deterministic discrete path from the fully left-leaning comb, through a balanced representative at the exact midpoint, to the fully right-leaning comb. Use structural imbalance (differences in child leaf counts) to order each arm; do not expose a symmetry score. Keep mirror trees distinct and retain every ordered tree. For sizes with several balanced shapes, pick the center deterministically and keep alternatives nearby. A balanced shape may repeat to give an exact selectable center; optimization must deduplicate physical trees. Disable the slider when only one tree exists. Save the actual tree, not just its slider position.

Draw a static binary tree with an **editable number inside every node**, following the repeater-protocol TikZ notation: leaves are freshly generated links; a parent is the output of swapping its two children; the root is the end-to-end link. The number is the recurrence-distillation count on that link after preparation, before its parent's swap. For example, tree `[[0,1],2]` can show root 2, left internal node 0, and all three leaves 1. Support clicking and keyboard editing of nonnegative integers. No zoom, pan, camera controls or toolbar. Keep link spans in hover text rather than crowded node labels. Edits update the draft immediately without evaluating it. Deep-copy library trees before adding rounds; keep operation sequences in advanced settings.

The circles themselves are the inputs, including leaves and root: click the displayed number and type, or use Tab/arrow keys. Do not duplicate these inputs in a form below or beside the tree. Use a local Streamlit component with SVG edges and HTML number inputs (see `src/visualization/tree_editor.py` and `tree_component/`); no frontend build or CDN. This static-tree requirement applies only to the protocol diagram; keep result plots interactive. Show selected-run protocol snapshots with read-only node numbers. Preserve edits through ordinary rerenders, use them in Run and saved configurations, and ignore delayed edits from a previous tree shape or loaded configuration.

## Three execution modes

- **Single run:** evaluate the displayed configuration once.
- **Parameter sweep:** vary parameter(s) for the same tree; keep everything else fixed. Choose range and point count (default 15), then explicitly run. Offer linear/log spacing and an optional T2=2T1 constraint for pure-relaxation sweeps. Mark invalid points rather than silently adjusting physical inputs. Plot SKR, SKF, average waiting time, and average Phi+ fidelity against the swept parameter; clicking/selecting a point opens its full results. No automatic multidimensional campaigns. Varying one parameter produces a simple plot. Varying two parameters produces a heatmap. Varying three produces a 3D plot.
- **Optimization:** maximize SKR at fixed hardware, initial state, memory, and horizon. Get inspired from bruteforce and Bayesian optimization logic in `src/optimization`.

## Main results window

Always show these five metrics for the selected completed run (the best eligible candidate initially in sweep/search mode):

**SKR (bits/slot) · SKF · Average waiting time (slots) · Average Φ+ fidelity · Covered probability**

Remove “pooled” from display labels; retain the existing pooled BB84 calculation. Directly underneath, show two prominent interactive plots:

1. **Completion-time PMF**, with an optional log-y toggle.
2. **Φ+ fidelity**, showing F(t) versus completion time and offering a PMF-weighted fidelity-histogram view. Weight by completion probability, never equally by time bin.

Keep these plots visible without opening diagnostics. Preserve hover, zoom, and saved-run overlays. Hide empty/negligible-probability bins in quality plots only; retain full-resolution data for calculations/export.

Use a collapsed **More visualizations** area for CDF, eX/eZ, local biases, XX/YY/ZZ correlations, Bell populations, density-matrix heatmap, time-resolved secret fractions, and detailed numerical diagnostics. Keep saved-run comparisons and exports available in a collapsed **Saved runs / Compare** area. Preserve useful visualizations without filling the initial screen with them.

## Advanced settings

Collapsed by default:

- Prediction model: **Full real-X** (default) or **Local Pauli channel twirl**. Keep these separate from initial-state mode; do not add unsupported Bell/Werner projection models.
- Memory parameterization: **T1/T2** (default) or direct Pauli–ADC rates; show derived rates here.
- Per-link initial states/generation probabilities and per-node memory tables; homogeneous inputs remain the default.
- Per-operation cutoffs, advanced Φ+ acceptance, operation sequence/table, raw configuration, optional seconds-per-slot, coverage target (0.999), and search settings.

## Scientific and implementation essentials

Reuse `src/visualization/{config,protocols,evaluate,metrics,storage}.py` where useful; keep non-UI logic shared by all three modes and the CLI. Keep existing configuration imports compatible. Use `src/visualization/presets/default.json` for the complete starting schema. No separate web server or frontend stack.

- Explicitly use `RepeaterChainEvaluation(state_type=RealXState, efficient=True, use_fft=True, use_gpu=False, w_twirling=False)` and `asymmetric_heterogeneous_protocol`; broadcast homogeneous inputs. For S links there are S+1 memory nodes. Use existing protocol serialization and cutoff alignment, not new physics implementations.
- Memory mapping: κ=1/T1, Γφ=1/T2−κ/2, Pauli mode rates `(Γφ, Γφ, 0)`, damping rate κ. Infinite lifetimes have zero reciprocal. Require T2≤2T1; **T2=2T1 means pure amplitude damping**. Validate probabilities, physical states, and direct rates through existing validators.
- Werner F0 maps to Bell weights `[F0, (1−F0)/3, (1−F0)/3, (1−F0)/3]`. Coordinate order is `(II, IZ, ZI, ZZ, XX, YY)`; Bell order is `(Φ+, Ψ+, Ψ−, Φ−)`.
- Backend `x[t]` is already normalized conditional on completion; never divide by PMF again. For C=Σp(t), use x̄=Σp(t)x(t)/C, eX=(1−x̄XX)/2, eZ=(1−x̄ZZ)/2, SKF=max(0,1−h₂(eX)−h₂(eZ)), and F̄=(1+x̄ZZ+x̄XX−x̄YY)/4. Use existing `secret_key_rate` and mean-wait helpers with extrapolation disabled; verify SKR=SKF/mean wait. Add no sifting or guessed Bell-acceptance factor: retries are already included.
- Local channel twirling uses `(ΓX+κ/2, ΓY+κ/2, ΓZ+κ)` with damping set to zero. Preserve the original physical configuration; do not confuse channel twirling with state projection.
- Keep destructive retries and memory-cutoff semantics. Compile No cutoff for each requested horizon; do not interpret cutoff as total protocol duration.
- If coverage is insufficient, mark metrics provisional and offer **Run at 2× horizon**. Exclude provisional/failed trials from a confident best-candidate claim; show when no eligible winner exists. Missing mass is unresolved completion, not Bell rejection. No completion gives SKR=0 and undefined state metrics, exported as null. Validate outputs; do not hide failures with broad clipping.
- Save configurations, results, arrays, diagnostics, and sweep/search history under `results/visualization/`. Allow JSON import/export and CSV/NPZ downloads. Keep original run settings attached to results; editing controls must not relabel old plots. Cache and reload safely without pickle.

## Delivery

Use the repository Python environment and optional visualization dependencies. Bind Streamlit to localhost and work offline after installation. Keep a shared CLI entry point for single runs, sweeps, and optimization; document one small working example per mode and the save location.

Run focused checks for physical mapping, tree ordering/mirror selection, distillation/cutoff alignment, direct-backend parity, pooled metrics, persistence, sweep isolation, and reproducible budgeted search. Verify inline node edits reach Run and saved configurations, reject invalid counts, and cannot replay after a tree/configuration change. Check the editor in a browser: editable nodes, no duplicate fields or tree toolbar. Verify that UI interactions never simulate automatically. Use short backend tests plus tiny sweep/search smoke checks; no long campaigns. Document commands and observed runtimes.

Defer state-projection baselines, per-swap/history-dependent acceptance, intermediate-operation tracing, physical twirling actions, and multidimensional regime maps. This version should make a small experiment easy to set up and inspect.

## Deployment

Keep the app deployable through a root `render.yaml` and a Render deploy button,
with three short setup steps. Use a self-contained CPU-only dependency file,
Python 3.11, and a shared launch script that supports the host's `PORT` and an
explicit public bind address while defaulting to localhost locally. Expose the
Streamlit health endpoint and bundle the editable tree component without a
frontend build. Preserve the local interface and all three execution modes.

Support `LILLA_RESULTS_DIR` in the app and CLI, retaining the existing local
default and the CLI's explicit output override. Start with Render's free plan;
document its temporary storage and how to mount a paid persistent disk. State
that hosted visitors share one saved-run library. Include a small working
example and verify the app in an isolated environment with only its deployment
dependencies. See `src/visualization/DEPLOY.md` for the implemented setup.
