# Local Pauli–ADC explorer: implementation brief

Build the MVP specified below in this repository. Complete the implementation, focused tests, setup instructions, and a working local example. The later extensions are deliberately outside the MVP.

## Purpose and scope

Build a small local research workbench for **one explicit repeater-protocol evaluation at a time** using the existing Pauli–ADC / real-X backend. I want to enter physically meaningful parameters, choose a protocol, see its waiting-time and state-quality distributions, and compare a few saved alternatives to develop hypotheses for [analysis.md](analysis.md).

Use the current code and `~/Research/lilla-overleaf/main.tex` as context, especially the section labeled `sec:pauli-adc` and its noise, swapping, distillation, and efficient-evaluation subsections. The Overleaf notes explain the mathematics; inspect the executable code for actual API behavior. Do not edit the Overleaf project or existing research notes. Preserve unrelated worktree changes.

The MVP must support:

1. Homogeneous and heterogeneous chains, with high-level relaxation/dephasing controls.
2. A selectable ordered swap tree, distillation at selected tree locations, and memory-time cutoffs.
3. Fixed Bell-outcome acceptance policies, including their retry costs.
4. Single-run diagnostics and local persistence, export, and reload.
5. Manual candidate comparisons, including full dynamics versus local channel twirling.

There is no optimizer, automatic parameter sweep, cloud service, authentication, database, or general scheduling engine in this version. Do not describe a manually selected best candidate as a globally optimal protocol. The tool is for finding interesting examples before launching a research campaign.

## Minimal application

Use **Streamlit and Plotly**, with ordinary Python modules containing all non-UI logic. Plotly is already in `requirements.txt`; add Streamlit as an optional visualization dependency in `requirements-viz.txt`, selecting and pinning a release compatible with the repository's Python environment. Avoid upgrading the scientific stack just to build the UI. No separate JavaScript project or API server.

Use a form with an explicit **Evaluate** button. Streamlit forms batch input changes on submission; chart interactions and ordinary widget reruns must not trigger another simulation. Keep exports and other buttons outside the form. See the official [form API](https://docs.streamlit.io/develop/api-reference/execution-flow/st.form) and [Plotly integration](https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart).

Run from the repository root, binding only to localhost:

```bash
source venv/bin/activate
python -m pip install -r requirements-viz.txt
python -m streamlit run src/visualization/app.py --server.address 127.0.0.1
```

Use `.venv` instead if that is the actual environment. Document installation of the existing base requirements for a fresh environment. Disable Streamlit usage-statistics collection for this local application. Once dependencies are installed, evaluation and plotting must work without network access, external assets, or Overleaf access.

Suggested organization, allowing small adjustments where they simplify implementation:

```text
src/visualization/
  __init__.py
  app.py             # forms, session state, results, comparison table
  config.py          # versioned schema, validation, physical parameter conversion
  protocols.py       # tree selection, operation IDs, cutoff alignment
  evaluate.py        # backend adapter, results, coverage diagnostics
  metrics.py         # scientific quantities derived from backend outputs
  plots.py           # Plotly figures and protocol-tree diagram
  storage.py         # JSON/NPZ/CSV persistence and cache identity
  cli.py             # evaluate a saved config without the UI
  presets/           # small JSON examples
src/tests/visualization/
requirements-viz.txt
```

Keep Streamlit imports out of the evaluation/configuration modules. A simple synchronous evaluator with a spinner and elapsed time is enough; do not build a background task service. Retain the last successful result when an evaluation fails. Store its submitted configuration with it, so editing the form never relabels an old plot as a new run.

## Existing backend contract: reuse this

| Responsibility | Existing implementation |
|---|---|
| Explicit backend selection and evaluation | `src/core/repeater_algorithm.py`: `RepeaterChainEvaluation`, `asymmetric_heterogeneous_protocol` |
| State construction and physicality | `src/core/pauli_adc/state.py`: `RealXState`, `bell_weights_to_coordinates`, `coordinates_to_matrix`, `phi_plus_fidelity_func` |
| Physical operations and channels | `src/core/pauli_adc/protocol_units.py` |
| Direct reference join | `src/core/pauli_adc/swap_join.py` |
| Efficient production join | `src/core/pauli_adc/swap_join_sequential_efficient.py` |
| Swap-tree generation and serialization | `src/types/protocol_types.py`: `generate_swap_space`, `SwapTreeVertex.get_sequence` |
| Heterogeneous input checks | `src/types/repeater_types.py`: `validate_heterogeneous_parameters` |
| Existing pooled BB84 objective | `src/utils/utility_functions.py`: `get_mean_sf`, `get_mean_waiting_time`, `secret_key_rate` |
| Relevant regression tests | `src/tests/pauli_adc/` |

Always select `RealXState` explicitly; automatic inference from `lambdas` selects the Pauli-Fourier backend. Route both homogeneous and heterogeneous configurations through the heterogeneous method by broadcasting scalar inputs. Instantiate a fresh evaluator per uncached call:

```python
evaluator = RepeaterChainEvaluation(
    state_type=RealXState,
    efficient=True,
    use_fft=True,
    use_gpu=False,
    w_twirling=False,
)
pmf, x = evaluator.asymmetric_heterogeneous_protocol(parameters, number_of_segments=S)
```

For `S` elementary links there are `S + 1` memory nodes. Supply:

- `p_gen`: length `S`.
- `lambdas`: shape `(S, 4)`, or `real_x_coordinates`: shape `(S, 6)`; choose one.
- `pauli_mode_decay_rates`: shape `(S + 1, 3)`.
- `amplitude_damping_rate`: shape `(S + 1,)`.
- `protocol`: a tuple of operation strings.
- `cutoff`: an integer per operation, including distillation operations.
- `p_swap`, `p_distillation`, `bell_outcomes`, `t_trunc`, `cut_type="memory_time"`, and `real_x_werner_twirling=False`.

The return values have shapes `(T,)` and `(T, 6)`, where `T=t_trunc` and time indices are `0,...,T-1`. Generation starts at time 1. `x[t]` is the normalized state conditioned on completion at time `t`; **do not divide it by `pmf[t]` again**. Weighted output is `pmf[:, None] * x`. Empty bins contain placeholder states and must not be presented as physical observations.

The six coordinates are `(II, IZ, ZI, ZZ, XX, YY) = (1, z_R, z_L, c_ZZ, c_XX, c_YY)`. Bell ordering is `(phi+, psi+, psi-, phi-)`. Preserve these conventions in every plot and export.

Do not reimplement convolutions, renewal sums, swap maps, or distillation maps in the frontend. The MVP should normally require no changes to core physics. Existing comments claiming that all asymmetric cutoffs are unsupported are stale for explicitly selected `RealXState`; use the actual implementation and tests.

## Input model and defaults

Use one versioned JSON schema shared by the form, CLI, saved presets, and exports. This is the default preset and the required schema spelling:

```json
{
  "schema_version": 1,
  "name": "Four-link relaxation and dephasing",
  "links": 4,
  "slot_duration_seconds": null,
  "hardware": {
    "p_gen": 0.2,
    "p_swap": 0.9,
    "p_distillation": 1.0
  },
  "initial_state": {"kind": "werner", "fidelity": 0.98},
  "memory": {"kind": "t1_t2", "t1_slots": 2000.0, "t2_slots": 1000.0},
  "protocol": {
    "tree": [[0, 1], [2, 3]],
    "distillation_rounds": {},
    "memory_cutoff_slots": null,
    "cutoff_overrides": {},
    "bell_outcomes": "all_corrected"
  },
  "prediction_model": "full_real_x",
  "numerics": {"t_trunc": 1024, "coverage_target": 0.999}
}
```

### Hardware and memory controls

Show a homogeneous form by default and a toggle revealing a per-link/per-node table. Node labels must distinguish physical nodes `0,...,S` from elementary links `0,...,S-1`. The current model assigns noise per node, shared by memories at that node; independent rates for its two memory qubits are outside scope.

All simulation times are **generation-attempt slots**. `slot_duration_seconds` is optional positive metadata: multiply times by it and divide rates by it for physical units. Changing only this field must not change the dynamics. Label the objective as bits/slot, or bits/second when converted, under the repository's existing BB84 convention; add no unrequested basis-sifting factor.

`p_gen` accepts a scalar or length-`S` list. Hardware success probabilities are scalars. Validate all probabilities as finite numbers in `[0,1]`. Zero success is allowed and must lead to an explicit no-completion result rather than a divide-by-zero error.

Default initial-state mode is Werner fidelity `F0`, accepting a scalar or per-link list. Translate it to Bell populations `[F0, (1-F0)/3, (1-F0)/3, (1-F0)/3]`, not to `w0=F0`. Advanced modes are `{"kind":"bell_weights","values":...}` with shape `(4,)` or `(S,4)`, and `{"kind":"real_x","values":...}` with shape `(6,)` or `(S,6)`. Validate nonnegative normalized Bell weights and use `RealXState` to reject nonphysical real-X inputs. Do not accept arbitrary density matrices outside the supported real-X family.

Default memory mode takes `T1` and `T2` in slots, each scalar or length `S+1`. JSON `null` means infinite lifetime, with zero reciprocal. For each node:

```text
kappa = 1/T1
gamma_phi = 1/T2 - kappa/2
pauli_mode_decay_rates = (gamma_phi, gamma_phi, 0)
amplitude_damping_rate = kappa
```

Reject finite nonpositive lifetimes and `gamma_phi < 0` beyond roundoff tolerance: physical inputs require `T2 <= 2*T1`. Display the derived relaxation and pure-dephasing rates. This makes it possible to hold `T2` fixed and change `T1` without silently changing the intended transverse decay.

Advanced memory mode is `{"kind":"pauli_adc","pauli_mode_decay_rates":...,"amplitude_damping_rate":...}`. Accept a shared triple or per-node triples, and a shared damping rate or per-node rates. Require finite nonnegative rates and the Pauli-mode triangle inequalities `Gamma_X <= Gamma_Y+Gamma_Z` and cyclic permutations. These are mode-decay rates, not Pauli jump probabilities.

The backend applies **AD followed by Pauli noise over each waiting interval**. For general Pauli rates this need not equal simultaneous evolution under the sum of the generators. For AD plus pure Z dephasing above, the channels commute. Preserve this convention, the immediate outcome-dependent correction convention in the implemented maps, and destructive full-reset retries. Tree evaluation is a fixed recursive protocol, not an online swap-ASAP scheduler.

### Protocol controls

Provide a dropdown of ordered binary tree shapes from `generate_swap_space(S)`, including balanced and comb shapes. Draw the selected tree with leaf labels and segment spans. Do not collapse mirror trees: left/right ordering and physical node identity matter under heterogeneous noise and the implemented correction convention.

The JSON tree uses an integer leaf index or a two-element child list. Validate that leaves are exactly `0,...,S-1` in that order. Deep-copy library trees before adding distillation settings.

At each tree location, allow 0 or 1 recurrence rounds, including elementary links and the root. `distillation_rounds` maps an inclusive elementary-link span `"i:j"` to 0 or 1; missing entries mean 0. Distillation consumes two independently prepared copies of that subtree. It does not combine an arbitrary pair of differently prepared subtrees. Keep the backend's control/auxiliary ordering.

Give each operation a stable ID `swap:i:j` or `dist:i:j`, using its output link span. A round occurs after preparing the segment and before its parent's swap. Use `SwapTreeVertex.get_sequence()` and verify the operation-to-ID mapping by replaying active segment slots. For the default tree, the swap sequence is `("s0", "s2", "s1")`. Adding one root distillation appends `"d3"`.

The cutoff form offers a shared nonnegative integer or **No cutoff**, plus optional per-operation overrides keyed by those IDs. An override of JSON `null` also means no cutoff. A memory cutoff accepts arrivals with `abs(t1-t2) <= cutoff`; it is not a bound on total protocol duration. Show the final operation table with ID, code, endpoint nodes, and cutoff before evaluation.

Compile no cutoff to `t_trunc` for this finite-horizon call, ensuring it cannot bind within the horizon. Preserve `null` in the saved high-level config and recompile it when the horizon changes. Reject nonexistent override IDs; never silently truncate or misalign the cutoff vector when distillation settings change.

Offer one **global** `bell_outcomes` choice: `all_corrected`, `phi_pair`, `psi_pair`, or advanced `phi_plus`. `p_swap` is independent hardware success, separate from quantum acceptance. The backend accounts for the accepted branch and failed-attempt waiting time. Never multiply the final rate by a guessed extra factor of 1/2 or 1/4. Per-swap and history-dependent policies are later extensions.

## Prediction models: two in the MVP

Provide exactly these implemented choices, using the same real-X evaluator:

| UI label / schema value | Translation |
|---|---|
| Full real-X / `full_real_x` | Use the configured physical memory channel unchanged. |
| Local Pauli channel twirl / `local_channel_twirl` | Replace each node's rates by `(Gamma_X+kappa/2, Gamma_Y+kappa/2, Gamma_Z+kappa)` and set its damping rate to zero. |

The second choice preserves the local contraction factors and removes the affine translation. Keep initial states, generation probabilities, operation maps, hardware failures, cutoffs, and reset rules fixed. For general real-X initial states this baseline need not produce Bell-diagonal states; label it by the channel approximation, not by a presumed output-state family.

Always preserve the original physical configuration alongside the derived backend parameters. **Re-evaluate in full model** must start from those original physical rates, not try to invert the twirled rates.

Do not add Bell/Werner state-projection labels as aliases of existing backends. Those are different approximations and require projection hooks with a specified schedule. `real_x_werner_twirling=True` only after distillation is not the complete Werner baseline in `analysis.md`. Keep it false in both MVP models.

## Results and plots

Keep a compact page: configuration, run summary, then tabs **Time**, **State**, and **Compare**. Explain quantities with short labels/tooltips. Use Plotly hover and zoom; no decorative dashboard components.

### Summary and numerical reliability

Show pooled secret-key rate, pooled secret fraction, repository mean-wait estimate, average fidelity, covered probability mass, horizon, and wall time. For `C=sum_{t>=1} pmf[t] > 0`, compute:

```text
x_bar = sum_t pmf[t] * x[t] / C
eZ = (1 - x_bar[ZZ]) / 2
eX = (1 - x_bar[XX]) / 2
r_pool = max(0, 1 - h2(eX) - h2(eZ))
F_bar = (1 + x_bar[ZZ] + x_bar[XX] - x_bar[YY]) / 4
R = secret_key_rate(pmf, x, state_type=RealXState, extrapolation=False)
```

Use the existing helper for the headline rate and check the independent displayed quantities agree with it. In particular, `get_mean_sf(..., RealXState)` returns a BB84 expression, not a mean coordinate vector. Never average normalized rows without PMF weights and never use a Werner fidelity-only key formula for real-X states.

For the current non-extrapolating waiting-time helper:

```text
E_repository = [sum_t t*pmf[t] + (1-C)*T] / C
E_observed = sum_t t*pmf[t] / C
```

Label the former as the repository's finite-horizon estimate and the latter, if shown, as conditional on completion inside the horizon. Neither is a certified infinite-horizon mean from a truncated distribution. Do not normalize the PMF to one before calling the objective.

If `C < coverage_target`, mark rate and mean as provisional. Provide **Re-evaluate at 2× horizon**, retaining both runs and reporting rate/coverage changes. No silent horizon growth or arbitrary tail extrapolation. Missing probability is unresolved late completion, not the per-attempt rejection probability. High coverage alone does not certify a precise mean or a close ranking; longer-horizon agreement is useful evidence.

Treat bin 0 as unsupported; check its mass is negligible. Reject materially invalid/NaN outputs. Allow only documented roundoff cleanup (e.g. `1e-10`) and record any cleanup; never hide a mass/physicality failure by broad clipping. For no completion, return rate 0, coverage 0, and undefined conditional state metrics, with a clear message. Export undefined values as JSON `null` with a status, not nonstandard `NaN`/`Infinity` tokens.

### Time tab

Plot completion-time PMF and CDF, followed by `F(t)`, `eX(t)`, and `eZ(t)`. Provide a logarithmic PMF-axis toggle. Mask quality curves where `pmf[t] <= 1e-12` for display only; retain all positive bins in metrics and exports. State the display threshold. Overlay selected saved runs with stable colors and descriptive labels.

### State tab

Plot `z_L(t)`, `z_R(t)`, `c_ZZ(t)`, `c_XX(t)`, and `c_YY(t)`. Offer a selector for the pooled state or a populated completion-time bin, then show its six coordinates and a 4×4 real density-matrix heatmap reconstructed with `coordinates_to_matrix`. A pooled heatmap means `rho(x_bar)`, not an unweighted average over time. Show trace and minimum eigenvalue as diagnostics.

Also show Bell populations in the documented order:

```text
phi+ = (1 + cZZ + cXX - cYY)/4
psi+ = (1 - cZZ + cXX + cYY)/4
psi- = (1 - cZZ - cXX - cYY)/4
phi- = (1 + cZZ - cXX + cYY)/4
```

Call them Bell populations, not a full Bell-diagonal description of a locally biased state. The biases and anisotropy are the quantities that help explain noise-model sensitivity. Fidelity alone is insufficient. Do not call the repository's `distillable_entanglement` surrogate for real-X states; it deliberately rejects this use.

As a small diagnostic computed from the same result, show the completion-time-resolved secret fraction `r(t)` and `sum_t pmf[t]*r(t)/C` next to `r_pool`. Label the former as ideal asymptotic processing separated by completion-time bins. It is not swap-syndrome resolution, physical postselection, a new timing-cutoff protocol, or a finite-key claim. Keep the headline objective pooled.

## Saved runs and controlled comparisons

Save successful evaluations locally under a git-ignored `results/visualization/` directory. A run bundle contains canonical high-level configuration, expanded backend parameters, schema version, timestamp, source revision and dirty status when Git is available, runtime, diagnostics, summary metrics, and compressed NPZ arrays. Use `allow_pickle=False` when loading arrays. Write bundles atomically; do not overwrite an earlier run merely because its display name matches.

Provide config JSON import/export, a per-time CSV download, a summary CSV, and self-contained HTML figure export with Plotly embedded. Keep numeric arrays at full resolution; plot decimation, if necessary, must affect display only. Reloaded runs must retain their original parameters and numerical status.

Maintain a bounded in-memory cache (e.g. eight results) keyed by canonical expanded inputs, horizon, model, evaluator options, and a fingerprint of relevant source files. Display names are not cache keys. Disk runs are history; do not silently reuse results from changed code. A schema/code fingerprint and cache identity must be shared by CLI and UI.

The Compare tab allows selecting up to 12 saved candidate protocols. Show tree, distillation locations, cutoffs, outcome policy, prediction model, rate, mean waiting time, fidelity, coverage, and horizon. Highlight parameter differences. Show separate rows for each evaluated model rather than obscuring which model produced a number.

For a controlled comparison, require identical **physical scenario**: link count, generation and initial states, memory parameters, hardware efficiencies, and units. Protocol choices and prediction model may vary. Different-hardware runs may be overlaid for exploration but cannot enter the same protocol-selection-loss calculation.

Provide an explicit **Evaluate candidates in both models** action for this bounded list. It evaluates each distinct candidate under both `full_real_x` and `local_channel_twirl`, reusing matching cached results. Show the number of pending calls first; progress advances per completed evaluation. This is a finite manual comparison, not a search. Preserve partial results if one candidate fails; do not calculate a selection loss from an incomplete matrix or mix incompatible horizons.

For the same completed candidate set `P`, let `pi_full` maximize full-model rate and `pi_twirl` maximize twirled-model rate. Display:

```text
prediction error for pi = R_twirl(pi) - R_full(pi)
absolute selection loss = R_full(pi_full) - R_full(pi_twirl)
relative selection loss = absolute loss / R_full(pi_full)
```

Call this **selection loss within these N candidates**. If the full-model best rate is zero, relative loss is undefined. Flag provisional comparisons when coverage is insufficient. Define ties with `abs(a-b) <= 1e-12 + 1e-3*max(abs(a),abs(b))` bits/slot; retain all near-tied winners and show the range of their full-model rates/losses instead of letting row order decide the conclusion. These tolerances describe ranking ambiguity, not a numerical error bound.

## How this serves the research questions

| Question in `analysis.md` | MVP interaction | Limit of the conclusion |
|---|---|---|
| Does a noise approximation change protocol selection? | Pin several trees/cutoffs, evaluate each under both models, compare predicted rankings and full-model performance. Inspect local biases and error composition. | Only local channel twirling and the explicit candidate set; no certified optimization. |
| Are cutoffs and distillation substitutes or complements? | Compare otherwise identical protocols in four groups: neither, cutoff only, distillation only, both. | Single parameter points and manually chosen settings; a regime map needs later sweeps/search. |
| Is Bell-outcome rejection worth retries? | Clone a protocol with `all_corrected`, `phi_pair`, and `psi_pair`; compare rate, waiting time, and quality together. | One fixed policy for the whole tree; no per-swap adaptation or syndrome-resolved output. |
| Which mechanism causes a change? | Vary T1 at fixed T2, introduce a slow link or poor-memory node, and inspect biases, correlators, Bell populations, and distributions. | Hypothesis generation; do not present a causal mechanism as established by a single plot. |

Ship presets for the default case, no memory noise, pure dephasing, a slow link (`p_gen[1]=0.05` with the others `0.2`), and one poor memory (`T1[2]=200`, `T2[2]=100`, others as default). Provide convenient protocol variants for root distillation, a shared cutoff of 10 slots, their combination, and Phi/Psi selection. Presets describe scenarios, not promised improvements or guaranteed positive key rates.

## CLI, verification, and completion criteria

The same evaluator must work without Streamlit:

```bash
source venv/bin/activate
python -m src.visualization.cli --config src/visualization/presets/default.json --output results/visualization
```

Authoring sanity check: the default configuration above was evaluated directly against the current backend. At `T=1024`, it gave coverage approximately `0.999999999999732`, repository mean waiting time `12.2741016165` slots, and pooled rate `0.02743022436` bits/slot. Treat these as a reproducible smoke reference with floating-point tolerance, not a substitute for independent controls or a universal benchmark.

Use focused tests with short horizons. Required scientific/integration checks:

1. A default homogeneous configuration expands to the correct link/node shapes, rates, and Bell weights. A heterogeneous case preserves endpoint indexing. Invalid lifetimes, shapes, probabilities, nonphysical states, or Pauli rates fail before evaluation.
2. The balanced four-link tree compiles to `s0,s2,s1`, root distillation adds `d3`, and cutoffs stay attached to their operation IDs. Reject malformed trees, duplicate/missing leaves, stale overrides, and invalid operations. Check a leaf and an internal distillation placement too.
3. A UI-service/CLI configuration produces the same PMF, weighted states, and headline metric as a direct `RepeaterChainEvaluation` call. Compare weighted states to avoid meaningless discrepancies in empty bins. Exercise heterogeneous noise, a cutoff, and postselection at small `T`.
4. With Bell-diagonal inputs and `kappa=0`, full and channel-twirled evaluations agree. For a perfect Bell pair under equal AD with survival `eta=0.8` and no Pauli noise, full-state fidelity is `0.82`, its subsequent Bell state twirl also has fidelity `0.82`, and the local channel-twirl fidelity is `0.81`. This is a channel/state control, not a repeater-rate prediction.
5. No-noise, perfect-state, deterministic-generation and deterministic-swap inputs without distillation complete at time 1 with fidelity and pooled secret fraction 1. With zero swap hardware success, show no completion without NaNs leaking into JSON or plots.
6. With no added memory noise, corrected swapping of prescribed real-X inputs and of their Bell-projected versions gives the same output Bell populations and pooled fraction. Use existing low-level maps for this control; it does not require a new production projection model. If reusing an equivalent existing test, name it rather than duplicate it.
7. A short-horizon result is marked provisional; doubling the horizon rebuilds no-cutoff entries and creates a distinct result. Storage round-trips arrays/configuration, and model conversion leaves the physical config unchanged.
8. Candidate comparisons reject mixed physical scenarios, handle zero-rate denominators and near ties, and distinguish prediction error from selection loss. Display interactions do not invoke the simulator again.

Run the new focused tests plus the relevant existing small Pauli–ADC numeric/metric/metamorphic tests. Inspect test parametrization before selecting tests; omit scaling benchmarks and long campaigns. Do not launch the optimizer suite or validation campaigns unrelated to this addition. Document the exact commands and results. Perform one CLI smoke evaluation and a local app startup check; exercise form submission/export/reload with a browser or Streamlit's test harness where available.

Done means the app starts from the documented command, the default preset produces real backend results, one can edit hardware/protocol choices and explicitly evaluate, compare/reload runs, inspect all specified plots and diagnostics, and reproduce an evaluation through the CLI. Add a short README entry linking this brief and the run commands. Report actual measured runtime for the default preset; make no unsupported interactivity or performance guarantee.

## Later extensions — do not implement in this pass

- **Bell/Werner state-projection baselines.** Implement trace-preserving linear projections of weighted coordinates: Bell `B(n,zR,zL,cZZ,cXX,cYY)=(n,0,0,cZZ,cXX,cYY)`; Werner `W(x)=(n,0,0,w,w,-w)` with `w=(cZZ+cXX-cYY)/3`. The schedule from `analysis.md` is after generation, after each actual waiting interval before the operation, and after each successful operation. This requires hooks inside the waiting/operation joins, including efficient/direct parity tests; projecting only final outputs is insufficient. These are prediction models: selected protocols are re-evaluated physically without inserting twirling gates.
- **Small parameter sweeps and regime maps.** Build on the same pure evaluation service and run schema. Start with one-axis slices, especially fixed T2 with variable T1; then bounded two-axis grids. Include positive-key boundaries and absolute rates, not percentage gains alone.
- **Bounded enumeration and cutoff optimization.** Enumerate the five ordered four-link swap trees and limited distillation placements; use a shared candidate space across all models. Label best-found versus exhaustive optima and verify leading candidates at longer horizons.
- **Intermediate-operation diagnostics.** Add an explicit trace/callback contract to the evaluator if needed; do not fake internal waiting-state trajectories by interpolating final completion-time states. Existing asymmetric evaluation does not expose all levels.
- **Per-swap outcome choices, syndrome-resolved processing, physical twirling as an action, and state/history-dependent cutoffs.** These change the model or action space and need separate designs.
