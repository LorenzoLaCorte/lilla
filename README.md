# Optimization of Quantum Repeater Chains

This repository now extends the original algorithm introduced in "Efficient optimization of cut‑offs in quantum repeater chains" (Li, Coopmans & Elkouss 2021) with the extensions presented in "Bayesian Optimization for Repeater Protocols" (La Corte, Goodenough, Maity, Santra & Elkouss 2025).

(i) The original work includes two implementations: 
- The numerical algorithm calculating the waiting time distribution and the fidelity of the delivered entangled state.
- The optimizer used to optimize the cut-off time for maximal secret key rate.

(ii) The extension adds:
- A more general version of the algorithm: any number of nodes and heterogeneous hardware parameters are supported.
- A Bayesian optimizer searches for the optimal protocols without brute‑forcing.

## Tutorial (Original Repository) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/BoxiLi/repeater-cut-off-optimization/master)

A tutorial is written in Jupyter Notebook. By clicking the badge above, you will be directed to an online Jupyter Notebook. After the loading succeeds, please find and click `tutorial.ipynb`. Then you can run examples in the notebook online without installing anything! (The loading can take a few minutes. It takes too long, please try again half an hour later.)
The tutorial is relative to the original work of (Li, Coopmans & Elkouss 2021).

## Setup
(Suggested) Create a virtual environment:
```bash
python -m venv venv
source ./venv/bin/activate
```
Install requirements:
```bash
pip install -r requirements.txt
```

## Classical communication

Classical communication is enabled by default in `RepeaterChainEvaluation`
and `repeater_sim`. Elementary segments have length 1. Each swap or distillation
attempt costs `t_c = max(input segment lengths)` in elementary-link time units;
swaps add lengths and distillation preserves length. Both successful and failed
operation attempts pay this delay. Cutoff rejections also pay `t_c` to herald
the abort before a full restart. A memory-cutoff failure therefore takes
`min(t1, t2) + cutoff + t_c`; other cutoffs add `t_c` to their decision time.
This assumes a one-way abort broadcast with negligible local reset time.

Following Table 1 of the [journal version of Li et al. (2021)](https://doi.org/10.1109/TQE.2021.3099003),
both inputs decohere during communication before the operation. The older link's
total storage duration is `abs(t1 - t2) + t_c`; the newer link's is `t_c`.
The Pauli/ADC backend evaluates its finite sequential channel at these full
durations. Elementary generation already includes its heralding time.

To reproduce simulations that omit communication, use
`RepeaterChainEvaluation(classical_communication=False)` or
`repeater_sim(parameters, classical_communication=False)`.
For an individual `compute_unit`, `swapping`, or `distillation` call, pass `t_c`
explicitly for longer links (default 1); the switch overrides it to zero.

## Real-X distillation

Select a strategy through the simulation parameters:

```python
parameters["distillation_strategy"] = "dejmps"
simulator = RepeaterChainEvaluation(state_type=RealXState)
pmf, states = simulator.nested_protocol(parameters)
```

All strategies target `Phi+`, use bilateral CNOT, and accept auxiliary outcomes
`00` and `11`, summed without an outcome-dependent correction. Storage noise
(waiting and communication) is applied before preprocessing. The selector
applies to every distillation in the protocol, in both direct and efficient joins.

| Strategy | Before CNOT, on both inputs | Accepted output |
|---|---|---|
| `bbpssw` (default) | Werner twirl | Werner twirl |
| `bell_diagonal` | Bilateral Pauli twirl | Bilateral Pauli twirl |
| `none` | No twirl or rotation | No twirl |
| `dejmps` | Alice `Rx(+pi/2)`, Bob `Rx(-pi/2)` | No twirl |

BBPSSW follows the input symmetrization and A1–A3 of
[Bennett et al.](https://arxiv.org/abs/quant-ph/9511027), in the `Phi+` frame
rather than the singlet frame. DEJMPS follows
[Deutsch et al., Eqs. (2)–(8)](https://arxiv.org/abs/quant-ph/9604039).
Neither strategy adaptively sorts Bell weights. `bell_diagonal` is the raw
CNOT recurrence with Bell-diagonal symmetrization, rather than DEJMPS.

For coordinates `(n,u,v,z,x,y)`, the twirl is
`(n,0,0,omega,omega,-omega)`, with `omega=(z+x-y)/3`. It preserves trace and
`Phi+` fidelity while discarding local biases and anisotropy. Bell twirling
only zeros `(u,v)`, preserving every Bell weight and all three correlations.
DEJMPS's complete rotated-CNOT map closes on real-X states even though the
individually rotated inputs may leave this family. For real-X inputs, summing
both accepted branches makes its output Bell-diagonal without an extra twirl.
All maps support probability-weighted inputs; efficient joins retain their
bilinear form. Hardware success scales the accepted mass afterward, and
retries use each selected map's success and failure probabilities.

The former `real_x_werner_twirling` flag has been replaced by this explicit
protocol selector. Subsequent non-unital storage noise can produce a general
real-X state again. Earlier raw-CNOT or output-only-twirl results must be
identified by their actual circuit; they do not describe BBPSSW.

## Compare distillation strategies

The focused study is the practical starting point. Stop an existing extensive
sweep with Ctrl+C, then run:

```bash
source venv/bin/activate
python -m src.scripts.compare_distillation_strategies --preset focused \
    --output-dir results/distillation_strategies/focused \
    --reuse-from results/distillation_strategies --time-budget-hours 4
```

It contains **144 scenarios and 17,424 strategy/candidate evaluations**:

- 72 main cases: five nodes, `p_gen=0.2`, `p_swap=0.9`, communication enabled,
  all three source/noise families, both visibilities and all three noise strengths.
- 24 chain-size controls at three/nine nodes, visibility `0.95` and strength `0.02`.
- 24 communication-off controls at five nodes and strength `0.02`.
- 24 hardware controls changing generation or swapping probability separately,
  at five nodes, visibility `0.95` and strength `0.02`.

All cases retain every placement of zero to two distillations and cutoff
factors `2,8,None`. The factor `0.5` is excluded because tight cutoffs can cause
extremely long retry tails. The `q=0.5` biased-X source is omitted because it
equals the Werner source at the same white-noise visibility; `q=0.2,0.35` remain.
Selection uses fixed physical/control settings, independently of observed wins.
This is a restricted protocol/cutoff comparison; it does not find an optimum
over every cutoff or hardware configuration.

`--reuse-from` imports matching saved tasks only when the physics fingerprint,
time grid, numerical tolerances and actual simulation parameters agree. The
source checkpoint stays untouched; partial horizon results can also be reused
and extended. Repeating the command skips completed tasks and can import new
compatible source results. Reports are kept separately in the focused directory.
`plan.json` records cached/remaining work and a heuristic cost estimate from
previous timings. The four-hour limit stops between evaluations and is a work
budget, rather than a guarantee of numerical convergence of every candidate.

To prepare/import and inspect the plan without new simulations, add
`--prepare-only`. To redraw focused results without evaluation:

```bash
python -m src.scripts.compare_distillation_strategies --plot-only \
    --output-dir results/distillation_strategies/focused
```

The much larger extensive grid is still available:

```bash
source venv/bin/activate
python -m src.scripts.compare_distillation_strategies --preset extensive --time-budget-hours 4
```

Repeat either command to continue its study. Extensive results are saved after every evaluation in
`results/distillation_strategies/evaluations.jsonl`; numerical settings and
a physics-code fingerprint prevent accidental reuse across changed models.
Increase `--max-horizon` beyond its default `262144` to extend under-covered
evaluations, retaining all previous horizon results. Completed evaluations
are always reused. Errors remain recorded for inspection in the checkpoint.

The extensive grid contains 2,160 physical scenarios and 357,120 requested
strategy/candidate evaluations; a four-hour run may complete a subset,
depending on the hardware and horizons. Scenario order is shuffled with a
fixed seed. Each scenario compares exactly the same source, noise and hardware:

- 1, 2 or 3 swap levels (3, 5 or 9 nodes); every placement of 0–2 distillations.
- Werner sources, anisotropic Bell-diagonal sources, and biased X sources
  `sqrt(1-q)|00> + sqrt(q)|11>`, with white-noise visibility `0.85` or `0.95`
  and, for biased sources, `q=0.2,0.35,0.5`.
- Isotropic Pauli, anisotropic Pauli, and matched anisotropic Pauli plus ADC;
  three noise strengths relative to generation probability.
- `p_gen=0.05,0.2`, `p_swap=0.5,0.9`, ideal distillation hardware, communication
  enabled/disabled, and cutoffs `ceil(factor/p_gen)` for `factor=0.5,2,8`,
  plus a genuine no-cutoff candidate.

Integer horizons double until coverage is at least `0.999` and successive
mean times, fidelities, X/Z errors and secret fractions converge. This is a
numerical convergence check, not a rigorous tail bound. Horizon-limited and
failed candidates are unresolved, never treated as zero physical rate.
Key rates use pooled asymptotic BB84 errors from the actual XX/ZZ correlations.
All gates/twirls are ideal and instantaneous; copies prepare independently
in parallel, with generation, rejections and full-reset retries included.
Memory cost and adaptive Bell-weight/basis optimization are not optimized.

`evaluations.csv` contains all numerical outcomes; `best_policies.csv` gives
each strategy's best key-rate policy **over candidate groups completed by all
four strategies**. It includes `search_complete` and matched/requested counts
to distinguish partial from complete finite searches. The PDF/PNG compare
these best rates against BBPSSW, keeping zero-rate cases explicit;
`summary.md` documents assumptions and numerical settings.

To regenerate all reports and plots without evaluations:

```bash
python -m src.scripts.compare_distillation_strategies --plot-only
```

To snapshot partial plots while the sweep is running, save reports separately:

```bash
python -m src.scripts.compare_distillation_strategies --plot-only \
    --report-dir results/distillation_strategies/partial
```

This reads completed checkpoint writes without modifying the live checkpoint.
The scatter compares best key rates; `strategy_summary.pdf/png` counts scenarios
with positive key rates and highlights those where BBPSSW gives zero.

A validated short study is saved separately under `results/distillation_strategies/smoke`:

```bash
python -m src.scripts.compare_distillation_strategies --preset smoke --output-dir results/distillation_strategies/smoke
```

Use `--levels`, `--max-distillations`, `--max-evaluations`, and `--output-dir`
for custom studies. A changed candidate grid requires a separate output directory.

## Distillation source mechanisms

The focused comparison already includes white-noise mixtures of partially
entangled sources (`q=0.2,0.35`, visibility `0.85,0.95`), alongside Werner and
anisotropic Bell-diagonal sources. These are not pure PE states. A separate
small comparison includes the exact pure source at `q=1/4` and a perfect Bell
control at `q=1/2`:

```bash
python -m src.scripts.compare_distillation_strategies --preset pure_pe \
    --output-dir results/distillation_strategies/pure_pe
```

This has 32 scenarios and 2,048 strategy/candidate evaluations: three/five
nodes, zero noise and three nonzero noise models, communication on/off, every
placement of zero to two distillations, and cutoff factors `2,None`. It has
already been run to numerical completion. Repeating the command reuses all
saved evaluations. The existing focused/extensive checkpoints remain compatible.

For an ideal state-level diagnostic, including the slides' dephased Bell example:

```bash
python -m src.scripts.distillation_source_mechanisms
```

PDF/PNG curves and CSV/JSON measurements are saved under
`results/distillation_strategies/mechanisms`, including fidelity, density-matrix
populations/coherence, purity, entanglement negativity and per-round acceptance.
The slides' twirling example uses `0.95 Phi+ + 0.05 Phi-`, for which the raw
CNOT recurrence lowers fidelity to `0.905`; Bell-diagonal twirling leaves that
example unchanged. An input Werner twirl gives conditional fidelity `0.96496`.

For a pure PE source, write `C=2 sqrt(q(1-q))`. Raw recurrence accepts both
equal outcomes with probability one, keeps populations `1-q,q`, and replaces
the coherence parameter `C` by `C^2`. At `q=1/4`, fidelity drops from `0.933013`
to `0.875`, and negativity from `0.433013` to `0.375`. Repeated identical-copy
rounds approach a separable diagonal mixture. The `11` branch alone produces
a perfect Bell pair, but the `00` branch is less balanced; the implemented raw
protocol accepts and averages both branches.

These diagnostics use ideal instantaneous gates and no storage noise. They
establish conditional state-quality loss, not a universal ranking of key rates;
the network comparison additionally accounts for preparation, noise and retries.
Regression tests check the analytic matrices, explicit CNOT circuit, repeated
rounds, perfect-Bell control, and direct/FFT evaluator retry probabilities.

## Reproduce Results

1) To reproduce (Li, Coopmans & Elkouss 2021) results, refer to the [original repository](https://github.com/BoxiLi/repeater-cut-off-optimization)
2) To reproduce (La Corte, Goodenough, Maity, Santra & Elkouss 2025) results, once the environment has been set up, run:
```bash
chmod +x ./src/scripts/reproduce_results.sh
source ./src/scripts/reproduce_results.sh
```

## Computation-time scaling plot

The plot compares
Werner and real-X backends for each of three swap probabilities:

```bash
python -m src.scripts.benchmark_scaling
```

It evaluates swap-only nested chains with 5 through 513 nodes, `p_gen=0.1`,
`p_swap` values 0.2, 0.5, and 0.9, and the paper's level-dependent coherence times
and memory cutoffs. Seaborn's `colorblind` palette identifies these probabilities;
circles/solid lines and squares/dashed lines identify Werner and real-X.
Figures use an 88 mm single-column width, with title-free legends and no
in-plot annotations; simulation settings remain in the JSON checkpoint.
For each chain, both backends use the same adaptively selected horizon covering
at least 99% probability. Calibration/JIT warmup is excluded from the median of
three evaluation timings. The real-X run includes amplitude damping; use
`--adc-factor 0` for the matched Werner physical submodel. Communication is on
by default; `--no-classical-communication` selects the paper's timing assumption.

PNG/PDF plots and CSV/JSON measurements are saved under
`results/benchmarks/chain_scaling`. Large chains can require millions of time
bins; `--max-level 6` gives a smaller run. Completed points are saved as the run
proceeds. Runs resume by default, reusing completed points and saved individual
timing samples. `--no-resume` starts fresh. Use `--help` for probability,
coverage, horizon, and repetition options.

To update the figures after changing colors, without running any simulations:

```bash
python -m src.scripts.benchmark_scaling --plot-only
```

This reads the existing JSON and its saved settings, regenerates PNG/PDF/CSV,
and leaves the measurement checkpoint unchanged. Use `--output-dir` to select
a different saved run.

To profile the evaluators at all completed points using their saved horizons:

```bash
python -m src.scripts.profile_scaling --repeats 3
```

The default CPU FFT Werner evaluator now combines cutoff and operation failures
into one full-reset retry kernel for swapping and distillation. It shares the
causal FFT solver with real-X and renews completion probability and weighted
Werner parameter together. Direct-convolution and GPU paths retain their
existing implementations.

To update a benchmark checkpoint made with the original Werner retry engine:

```bash
python -m src.scripts.benchmark_scaling --refresh-werner
```

This reuses the saved horizons and X-state samples, retimes only Werner, and
retains previous Werner timings in each record as provenance. Partial refreshes
resume normally. `--plot-only` renders saved checkpoints without evaluations.

The profiler leaves the timing checkpoint unchanged and saves `profiling.json`,
`profiling.csv`, `profiling.md`, and raw `*.prof` files beside it, all directly
under `results/benchmarks/chain_scaling` by default.
It measures fresh reference timings separately from instrumentation, records
disjoint phase timings, and checks agreement of the completion PMFs and one
shared six-column X-state renewal per level. Completed compatible profiles
are reused; `--overwrite` reruns selected cases. No missing benchmark cases
or horizon calibrations are launched. For a single 513-node example:

```bash
python -m src.scripts.profile_scaling --p-swap 0.9 --min-level 9 --max-level 9 --repeats 3
```

To regenerate the single-column companion plot of runtime composition from
saved profiles, without rerunning evaluations:

```bash
python -m src.scripts.plot_scaling_profile
```

This writes `runtime_breakdown.pdf`, `.png`, and `.csv`, comparing the largest
profiled chain for each swap probability. Each panel shows measured seconds
on its own linear time axis. Stacked bars separate scalar retry setup, its
batched application, state/noise maps, and other work. The plotted components
sum to each backend's profiled runtime; there is no normalization.

## File overview
- The core code for the numerical simulation of repeater chains is under `core/repeater_algorithm.py`.
- The protocol units such as entanglement swap, distillation or cut-off are defined in `core/protocol_units.py` and `core/protocol_unit_efficient.py`. The later makes use of customized convolution in Fourier space.
- (New!) The extended version of the algorithm works representing protocols using the full binary trees class defined in `types/protocol_types.py`.
- The cut-off optimizer can be found in `optimization/optimize_cutoff.py`.
- (New!) The Bayesian optimization logic is implemented in the files `optimization/gp_*.py`.
