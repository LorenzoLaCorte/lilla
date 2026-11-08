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
