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

## File overview
- The core code for the numerical simulation of repeater chains is under `core/repeater_algorithm.py`.
- The protocol units such as entanglement swap, distillation or cut-off are defined in `core/protocol_units.py` and `core/protocol_unit_efficient.py`. The later makes use of customized convolution in Fourier space.
- (New!) The extended version of the algorithm works representing protocols using the full binary trees class defined in `types/protocol_types.py`.
- The cut-off optimizer can be found in `optimization/optimize_cutoff.py`.
- (New!) The Bayesian optimization logic is implemented in the files `optimization/gp_*.py`.
