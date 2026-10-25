# Partially-Entangled (PE) source imbalance and distillation study

This analysis goes a bit beyond the paper's scope because the tool capability is restricted to Bennet purification and I would not go beyond that for now.

Let entanglement source prepare
$|\psi_q\rangle=\sqrt{1-q}|00\rangle+\sqrt{q}|11\rangle$.

Bell sources set $q=1/2$, partially entangled sources set $q=1/4$.

Apply independent amplitude damping to both qubits for a fixed time $\tau$:
$\eta=\exp(-\tau/T_1)$, with $T_2=2T_1$. Connect 2, 4, 8, or 16 links using a balanced tree and one global acceptance policy.

There is no additional waiting noise, cutoff, or distillation. Find the critical $T_1/\tau$ where the unclipped BB84 score crosses zero; key is positive above it. Thresholds use the backend swap maps directly. Since accepted swaps have nonzero probability and finite retry times here, a positive SKF gives positive SKR. A full timing check follows below. 

Source imbalance changes which protocol is optimal.
In particular, the source imbalance changes the protocol maximizing the SKR in, e.g., 
the basis of distillation (Bennet's) recurrence, the accepted syndrome (postselection), and placement of operations.

These are some exploratory tests, reproducible with:

```bash
source venv/bin/activate
python -m src.analysis.pe_distillation
python -m pytest -q src/tests/analysis src/tests/pauli_adc
```

Code: [pe_distillation.py](pe_distillation.py). Observed data:
[timing.csv](../../results/analysis/pe-distillation-20261022T110904Z/timing.csv),
[state controls / noiseless scenario](../../results/analysis/pe-distillation-20261022T110904Z/state-controls.csv),
[matched bias control](../../results/analysis/pe-distillation-20261022T110904Z/bias-control.csv),
[settings](../../results/analysis/pe-distillation-20261022T110904Z/settings.json).

## Operations

- **Z:** existing bilateral-CNOT recurrence, accepting target outcomes 00 and 11.
- **X:** Hadamards on all four inputs immediately before Z recurrence, followed by Hadamards on the retained pair. 
  Waiting noise acts before these gates in the original physical basis.
- **11:** the Z circuit accepting only target outcome 11.

The X and 11 maps are derived from dense four-qubit circuits and verified to close on real-X states. 
The script installs each map only inside a scoped analysis context. 
It uses the existing waiting-time join, renewal, validators, and pooled BB84 metrics, without changing production operations or app options.

## Noiseless scenario

For two identical noiseless inputs with q=1/4:

| Operation | Intrinsic success probability | Output SKF |
|---|---:|---:|
| Input pair | — | 0.64542 |
| Z recurrence | 1 | 0.45644 |
| X recurrence | 0.875 | 0.95361 |
| Keep only target 11 | 0.375 | 1 |

The standard pooled (Z) recurrence does not concentrate this pure-state imbalance.
X recurrence and syndrome selection do. 
This table describes conditional state quality only whereas the tests below include preparing both copies and retries.

## Noisy scenario

We consider four fresh links, a balanced tree (doubling), p_gen=0.2, 
deterministic swap/distillation hardware, pure AD with T2=2T1, no cutoffs. 

We consider three source choices q=0.25, 0.45, 0.5, and two lifetimes T1=100 and 400. 

We compare no-distillation with all/Ψ swaps, and one round of Z, X, or 11 recurrence at leaves, both first-level outputs, or the root.

All 66 physical configurations and four bias-control configurations converged
at horizons 1024/2048 with coverage above 0.999999 and stable rate, mean wait,
eX, and eZ. The best tested choices at T1=400 were:

| q | Best tested choice        | SKR (bits/slot) | No-distillation/all-outcome SKR |
|---|---|---:|---:|
| 0.25 | X recurrence at leaves | 0.030086        | 0.010357 |
| 0.45 | No distillation        | 0.063861        | 0.063861 |
| 0.50 | Z recurrence at root   | 0.069437        | 0.067594 |

For q=0.25, early X recurrence improves rate by 2.90 times. (Early Z recurrence instead gives zero key at this point.)
Bell sources still beat the tested PE sources after choosing the best operation from this set of possibilities.

## A matched decision control

At q=0.25 and T1=100, erase IZ and ZI only at each distillation input, preserving
its trace, XX/YY/ZZ, Bell populations, fidelity, and immediate BB84 fraction.
This is a local prediction control; it is not a full Bell-projection schedule.

| Root operation | Full-model SKR | Bias-erased prediction |
|---|---:|---:|
| Z | 0 | 0 |
| X | 0.00078987 | 0.00078987 |
| Keep 11 | 0.00088397 | 0 |

The full model selects 11; the bias-erased predictor selects X. Executing X
physically loses 10.6% of the best tested physical SKR. Before the operation,
the two representations agree on all the headline Bell-diagonal observables.
The local biases affect the 11-branch success probability and conditional state.

## Contribution to investigate

Concentration of partial entanglement is established
([Bennett et al.](https://arxiv.org/abs/quant-ph/9511030)); SKR-based distillation
decisions have also been studied
([Comparing one- and two-way quantum repeater architectures](https://www.nature.com/articles/s42005-025-02222-x)).
AD reversal in repeaters is another relevant baseline
([Jeong et al.](https://arxiv.org/abs/2602.21563)).

A stronger research target is to establish when omitted population biases change the preferred acceptance rule, recurrence basis and placement, then quantify the resulting physical SKR loss across network sizes. 
Source q, hardware success and cutoffs would need a broader controlled optimization to support general design rules or novelty claims.

These tests assume instantaneous ideal gates and measurements, parallel copy preparation, and enough memories for distillation. 
Rates are per slot and do not charge total memory or elementary-pair consumption. 
The reported winners are restricted-menu winners under those assumptions.

Verification: 101 analysis/backend tests passed, including independent dense
circuit comparisons, unequal-age AD before basis rotation, analytic pure-state
concentration, exact four-link retry timing, and the matched decision control.
