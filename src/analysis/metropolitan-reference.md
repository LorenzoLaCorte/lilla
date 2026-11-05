---
title: "A metropolitan reference with positive-key baselines"
date: "29 October 2026"
---

**Use 5 km elementary links, giving 20/40 km chains for four/eight links.**
The new `metro97` and `metro99` regimes keep the earlier memory times and
swap hardware, while reducing fibre length and updating both generation
probability and attempt duration. Converged pilot calculations already
produce positive SKR. This is an optimistic hybrid benchmark, not a claim
that one experiment has integrated all its components.

**Update, 30 October 2026:** the 288-job optimization batch has now completed.
Its results are described in the [main report](pauliAD-overnight-report.md).
This note retains the calibration and fixed pilot checks performed before
that run; its pilot numbers should not be mistaken for optimized winners.

## First estimate the generation rate needed

Using reference memories $T_1=2.4$ s, $T_2=1.71$ s, $p_{\rm swap}=0.85$,
Bell-source fidelity $F_0=0.97$, balanced swaps and all corrected outcomes,
the following brackets locate the sign change of the pooled BB84 margin.
They use the **old 250-microsecond slot** and the AD waiting model:

| Elementary links | Terminal recurrence rounds | Nonpositive / positive $p_{\rm gen}$ | Sufficient generation rate |
|---:|---:|---:|---:|
| 4 | 0 | 0.009652 / 0.009739 | about 39/s |
| 4 | 2 | 0.003856 / 0.003891 | about 16/s |
| 8 | 2 | 0.014491 / 0.014609 | about 59/s |

These are thresholds of specific baseline protocols, not globally optimal
hardware requirements. Their rate-equivalent probability estimates at a
35-microsecond slot are approximately 0.00136, 0.00055 and 0.00205. Exact
discrete-time thresholds need not scale perfectly with slot duration;
the actual metropolitan baselines below were evaluated directly.

There is also a source-quality limit on the noiseless all-corrected
swaps-only benchmark. For $L$ Bell sources with visibility
$v=(4F_0-1)/3$, its endpoint correlations are $c_{XX}=c_{ZZ}=v^L$.
For $F_0=0.97$ and eight links, $v^8=0.72139$ and

$$
m=1-2h_2\left(\frac{1-v^8}{2}\right)=-0.16483.
$$

Thus even the no-waiting-noise version of that swaps-only baseline has
zero key. Raising generation alone cannot make its *noiseless* value
positive. The 0.99 source control makes the baseline comparison useful
at eight links; additional recurrence can also produce positive key with
0.97 sources. This is not a no-key theorem for all protocols or states.

## Distance and loss reference

[Agarwal et al., Physical Review X 4, 041038 (2014)](https://doi.org/10.1103/PhysRevX.4.041038)
report a measured **1.5 dB passive loss through 5 km of dispersion-shifted
fibre**, including approximately 0.2 dB connector loss. The same passive
measurement is described in [Dailey et al., Methods (2015)](https://doi.org/10.1038/srep16296).
We use the passive loss measurement only; **no amplification or gain is
assumed**. This is a laboratory fibre reference, not a promise about a
deployed city route with patch panels and switches.

The model uses $0.26$ dB/km propagation plus $0.2$ dB combined insertion loss:

$$
\mathcal L(d)=0.26d+0.2\ \mathrm{dB},\qquad
\eta_{\rm ch}(d)=10^{-\mathcal L(d)/10}.
$$

At $d=5$ km, transmission is 0.70795. In a midpoint two-photon architecture,
the two photon paths have combined length $d$; their *product* transmission
is this value. It must not be squared again as if each photon traversed $d$.

Loss alone does not specify success probability. We retain a non-fibre
attempt prefactor by a phenomenological fit through the existing C/D
benchmarks $(d,p)=(50,0.00092)$ and $(20,0.0026)$:

$$
\alpha_{\rm fit}=\frac{10\log_{10}(0.0026/0.00092)}{30}
=0.15040\ \mathrm{dB/km},\qquad
p_0=0.00092\,10^{\alpha_{\rm fit}50/10}=0.0051971.
$$

The fitted slope is only used to extract the assumed prefactor; the new
fibre uses the measured loss model above. We assume unchanged non-fibre
efficiency, giving

$$
p_{\rm gen}=p_0\eta_{\rm ch}=0.0036793.
$$

This prefactor is a benchmark assumption, not a measured memory-interface
efficiency from the fibre paper. Source coupling, conversion, heralding and
storage efficiencies must ultimately be calibrated for a specific device.

The attempt duration includes the midpoint optical round trip and an
**assumed 10-microsecond local overhead**:

$$
\Delta t=\frac{d}{2\times10^8\ \mathrm{m/s}}+10\ \mu\mathrm{s}
=35\ \mu\mathrm{s},\qquad
\frac{p_{\rm gen}}{\Delta t}=105.1\ \mathrm{s}^{-1}.
$$

Here $d$ is converted to metres. Each photon travels $d/2$ to the midpoint
and its herald travels back, so the propagation part is $d/c$. We recompute
memory lifetimes in slots using this duration; they remain **2.4/1.71 s**
at memory scale 1. The expected elementary generation time is 9.51 ms,
versus 272 ms in the old C reference.

## References for the source fidelities

**Keep $F_0=0.97$.** [Saha et al., Nature Communications 16, 2533
(2025)](https://doi.org/10.1038/s41467-025-57557-4) measured remote trapped-atom
Bell-state fidelities 0.968(4) and 0.972(3). This supports a 0.97
conditional source-quality reference, though not a 5 km memory link with
all the remaining parameters used here.

**Add optimistic $F_0=0.99$.** [Craddock et al., PRX Quantum 5, 030330
(2024)](https://doi.org/10.1103/PRXQuantum.5.030330) distributed photonic
entanglement over 34 km of deployed NYC fibre and separately demonstrated
approximately 99% fidelity at rates up to $2\times10^4$ pairs/s. This is a
strong metropolitan photonic reference; loading such states into the
assumed long-lived memories remains an integration assumption. Their
highest-rate and highest-fidelity operating points should not be combined.

The scripts use isotropic source models matched to these Bell fidelities,
not fits to either experiment's complete density matrix:

| Regime | Bell $F_0$ | Visibility $v$ | Link / chain lengths | Loss | $p_{\rm gen}$ |
|---|---:|---:|---|---:|---:|
| `metro97` | 0.97 | 0.960000 | 5 km / 20 or 40 km | 1.5 dB | 0.0036793 |
| `metro99` | 0.99 | 0.986667 | 5 km / 20 or 40 km | 1.5 dB | 0.0036793 |

The perfect-source and AD-5%/AD-15% controls are retained. Those damping
probabilities are controlled sensitivity tests, not measured preparation
noise from these papers. Variable-q PE sources retain the existing model
and the assumption that preparation cost is independent of q.

## Directly validated positive baselines

At memory scale 1 under AD, the pilot gives:

| Source | Links | Terminal rounds | SKF | SKR (bits/s) |
|---|---:|---:|---:|---:|
| $F_0=0.97$ | 4 | 0 | 0.13503 | 4.86868 |
| $F_0=0.97$ | 8 | 0 | 0 | 0 |
| $F_0=0.97$ | 8 | 2 | 0.05974 | 0.49536 |
| $F_0=0.99$ | 4 | 0 | 0.50470 | 18.19766 |
| $F_0=0.99$ | 8 | 0 | 0.14845 | 3.36824 |
| $F_0=0.99$ | 8 | 2 | 0.24676 | 2.60374 |

This already illustrates the desired tradeoff: recurrence enables eight-link
key with 0.97 sources, while at 0.99 it increases SKF but lowers SKR in this
matched example. The unital control is also positive: eight links give
0.49331 bits/s for 0.97 with two rounds, and 3.35299 bits/s for 0.99 without
rounds. At memory scale 2, the corresponding AD rates are 0.95589 and
6.18822 bits/s.

All 48 metropolitan baseline evaluations converged; **38 yielded positive
key**. They cover both fidelities, sizes, memory models, scales 0.5/1/2,
and terminal counts 0/2. This confirms a useful regime before spending the
full optimization budget. It does not claim those anchors are optimal or
that every source/defect condition will yield key. Local gate errors and
classical delays beyond the assumed elementary-attempt duration remain
omitted, as in the earlier runner.

## Overnight run

From the repository root:

```bash
source venv/bin/activate
caffeinate -i python -m src.scripts.gp_pauliAD \
  --preset hardware --hardware-regimes metro97 metro99 \
  --metro-link-km 5 --nodes 5 9 --memory-scales 0.5 1 2 \
  --relaxation-models ad unital --seeds 7 --gp-calls 60 \
  --max-dists 2 --max-total-dists 2 \
  --workers 2 --case-timeout 7200 \
  --output results/pauliAD/metropolitan-followup
```

This schedules **288 jobs** with the same four scientific questions,
controls, strict coverage checks and saved tables/plots. Add `--dry-run`
to inspect commands; repeat the exact command with `--resume` after an
interruption. Use a new directory, since these physics differ from the old
hardware batch. Runtime is not guaranteed to fit one night. Add seed 42
in a separate replication run if needed.

`plan.json` records distance, fibre/insertion loss, prefactor, attempt
overhead, physical/slot lifetimes and reference links. Batch tables include
distance, span, source fidelity and rates in bits/second. The completed
optimization results are now available in the main report.

Reproduce the pilot, including convergence histories and threshold brackets:

```bash
python -m src.analysis.metropolitan_reference
```

Its data are in `results/pauliAD/metropolitan-reference-20261029/`.
