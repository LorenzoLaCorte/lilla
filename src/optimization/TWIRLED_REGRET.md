# Does optimizing a twirled model lose physical secret-key rate?

This study implements the decision-error direction in `slides.md`. It scores
an identical, predeclared finite policy family under each prediction model,
then measures the **physical** SKR of the selected policy. There is no
visualization-app dependency and no model-dependent search budget. It reuses
the standalone Pauli–AD evaluator, adaptive horizons and durable journals.

The preliminary 59.3%/54.2% Werner-source examples, 13.8% cutoff example and
4.94% channel-twirl lifetime shift motivate the experiment. They are **not**
new results of this metropolitan grid. The main sweep remains for you to run.

## Run

From the repository root:

```bash
source venv/bin/activate

# Inspect the complete plan without evaluation.
python -m src.scripts.twirled_regret --preset overnight --dry-run \
  --output results/twirled-regret/noise-size

# Run that saved plan. On macOS caffeinate prevents idle sleep.
caffeinate -i python -m src.scripts.twirled_regret --preset overnight --resume \
  --workers 2 --output results/twirled-regret/noise-size
```

The default grid contains **192 hardware/source cells**, **12,960 policies
summed across cells**, and **77,760 candidate/model evaluations**. Every
evaluation can require several doubled horizons. This is a substantial
batch; an overnight completion time is not guaranteed. Two worker processes
limit concurrent memory use and each uses one numerical thread. Each case
has a two-hour limit (`--case-timeout`); a timeout preserves all completed
evaluations. `--resume` reuses them, skips completed cases and continues
interrupted cases. Keep the same grid arguments when resuming.

For a smaller first batch, add the same restriction to **both** commands:
`--links 2 4 --sources werner phase_error --memory-scales 0.5 2`.

Rebuild all tables and figures from journals, without simulation:

```bash
python -m src.scripts.twirled_regret --summarize \
  --output results/twirled-regret/noise-size
```

Small end-to-end validation:

```bash
python -m src.scripts.twirled_regret --preset smoke \
  --output results/twirled-regret/my-smoke
```

The validated example is in `results/twirled-regret/final-smoke/`: one realistic
two-link Pauli-noise cell, R-state source, twelve policies and six models.
It exercises physical-unit conversion, prediction errors, policy transfer,
all four nested spaces and the text/plot exporter.

In this control, all-corrected Bell-source, Bell-stage and combined
Bell/channel reductions have zero fixed-policy Bell-population and SKR
error, including recurrence. The Werner-source surrogate instead selects
no recurrence: its physical rate is **26.7300 bits/s**, compared with
**31.3870 bits/s** from one terminal round, a **14.84% loss** in this small
family. Both selected policies use no cutoff and all corrected outcomes.
This is an error-composition effect under Pauli noise, not evidence that
Bell-diagonal reduction itself fails under unital noise.

A second validation in `results/twirled-regret/final-ad/` covers four
links, AD plus dephasing, $F=0.97$ R-state sources, $s=0.5$, no cutoffs and
at most one terminal recurrence. All 72 candidate/model evaluations
converged. Bell/source/channel approximations have nonzero per-policy
prediction errors, but **all select the same physical optimum** in this
twelve-policy family. This is a useful zero-decision-loss control: inaccurate
predictions need not change the best action.

## Hardware grid and controlled source families

The default grid crosses:

| Axis | Values |
|---|---|
| Elementary links / nodes | 2, 3, 4, 8 / 3, 4, 5, 9 |
| Source fidelity to Phi+ | 0.97, 0.99 |
| Source error structure | Werner, phase-error mixture, R-state |
| Memory scale $s$ | 0.05, 0.2, 1, 2 |
| Physical waiting noise | matched unital Pauli; amplitude damping plus dephasing |

The optical/memory calibration is the one used in the previous metropolitan
batch: 5 km elementary span, 1.5 dB combined midpoint-path loss,
$p_{\rm gen}=0.0036793$, 35-microsecond attempts, swap hardware success 0.85,
and reference $(T_1,T_2)=s(2.4,1.71)$ seconds. See
[the hardware reference and its sources](../analysis/metropolitan-reference.md).
This is an optimistic hybrid assumption, not a demonstrated integrated
device. In particular, the effective memory decay measurements do not
establish zero-temperature relaxation toward $|0\rangle$.
The scales span weak to optimistic operational memories: $T_1=0.12$–4.8 s
and $T_2=0.0855$–3.42 s. Scaling below the reference is a sensitivity
assumption; it deliberately probes noisy and near-key-threshold cells.

With rates measured per slot, AD uses
$\kappa=1/T_1$ and $(\Gamma_X,\Gamma_Y,\Gamma_Z)=
(1/T_2-1/(2T_1),1/T_2-1/(2T_1),0)$.
The Pauli control uses $(1/T_2,1/T_2,1/T_1)$ and zero AD. Thus both have
identical one-qubit contractions and different affine translations.
`--noises none dephasing pauli ad ad_pure` enables additional controls;
`ad_pure` sets $T_2=2T_1$.

Source definitions, with $F$ fixed:

- `werner`: $F\Phi^+ +(1-F)(I-\Phi^+)/3$.
- `phase_error`: $F\Phi^+ +(1-F)\Phi^-$.
- `bit_error`: $F\Phi^+ +(1-F)\Psi^+$ (optional).
- `r_state`: $F\Phi^+ +(1-F)|01\rangle\langle01|$.
- `pe`: $v|\psi_q\rangle\langle\psi_q|+(1-v)I/4$,
  $|\psi_q\rangle=\sqrt{1-q}|00\rangle+\sqrt q|11\rangle$ (optional).
  Here $v=(4F_{\rm ref}-1)/3$, so **the input `--fidelities` is a
  Bell-reference fidelity, not the actual PE fidelity**. The default
  `--pe-q` is 0.25; it is held fixed across models, not optimized.

All links in one cell have the same source. These are controlled error
compositions, not claims that a specific experiment prepares every family
at the same rate. Generation cost is held fixed across source families.

Increasing link count at fixed elementary span increases total distance
(10–40 km by default). For a complementary fixed-distance comparison,
replace `--link-km 5` with `--total-km 20`: the span per link, generation
success and slot duration are then recomputed at each chain size. Keep this
comparison in its own output directory. `--geometry slow_link`,
`weak_memory` or `bottleneck` adds the previously used fivefold generation
or fourfold memory asymmetry.

## Precisely defined prediction models

For real-X coordinates $x=(s,z_R,z_L,c_{ZZ},c_{XX},c_{YY})$, define

$$
B(x)=(s,0,0,c_{ZZ},c_{XX},c_{YY}),\qquad
W(x)=(s,0,0,w,w,-w),\quad
w=(c_{ZZ}+c_{XX}-c_{YY})/3.
$$

These maps preserve success weight $s$ and Phi+ fidelity. They are applied
to each time bin, rather than to a pooled end-to-end state.

| Model | Source | Waiting channel | After each successful operation |
|---|---|---|---|
| `full` | physical | physical | preserve all six coordinates |
| `bell_source` | $B(\rho_0)$ | physical | preserve all six |
| `werner_source` | $W(\rho_0)$ | physical | preserve all six |
| `channel_twirl` | physical | remove local affine shifts, retain contractions | preserve all six |
| `bell_channel_twirl` | $B(\rho_0)$ | same local channel twirl | preserve all six; states stay Bell-diagonal |
| `bell_each_stage` | $B(\rho_0)$ | physical | apply $B$ |

All six run by default. `--models` selects a subset of approximations;
`full` is always included. Optional `werner_each_stage` uses $W$ at the
source and after every operation, with physical waiting noise.

**These are different approximations.** In particular, the source-only
models still propagate real-X states under physical AD. The boundary
projection model allows biases to regenerate *within* a storage interval;
it is not a continuous Bell-diagonal memory model. The combined
`bell_channel_twirl` is the closed Bell-diagonal surrogate. The channel-only
model keeps any original source biases, so it isolates a change in memory
channel rather than silently changing both source and channel.

The channel twirl is implemented as
$(\Gamma_X,\Gamma_Y,\Gamma_Z)\mapsto
(\Gamma_X+\kappa/2,\Gamma_Y+\kappa/2,\Gamma_Z+\kappa)$ and $\kappa\mapsto0$.
Simply zeroing AD while leaving Pauli rates unchanged would be an unmatched
comparison and is deliberately not used.

These projections are **prediction substitutions**. Policies selected under
them are executed on the original, untwirled hardware/source. Deliberately
applying a physical twirl, basis-adapted recurrence or extra gate sequence
is not an action in this family. Such actions could improve the physical
optimum and should be a later, separately costed experiment.

## Exactness boundary, including recurrence

[Davies, Avis and Wehner, Theorem 1](https://arxiv.org/html/2509.16689v1)
establish the non-postselected swap-and-correct Bell-population equivalence
for their input-state replacement setting. Intervening non-unital storage
and recurrence are not automatically covered by that theorem.

**For our implemented real-X recurrence, the extension under Pauli storage
is immediate from its map.** Write $z=c_{ZZ}$, $x=c_{XX}$ and $y=c_{YY}$,
retaining the success weight $s$. For kept/auxiliary pairs $a,b$,

$$
\begin{aligned}
s'&=(s_as_b+z_az_b)/2,& z'&=(z_as_b+s_az_b)/2,\\
x'&=(x_ax_b+y_ay_b)/2,& y'&=(y_ax_b+x_ay_b)/2.
\end{aligned}
$$

No local biases occur. Thus $B D(a,b)=D(Ba,Bb)$, including success
probability. All-corrected swapping also has a closed Bell sector, and
local Pauli storage commutes with $B$. These identities extend to the
success-weighted time bins, time-only cutoffs and renewal retries. Induction
therefore preserves the completion PMF, Bell populations, fidelity, errors,
pooled SKF and SKR for every policy with all-corrected swaps, even with this
recurrence. A lost bias can still affect other observables or other
recurrences; this is not a theorem for arbitrary purification protocols.

**Why it can fail:** after a local affine noise interval,

$$
c'_{ZZ}=a_La_Rc_{ZZ}+a_Lb_Rz_L+b_La_Rz_R+b_Lb_Rs.
$$

Pauli noise has $b_L=b_R=0$; AD generally does not. Source/stage Bell
projection discards the middle terms. Channel twirling removes all shift
terms, including $b_Lb_Rs$ even for an initially Bell-diagonal input.
Furthermore, Phi/Psi acceptance weights $(s_as_b\pm z_{R,a}z_{L,b})/2$
expose biases even without additional damping.

Consequently, expected controls are:

- Pauli/noiseless storage, all-corrected swaps, with or without recurrence:
  Bell-source/stage reductions agree candidate by candidate.
- Bell-diagonal sources and Pauli storage: Bell reductions remain exact
  also for the supported postselection policies, since no biases arise.
- Bell-source replacement of an already Bell-diagonal source changes
  nothing even under AD; this trivial identity is a useful sanity check.
- Werner-source replacement can lose accuracy even under Pauli or no
  storage noise, because it discards error composition.
- Biased sources with selection, or non-unital waiting with discarded
  biases/shifts, can have prediction errors. **A prediction error does not
  imply a policy-selection loss**; both are measured separately.

The recurrence here is the untwirled bilateral-CNOT core, with equal target
outcomes accepted. It is not the complete BBPSSW protocol including random
Werner preprocessing. Do not interpret a Werner-policy loss as an advantage
over physically twirled or basis-optimized purification.

## Policy family and measurements

Default tree: balanced (near-balanced for three links). `--trees all_small`
enumerates all ordered trees through four links, and balanced/left/right
trees for larger chains. This tests structural choices without attempting
Catalan enumeration at large size.

Every tree uses the same cutoff list: no cutoff, or rounded
$0.5/\min p_{\rm gen}$ and $2/\min p_{\rm gen}$ slots. These are shared
memory-time cutoffs. `--cutoff-factors 0.1 0.25 0.5 1 2 4` refines the grid;
an empty `--cutoff-factors` means no-cutoff only.

Recurrence placements: none; one/two at the root; one at each root child;
one at every leaf or every elementary swapping output if the total budget
permits it. The default total cap is two. `--placements vertices` additionally
tries one/two rounds at every individual vertex. These are explicit finite
families, **not all distributions of two rounds over all vertices**.

Acceptance: all corrected, uniform Phi pair, uniform Psi pair, Psi at
the first swapping layer only, or Phi/Psi at the final swap only.
Final-only filtering matters: it can exploit biases accumulated at lower
swapping levels even for Bell sources. `--acceptance uniform` keeps only
the first three patterns. The same family is scored by every
model. Four nested comparisons separate causes:

| Space | Tunable controls |
|---|---|
| `swaps` | tree and cutoff; all corrected, no recurrence |
| `distillation` | additionally recurrence placement/count; all corrected |
| `postselection` | tree, cutoff, acceptance; no recurrence |
| `joint` | every listed control |

For each space $\mathcal P$, model $m$ selects
$\hat\pi_m=\arg\max_{\pi\in\mathcal P}R_m(\pi)$. The comparison is

$$
\Delta R_m=\max_{\pi\in\mathcal P}R_{\rm full}(\pi)
                -R_{\rm full}(\hat\pi_m),\qquad
\ell_m=\frac{\Delta R_m}{\max_{\pi\in\mathcal P}R_{\rm full}(\pi)}.
$$

The denominator is the finite-family optimum, not a global protocol optimum.
Every candidate is physically scored already; policy transfer is a lookup
in that column, never a comparison of the models' independently predicted
maxima. The metric is pooled asymptotic BB84
$R=\max(0,1-h_2(e_X)-h_2(e_Z))/\mathbb E[T]$. Acceptance failures,
hardware failures, distillation and retries are included in $\mathbb E[T]$.
There is no second multiplication by postselection probability.

Strict predicted SKR chooses the policy; exact ties use candidate ID only,
without physical information. Also report the best/worst physical loss over
all near-optimal prediction ties (default relative tolerance $10^{-4}$ plus
$2\times10^{-9}$ bits/slot). An all-zero surrogate can have a large deterministic
loss and zero minimum tie loss: call that a false-negative/tie, **not robust
optimizer regret**. Relative loss is null if the physical optimum is zero.
The near-tie range is a numerical sensitivity check, not a rigorous error
bound or statistical confidence interval.

## Output and numerical certification

`plan.json` records every case, exact hardware, source, models, candidate
count and worker command before execution. Per case:

- `case.json`: immutable specification.
- `MODEL/config.json`, `environment.json`, `trials.jsonl`: hardware,
  source/code hashes, versions, every score and successive-horizon history;
  each appended trial is flushed and fsynced.
- `paired.json`: aligned candidate/model results.
- `summary.json`: all four comparisons, transferred policy, error metrics,
  tie ranges and unresolved counts.
- `console.log`, `runner_status.json`: progress, timing and exit status.

At batch level, `comparisons.csv/json`, `evaluations.csv/json`,
`fixed-policy-errors.csv/json`, `key-margins.csv/json`, `statuses.csv/json`,
`manifest.json` and `README.md` support text-based inspection.
`loss-*.png/pdf` separates source/noise and chain size;
`transfer-and-ties.*` shows actual transferred throughput and tie sensitivity;
`prediction-controls.*` shows per-policy errors and sampled key margins;
`key-regimes.*` separates signed-margin curves by source and physical noise;
`policy-examples.*` draws the largest absolute-loss policy changes. All can
be regenerated read-only, including tables from interrupted journals.

Two consecutive horizons must have coverage at least 0.999999 and stable
SKR, SKF, mean wait and both errors (relative $10^{-5}$, absolute $10^{-9}$).
Default horizons start at 16,384 and double up to 1,048,576. Low cutoffs,
selection and recurrence can require still larger horizons. Unresolved or
failed evaluations are **missing, never zero**. A comparison missing any
candidate/model pair is labelled `paired_subset` and excluded from the main
loss plots. Increase `--max-horizon` in a **new output directory** for a
refinement; numerical or source-code changes intentionally invalidate resume.

Threshold follow-up, with recurrence disabled and a wider lifetime grid:

```bash
caffeinate -i python -m src.scripts.twirled_regret --preset threshold \
  --models channel_twirl --workers 2 \
  --output results/twirled-regret/thresholds
```

This plots sampled signed margins and brackets useful regions. It does not
fit or certify a new 4.94% threshold shift. Refine `--memory-scales` around
the crossings before quoting a lifetime penalty.

Additional exactness controls:

```bash
caffeinate -i python -m src.scripts.twirled_regret --preset controls \
  --output results/twirled-regret/controls
```

Do not infer a monotone scaling law from four sizes, aggregate zero-key
cells as zero regret, or claim the first comparison of AD with its Pauli
twirl: [Mondal et al.](https://arxiv.org/abs/2509.18064) already study that
comparison for fixed policies. The contribution sought here is controlled,
rate-aware **decision loss**, with exact zero-loss controls and explicit
noise/source/size boundaries. Finite-memory-resource normalization, physical
twirl actions, gate noise/duration and finite-key security are outside this
batch. Parallel copies are assumed for recurrence; rates are per chain,
not per memory qubit.
