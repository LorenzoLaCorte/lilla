---
title: "Optimizing repeater protocols with twirled approximations"
date: "1 November 2026 — interim results"
---

We consider homogeneous repeater chains and investigate whether a protocol
that maximizes the secret-key rate in an approximated model remains efficient
when evaluated with a more complete description of the same hardware. The
present results show a **23.94% loss from a Werner-selected cutoff in a fully
evaluated, three-protocol space**. They also identify larger losses from
Werner-selected distillation and a smaller loss from channel-twirled
optimization, although those joint protocol spaces are not yet fully resolved.

This note follows the terminology and notation of La Corte et al.,
[*Bayesian Optimization for Repeater Protocols*](https://arxiv.org/abs/2502.02208)
(QCNC 2025): **repeater protocol**, **protocol space**, **hardware regime**,
**figure of merit**, and distillation at the **link-level**, **intermediate
levels**, or **end-to-end level**. In particular, $N$ denotes the number of
nodes, so the chain contains $N-1$ elementary links. The local manuscript is
[`qcnc2025-paper/main.tex`](</Users/lorenzolacorte/Research/qcnc2025-paper/main.tex>).

**Snapshot:** 1 November 2026, **13:54:48 CET** (12:54:48 UTC), with **158/192
jobs completed**. The batch was still running. Tables below use completed
jobs only; they exclude the partially written journals of active workers.
The [frozen evidence](twirled-model-optimization/evidence.json) records the
input checksums, comparison scopes, selected protocols and numerical histories
for the representative results. A completed job means that its protocol list
has been processed; it does not mean that every evaluation converged.

## 1. Methodology

### Repeater protocols and protocol spaces

Following Section III of the QCNC paper, a repeater protocol is represented
by a binary tree. Leaves represent freshly generated elementary links;
each internal vertex represents the link produced by swapping its two
children. A vertex label specifies the number of rounds of entanglement
distillation performed on that link. There are $v=2N-3$ vertices. We denote
the label at vertex $i$ by $\kappa_i$ and the total number of rounds by
$\mathcal K=\sum_i\kappa_i$.

The QCNC parameter $\beta$ limits rounds per vertex. The present global
budget $\mathcal K\leq2$ is more restrictive than taking $\beta=2$ over
the full protocol space.

We extend this description with a shared memory-time cutoff $t_{\rm cut}$
and an acceptance rule at each swapping vertex: accept all corrected Bell
outcomes, the Phi pair, or the Psi pair. Upon failure, the consumed input
links must be generated again. These retries are included in the waiting
time of the protocol.

The current batch fixes a balanced swapping tree for four and eight links,
and a near-balanced tree for three links. **It does not optimize tree
symmetricity.** The QCNC paper's finding that balanced trees perform well
in homogeneous hardware motivates this choice; it does not establish
optimality for the enlarged operation and noise model used here.

The restricted protocol space contains:

- Three cutoff choices: no cutoff, 136 slots, or 544 slots. These are
  approximately $0.5/p_{\rm gen}$ and $2/p_{\rm gen}$ for the finite cutoffs.
- Distillation with $\mathcal K\leq2$: none; one or two rounds at the root;
  one round at each root child; or one round at every elementary link or
  first swapping output when the total budget permits it. This is a
  selected family of placements, not every assignment with $\mathcal K\leq2$.
- Six acceptance patterns before deduplication: all corrected; uniform Phi;
  uniform Psi; Psi only at the first swapping layer; Phi only at the final
  swap; or Psi only at the final swap.

Four nested restrictions separate the effects of the available controls:

| Protocol space | Optimized controls | Fixed controls |
|---|---|---|
| $P_{\rm swap}$ (`swaps`) | Shared cutoff | No distillation, all corrected |
| $P_{\rm D}$ (`distillation`) | Cutoff and distillation count/placement | All corrected |
| $P_{\rm A}$ (`postselection`) | Cutoff and Bell-outcome acceptance | No distillation |
| $P_{\rm DA}$ (`joint`) | All listed controls | Fixed swapping tree |

After deduplication, the joint space has 36, 90, 72 and 72 protocols for
$N-1=2,3,4,8$, respectively. The larger count for three links arises because
its near-balanced tree gives additional distinct intermediate-level
distillation placements. The cutoff-only space has exactly three protocols
for every chain size.

Section IV-B of the QCNC paper uses Bayesian optimization to investigate
large protocol spaces. **Here we use brute-force evaluation of a restricted
space.** Every model evaluates the same protocols, so a disagreement cannot
be attributed to different stochastic search trajectories or shot budgets.
An optimum is always qualified by the protocol space over which it is found.

### Figure of merit and loss from model-based optimization

As in Section IV-A of the QCNC paper, the figure of merit is the BB84
secret-key rate (SKR), combining the quality of the end-to-end link and the
average waiting time needed to distribute it. We first pool the delivered
states over their completion times, then compute

$$
e_X=\frac{1-c_{XX}}2,\qquad e_Z=\frac{1-c_{ZZ}}2,\qquad
\mathrm{SKF}=\max\{0,1-h_2(e_X)-h_2(e_Z)\},\qquad
R=\frac{\mathrm{SKF}}{\mathbb E[T]}.
$$

Here $h_2$ is binary entropy. This is a pooled, asymptotic BB84 figure of
merit; it does not average the separately clipped SKF of individual time
bins. Hardware failures, rejected Bell outcomes and distillation retries
already contribute to $\mathbb E[T]$. No additional success factor is
multiplied into the rate. Unless indicated otherwise, rates below are in
**bits per second**.

For a hardware regime $h$ and a protocol space $P$, let $R_{\rm full}(\pi;h)$
denote the rate computed with the full real-X model and $R_m(\pi;h)$ the
prediction of approximation $m$. We compare

$$
\pi^*_{\rm full}\in\arg\max_{\pi\in P}R_{\rm full}(\pi;h),\qquad
\widehat\pi_m\in\arg\max_{\pi\in P}R_m(\pi;h),
$$

$$
\Delta R_m=R_{\rm full}(\pi^*_{\rm full};h)
              -R_{\rm full}(\widehat\pi_m;h),\qquad
\ell_m=\frac{\Delta R_m}{R_{\rm full}(\pi^*_{\rm full};h)}.
$$

Both rates in the loss are evaluated on **the original source and hardware**.
The comparison therefore measures the cost of selecting a protocol using
the approximation, not the difference between two predicted maxima. The
full model is our reference simulation, not an experimental ground truth.

If some evaluations are missing, the implementation replaces $P$ by the
subset on which both models converge. Such comparisons are labelled
**provisional** below: their loss need not equal the loss over the full $P$.
The physical execution score is already available from evaluating every
candidate under the full model; no additional approximate-to-physical
conversion is applied to that score.

Exact prediction ties are broken by candidate ID, without consulting the
physical rate. We also report the minimum and maximum physical loss over
near-optimal prediction ties, using relative tolerance $10^{-4}$ and absolute
tolerance $2\times10^{-9}$ bits/slot. These are sensitivity ranges, not
statistical confidence intervals. A model predicting zero key for every
protocol has a tied optimum; this alone does not establish an unavoidable
optimization loss. Relative loss is undefined when the full-model optimum
is zero.

### Hardware regimes and initial link quality

The grid crosses $N\in\{3,4,5,9\}$, three source families, two initial
fidelities, four memory scales and two physical noise models: **192 regimes**.

| Parameter | Values |
|---|---|
| Internode distance $L_0$ | 5 km |
| End-to-end distance $(N-1)L_0$ | 10, 15, 20, 40 km |
| Elementary generation success $p_{\rm gen}$ | 0.0036793 per attempt |
| Attempt duration $\delta t$ | 35 microseconds |
| Swapping hardware success $p_{\rm swap}$ | 0.85 |
| Distillation hardware success | 1; state-dependent acceptance remains probabilistic |
| Initial Phi+ fidelity $F_0$ | 0.97, 0.99 |
| Memory scale $s$ | 0.05, 0.2, 1, 2 |
| Local memory times | $(T_1,T_2)=s(2.4,1.71)$ seconds |

The optical calibration retains a generation prefactor fitted to the C/D
benchmarks used with the QCNC framework, while shortening the elementary
span to obtain positive rates. The measured passive-fibre loss reference,
memory reference and source-fidelity references are documented in
[the metropolitan calibration note](../metropolitan-reference.md). This is an
optimistic hybrid hardware model; its components have not been demonstrated
together. The QCNC paper's scalar $t_{\rm coh}$ is a Werner decay parameter
with its stated local/joint convention. We do **not** identify it directly
with either of these local $T_1,T_2$ values.

Writing $\Phi^\pm$ for the corresponding Bell projectors, the sources are

$$
\begin{aligned}
\rho_{\rm W}&=F_0\Phi^+ +\frac{1-F_0}{3}(I-\Phi^+),\\
\rho_Z&=F_0\Phi^+ +(1-F_0)\Phi^-,\\
\rho_R&=F_0\Phi^+ +(1-F_0)|01\rangle\langle01|.
\end{aligned}
$$

The three families have the same Phi+ fidelity but different error
composition or local biases. Their preparation costs are held fixed. They
are controlled input families, not a claim that a particular interface
produces all three at the same rate. Partially entangled sources and
optimization of their imbalance are not part of this batch.

### Physical noise and approximated models

The AD model combines relaxation toward $|0\rangle$ and pure dephasing. In
physical time, its local Pauli mode rates and AD rate are

$$
\kappa=1/T_1,\qquad
(\Gamma_X,\Gamma_Y,\Gamma_Z)
=\left(1/T_2-1/(2T_1),1/T_2-1/(2T_1),0\right).
$$

The unital Pauli control has rates $(1/T_2,1/T_2,1/T_1)$ and $\kappa=0$.
Thus both have the same Bloch contractions
$(e^{-t/T_2},e^{-t/T_2},e^{-t/T_1})$; only the AD model has the affine shift
$z\mapsto e^{-t/T_1}z+1-e^{-t/T_1}$. Rates are converted to slots in the
implementation. The reference memory measurements motivate the decay times,
not the assumed ground-state fixed point of AD.

Let $B$ denote Bell-diagonal state twirling and $W$ fidelity-preserving
Werner state twirling. We distinguish six models:

| Model | Source replacement | Waiting noise | Replacement after each successful operation |
|---|---|---|---|
| `full` | None | Physical | None |
| `bell_source` | $B(\rho_0)$ | Physical | None |
| `werner_source` | $W(\rho_0)$ | Physical | None |
| `channel_twirl` | None | Local Pauli-channel twirl | None |
| `bell_channel_twirl` | $B(\rho_0)$ | Local Pauli-channel twirl | None |
| `bell_each_stage` | $B(\rho_0)$ | Physical | Apply $B$ |

For general Pauli–AD rates, the local channel twirl replaces
$(\Gamma_X,\Gamma_Y,\Gamma_Z,\kappa)$ with
$(\Gamma_X+\kappa/2,\Gamma_Y+\kappa/2,\Gamma_Z+\kappa,0)$.
It preserves the contractions while removing the translation.

In particular, `werner_source` is **only an input-state approximation**;
it still propagates real-X states during waiting and distillation. It is
not the QCNC paper's assumption of Werner states at every stage. Likewise,
`bell_each_stage` permits biases to regenerate during storage intervals.
The combined `bell_channel_twirl` is the closed Bell-diagonal surrogate.
None of these prediction substitutions physically twirls the input of the
executed protocol.

## 2. Exactness controls

Davies, Avis and Wehner establish Bell-population equivalence for
non-postselected swap-and-correct protocols when replacing the input states
by their Bell-diagonal approximations
([Theorem 1](https://arxiv.org/html/2509.16689v1)). This provides a control
for the present study, rather than a competing explanation for all its
noise and operation settings.

For the implemented real-X bilateral-CNOT recurrence, let
$(u,z,x,y)=(c_{II},c_{ZZ},c_{XX},c_{YY})$ retain the success weight $u$.
The unnormalized output Bell sector is

$$
\begin{aligned}
u'&=(u_au_b+z_az_b)/2,&z'&=(z_au_b+u_az_b)/2,\\
x'&=(x_ax_b+y_ay_b)/2,&y'&=(y_ax_b+x_ay_b)/2.
\end{aligned}
$$

No local biases occur. Consequently,
$B D(a,b)=D(Ba,Bb)$, including the success probability. All-corrected
swapping also has a closed Bell sector, and local Pauli noise commutes
with $B$. The same identities hold for success-weighted time bins, time-only
cutoffs and retries. **Bell reduction therefore preserves the waiting-time
distribution and pooled BB84 SKR for all-corrected protocols under Pauli
storage, including this recurrence.** This argument is restricted to the
implemented state and operation family, not arbitrary purification maps.

The stored Pauli-noise diagnostics agree: across **886 resolved
all-corrected candidate pairs per Bell-source/stage model**, the maximum
fidelity discrepancy is zero and the maximum rate discrepancy is about
$1.1\times10^{-19}$ bits/slot. These are within-job candidate counts, not
independent statistical replicates.

There are two relevant boundaries. A local affine channel gives

$$
c'_{ZZ}=a_La_Rc_{ZZ}
 +a_Lb_Rc_{ZI}+b_La_Rc_{IZ}+b_Lb_Rc_{II}.
$$

Discarded local biases can therefore matter under subsequent non-unital
storage; channel twirling also removes the last term. Moreover, acceptance
probabilities of the Phi/Psi pairs depend on the local-bias product. Thus
postselection can expose information lost by state twirling even without
AD. Conversely, replacing an already Bell-diagonal source by $B(\rho_0)$
is an identity even when later storage is non-unital. This trivial control
must not be mistaken for evidence that intermediate Bell reduction is exact.

The recurrence here is the untwirled bilateral-CNOT core with equal target
outcomes accepted. It is not the complete BBPSSW procedure with Werner
preprocessing. A loss from a Werner-predicted protocol is not evidence of
an advantage over physically twirled or basis-adapted purification.

## 3. Numerical results

![Physical performance of the full-model and approximation-selected protocols. The top row contains complete Werner-source cutoff comparisons; the bottom row contains provisional Werner-source and channel-twirl joint comparisons, respectively. Every bar is a full-model physical rate.](twirled-model-optimization/physical-rate-comparisons.png)

### A. A Werner-selected cutoff can reduce the rate without distillation

Consider $N=5$ nodes, $F_0=0.97$ phase-error sources and $s=0.2$, so
$(T_1,T_2)=(0.48,0.342)$ seconds. The swapping tree is balanced and all Bell
outcomes are corrected and accepted. Only the shared cutoff is optimized.
All three candidate cutoffs converge under both models.

| Physical waiting noise | Full-model cutoff | Werner-selected cutoff | Full-model optimum | Physical rate of Werner-selected protocol | Loss |
|---|---:|---:|---:|---:|---:|
| Pauli | 544 slots | 136 slots | 3.97945 | 3.07636 | 22.69% |
| AD plus dephasing | 544 slots | 136 slots | 4.04823 | 3.07926 | 23.94% |

Rates are bits/s. The AD case makes the quality–waiting-time trade-off explicit:

| Physical protocol | SKF | Average waiting time | Average Phi+ fidelity | SKR |
|---|---:|---:|---:|---:|
| Cutoff 544 | 0.138063 | 34.10 ms | 0.826428 | 4.04823 |
| Cutoff 136 | 0.353419 | 114.77 ms | 0.868598 | 3.07926 |

The shorter cutoff produces higher-quality links, but their average waiting
time is more than three times larger. The resulting SKF increase does not
compensate for the additional retries. The fidelity-only source description
therefore selects an excessively restrictive cutoff for the physical
phase-error source.

This is a **complete finite-space result**: $3/3$ protocols are resolved and
the approximate optimum is unique at the configured tie tolerance. It does
not establish optimality among arbitrary cutoffs or swapping trees. The
similar loss under Pauli noise shows that the main mechanism is loss of
error-composition information, not exclusively non-unitality.

### B. Werner-selected distillation can be detrimental

For the same $N=5$ and phase-error source with $F_0=0.97$, now take reference
memories $s=1$ and AD plus dephasing. In the resolved joint subset, the full
model selects no distillation, whereas the Werner-source model selects one
round on each two-elementary-link output, before the final swap. Both
protocols accept all corrected outcomes and use no cutoff.

| Physical protocol | $\mathcal K$ | SKF | Average waiting time | SKR |
|---|---:|---:|---:|---:|
| Balanced swapping without distillation | 0 | 0.357348 | 27.73 ms | 12.88463 |
| One intermediate-level round on each half | 2 | 0.192936 | 37.26 ms | 5.17824 |

The observed reduction is **59.81%** in the common converged subset.
Distillation increases $e_X$ from 0.12410 to 0.21490, while reducing $e_Z$
from 0.01325 to 0.00646. For this source, the reduction in bit errors is
insufficient to compensate for the increase in phase errors. Both SKF and
delivery efficiency deteriorate.

This is the QCNC paper's trade-off between improved quality through
distillation and additional waiting time, with a further issue: a
fidelity-matched source approximation can misidentify whether the chosen
recurrence improves the relevant quality at all. There is no universal
conclusion that intermediate-level distillation is detrimental; its utility
depends on source error composition and the recurrence primitive.

**Qualification:** only **29/72** candidate pairs converge for this joint
comparison. The saved choices also match each model's own optimum over all
its converged candidates, not just the intersection. Nevertheless, an
unresolved candidate could change either optimum. The two displayed rates
are converged evaluations; 59.81% is not yet a certified loss for the entire
joint space. A comparable Pauli-noise case gives 59.84% on a converged subset.

### C. Channel twirling can change distillation placement

Consider $N=9$ nodes, Werner sources with $F_0=0.99$, $s=0.2$, and AD plus
dephasing. Both compared protocols use cutoff 544 slots and accept all
corrected outcomes. The full model selects one round on each four-link
output before the final swap. The locally Pauli-twirled channel model instead
selects one round on the end-to-end link.

| Physical protocol | $\mathcal K$ | SKF | Average waiting time | SKR |
|---|---:|---:|---:|---:|
| One round on each half, then final swap | 2 | 0.068353 | 290.25 ms | 0.235498 |
| Final swap, then one end-to-end round | 1 | 0.078611 | 335.37 ms | 0.234403 |

The latter has a larger SKF but a longer average waiting time. Its physical
rate is **0.4648% lower**, an absolute difference of approximately
0.001095 bits/s. This is a decision change caused by the memory-channel
approximation: the source is already Werner and is identical in both models.
Postselection is absent from both selected protocols.

The predicted optimum is unique at the stated tie tolerance. Both physical
rates converge at horizon 262,144; their changes between the last two
horizons are below $7.2\times10^{-14}$ bits/slot, compared with a rate
separation of $3.83\times10^{-8}$ bits/slot. The displayed separation is
therefore much larger than the observed successive-horizon change. This
is a numerical stability check, not a rigorous bound on all tail errors.

**Qualification:** only **31/72** candidate pairs converge. Again, the
reported choices coincide with each model's own best converged candidate,
but the complete joint-space optimum remains unresolved. The result is a
small, provisional channel-induced decision loss, not evidence of a large
general advantage of full-state optimization. The two protocols also use
different total distillation counts; the objective is not normalized by
the number of memory qubits.

### D. Dependence on hardware noise and chain size

At fixed phase-error source $F_0=0.97$ and scale $s=0.2$, the complete
cutoff-only comparisons give:

| Elementary links $N-1$ | Werner-policy loss under Pauli noise | Werner-policy loss under AD |
|---:|---:|---:|
| 2 | 0% | 0% |
| 3 | 0% | 0% |
| 4 | 22.69% | 23.94% |
| 8 | Unresolved | Unresolved |

![Loss versus chain size with source fidelity, memory scale, elementary span and optimized controls fixed. Eight-link results are omitted because the cutoff-only comparisons are unresolved.](twirled-model-optimization/cutoff-loss-by-size.png)

This establishes that the approximation can preserve the preferred cutoff
in smaller chains and change it in a larger chain under otherwise matched
conditions. It does not establish monotone scaling. For example, at four
links and $s=1$ the corresponding complete cutoff-only losses are only
8.15% under Pauli noise and 8.22% under AD. Source family matters as well:
three-link R-state sources at $s=0.2$ give losses of 1.97% and 1.23%,
respectively. AD does not universally amplify the loss.

Increasing $N$ here also increases the end-to-end distance, since $L_0$ is
fixed. A fixed-total-distance study would change generation probabilities
and slot duration and must be reported separately.

Across the complete comparisons, counting a material loss as a minimum
near-tie loss above 0.01%:

| Protocol space | Werner source: complete / positive reference / material loss | Channel twirl: complete / positive reference / material loss |
|---|---:|---:|
| $P_{\rm swap}$ | 144 / 140 / 6 | 144 / 140 / 0 |
| $P_{\rm D}$ | 30 / 30 / 0 | 34 / 34 / 0 |
| $P_{\rm A}$ | 86 / 82 / 2 | 74 / 70 / 0 |
| $P_{\rm DA}$ | 0 / 0 / 0 | 0 / 0 / 0 |

These nested spaces are not independent observations. The resolved hardware
subsets also differ between models and spaces. In particular, zero complete
joint comparisons means **no resolved sample**, not zero joint-space loss.
Current data show that prediction error is more common than a change in
the preferred protocol. For example, resolved all-corrected candidates under
AD show channel-twirl fidelity errors as large as 0.0902, while none of the
complete all-corrected comparisons has a material channel-induced decision
loss. This does not constrain the unresolved cases.

## 4. Numerical completeness and scope of the conclusions

Each candidate is evaluated using the standalone real-X waiting-time
algorithm. Two successive horizons must both cover at least 0.999999 of
the completion distribution, with stable SKR, SKF, average waiting time,
$e_X$ and $e_Z$ (relative tolerance $10^{-5}$, absolute tolerance $10^{-9}$).
Horizons start at 16,384 and double up to 1,048,576 slots.

At the snapshot, the completed 158 jobs contain **63,072 candidate/model
evaluations**:

| Numerical status | Count | Share |
|---|---:|---:|
| Converged | 42,302 | 67.07% |
| Failed physicality validation | 10,752 | 17.05% |
| Horizon limit reached without certification | 10,018 | 15.88% |

The recurring failure message is `selected swap input has materially negative
populations`. It is an evaluation failure, not evidence that the physical
protocol has zero key. Its cause has not been diagnosed in this reporting
exercise. A convergence exception can terminate evaluation at an early
horizon; increasing the horizon cap alone is not an established fix.
Unconverged evaluations likewise remain missing rather than being assigned
zero SKR. When candidate pairs are missing, optimization is explicitly
restricted to the common converged subset.

The completed jobs comprise all 48 regimes at each of two, three and four
links, and 14/48 at eight links. Mean elapsed times per job were 2.18,
11.68, 17.30 and 48.23 minutes, respectively, with two workers. These are
observed job timings, not algorithmic scaling estimates; different coverage
requirements and early failures also affect them.

The strongest currently complete optimization result is the Werner-induced
cutoff loss. The large Werner-induced distillation loss and the smaller
channel-induced placement change are useful concrete examples, but require
resolution of the remaining protocols before quoting full-space optimality.
No conclusion about an eight-link scaling law or global protocol optimum is
justified yet.

As in the QCNC framework, the implemented recurrence assumes that the
required parallel copies can be prepared. The comparison includes their
waiting-time cost but does not fix a common total memory-qubit budget.
Additional gate errors, operation durations, deliberately applied twirls,
basis-adapted purification and finite-key effects are outside this batch.

## 5. Relation to the QCNC study and earlier evidence

The QCNC study asks which swapping and distillation sequence maximizes
the figure of merit under given hardware conditions. The present question
is complementary: **does the model used to investigate the protocol space
preserve the identity of that optimum?** Its tree notation, placement
language and quality–waiting-time interpretation remain applicable.
The Werner closure and noise assumptions, and the use of Bayesian
optimization, must not be transferred to this experiment without qualification.

Davies et al.'s Bell-population equivalence supplies an exact control, while
[Mondal et al.](https://arxiv.org/abs/2509.18064) already compare full AD with
its Pauli-twirled counterpart for fixed repeater policies. The contribution
sought here is the physical rate consequence of **selecting a different
repeater protocol** under an approximation, with matched hardware, explicit
protocol spaces and zero-loss controls. This note does not establish
literature priority for that contribution.

The earlier 59.3% two-link and 54.2% four-link Werner examples, 13.8%
cutoff example, and 4.94% lifetime-threshold shift belong to separate
exploratory experiments documented in [analysis.md](../analysis.md). They
motivate this grid but are not additional results or replications of it.

## Evidence and reproduction

The optimizer and runner are
[`optimization/twirled_regret.py`](../../optimization/twirled_regret.py) and
[`scripts/twirled_regret.py`](../../scripts/twirled_regret.py). Exact model
definitions and launch options are in
[`TWIRLED_REGRET.md`](../../optimization/TWIRLED_REGRET.md).

Representative input summaries, all relative to
`results/twirled-regret/noise-size/`, are:

| Result | Saved summary |
|---|---|
| Complete AD cutoff loss | `L4_phase_error_F0.97_ad_s0.2_5km_68aa81d4/summary.json` |
| Complete Pauli cutoff loss | `L4_phase_error_F0.97_pauli_s0.2_5km_6fc1f3f0/summary.json` |
| Provisional Werner distillation loss | `L4_phase_error_F0.97_ad_s1_5km_36d8a11f/summary.json` |
| Provisional channel-twirl placement loss | `L8_werner_F0.99_ad_s0.2_5km_74aa41d7/summary.json` |

The evidence bundle contains the extracted comparisons and diagnostics for
all completed jobs, the detailed representative results, and 170 input
checksums, including the QCNC source manuscript. Live manifest files may
advance after extraction. This report does not modify the batch, its
configuration or its evaluations.

Regenerate figures from the frozen evidence, without evaluation:

```bash
source venv/bin/activate
python -m src.analysis.twirled_model_report --replot
```

To extract a later snapshot, use a new output directory with
`python -m src.analysis.twirled_model_report --output PATH`; update the
report's timestamp, tables and qualifications to match that snapshot.
The extractor reads saved outputs only and does not call the evaluator or
optimizer.

## References

1. L. La Corte, K. Goodenough, A. G. Maity, S. Santra and D. Elkouss,
   [*Bayesian Optimization for Repeater Protocols*](https://arxiv.org/abs/2502.02208),
   QCNC 2025, [doi:10.1109/QCNC64685.2025.00029](https://doi.org/10.1109/QCNC64685.2025.00029).
   Local terminology and notation: Sections III, IV-A/B and V of
   [`main.tex`](</Users/lorenzolacorte/Research/qcnc2025-paper/main.tex>).
2. B. Li, T. Coopmans and D. Elkouss,
   [*Efficient optimization of cutoffs in quantum repeater chains*](https://doi.org/10.1109/TQE.2021.3099003),
   IEEE Transactions on Quantum Engineering 2 (2021).
3. B. Davies, G. Avis and S. Wehner,
   [*On the accuracy of twirled approximations in repeater chains*](https://arxiv.org/html/2509.16689v1),
   arXiv:2509.16689v1, especially Theorem 1.
4. S. Mondal, P. Halder, S. Haldar and A. Sen De,
   [*Efficient formulation of quantum network under amplitude damping noise:
   Highlighting benefits over its Pauli-twirled counterpart*](https://arxiv.org/abs/2509.18064),
   arXiv:2509.18064.
5. Hardware calibration and primary fibre, memory and fidelity references:
   [*A metropolitan reference with positive-key baselines*](../metropolitan-reference.md).
