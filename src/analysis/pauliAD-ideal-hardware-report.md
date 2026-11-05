---
title: "What the first Pauli–AD optimization batch teaches us"
date: "27 October 2026"
header-includes:
  - \usepackage{needspace}
  - \usepackage{etoolbox}
  - \BeforeBeginEnvironment{longtable}{\Needspace{5\baselineskip}}
---

**The strongest contribution candidate is choosing where in the tree to
postselect, rather than a general advantage of partially entangled sources.**
The results below are ranked by scientific interest and estimated novelty.
Novelty assessments reflect a targeted primary-source literature check, not
established priority.

**Scope.** Four/eight elementary links (five/nine nodes), homogeneous hardware,
$p_{\mathrm{gen}}=0.2$, ideal instantaneous swap/distillation hardware, and
amplitude damping during waiting: $T_2=2T_1$. Sources are
$|\psi_q\rangle=\sqrt{1-q}|00\rangle+\sqrt q|11\rangle$; Bell means $q=1/2$,
PE here means $q=1/4$. The grid is $T_1\in\{25,74,200,600\}$ slots, with
optimizer seeds 7 and 42. Our metric is

$$
R=\mathrm{SKR}=\frac{\max\{0,1-h_2(e_X)-h_2(e_Z)\}}{\mathbb E[T]}.
$$

Rejection, regeneration and distillation preparation are included in $T$.
There are no cutoffs, preparation damping, classical-message delays or
normalization by memory count. Distillation uses the existing recurrence,
without Werner twirling. Placement experiments fix a balanced tree and no
distillation; protocol searches also vary the tree and allow one recurrence
round per tree vertex; their acceptance policy remains uniform. Rates below
are bits/slot.

## 1. The useful postselection level depends on the source and noise regime

**Novelty estimate: medium as a specific network-rate result.** Entanglement
concentration through swapping is established: [Song et al. (2014)](https://arxiv.org/abs/1306.4451)
study recovery under amplitude damping, and [Davies et al., Sec. IV](https://arxiv.org/html/2509.16689v1#S4)
show outcome-dependent concentration for PE inputs. Our additional result is
the optimized *placement* and its throughput after noisy waiting and retries.
Heralded AD recovery using an extra reversal operation is also studied by
[Jeong et al. (2026)](https://arxiv.org/abs/2602.21563).

![Optimal level schedules and their rate advantage over the best uniform policy. “All” accepts all corrected outcomes; $\Psi$ accepts the odd-parity pair. Each ratio uses the same source and tree on both sides.](pauliAD-overnight-report/1-placement.png){width=100%}

For **four Bell links at $T_1=74$**, the exact optimum among all 27 schedules
accepts all first-level outcomes and selects $\Psi$ only at the root:
$R=0.00027212\to0.00039563$ (**+45.4%**). SKF rises
$0.002677\to0.007820$, outweighing the waiting-time increase
$9.84\to19.77$ slots. This confirms and certifies the earlier notebook example.

For **PE links**, select $\Psi$ at the first level and accept all outcomes
afterwards. Four links at $T_1=200$ gain **2.00×** over the best uniform policy.
Eight links at $T_1=600$ gain **12.16×**:
$R=0.00055142\to0.00670679$, SKF $0.08849\to0.25552$, and wait
$160.48\to38.10$ slots. Here avoiding later rejection improves both quality
and delay. The interpretation is concentration early for imbalanced sources,
versus filtering accumulated waiting noise late for Bell sources; this is a
mechanistic explanation, not a separately isolated causal test.

The 12× comparison is against the **same PE source**. Bell sources accepting
all outcomes already reach $0.04128$ at that eight-link point. It does not
establish PE superiority, or a scaling law from two chain lengths.

## 2. Terminal distillation can dominate postselection, even with fresh perfect Bell links

**Novelty estimate: medium for this quantitative regime; low for the general
swap-then-distill idea.** [Haldar et al., Fig. 7 and distillation discussion](https://arxiv.org/html/2401.13168v3)
already compare both orders. [La Corte et al., Sec. V.A](https://arxiv.org/html/2502.02208v1#S5.SS1)
find early distillation in their studied Werner-source regimes. Our AD-memory
example illustrates a different optimum; it does not contradict their
parameter-specific conclusions.

![Matched four-link Bell protocols: the same balanced tree, source and acceptance, with or without one terminal recurrence. Labels give the rate ratio with/without recurrence. These are saved trial evaluations, not new simulations.](pauliAD-overnight-report/2-terminal-distillation.png){width=100%}

At $T_1=74$, appending one end-to-end recurrence to the balanced Bell protocol
gives $R=0.00027212\to0.02541359$ (**93.39×**). SKF rises
$0.002677\to0.44044$, while mean wait increases only
$9.84\to17.33$ slots. It also beats the best no-distillation postselection
schedule by **64.24×**. At $T_1=200$, the matched gain is 27.1%; at $600$,
terminal recurrence instead lowers SKR by 4.79%, despite improving SKF.

Fresh links start perfect, so waiting creates the errors that later
purification removes. The useful choice depends on how much key fraction
there is to recover. This is the strongest practical warning against
optimizing postselection in isolation. Extra parallel-copy resources are
allowed by the evaluator, so this is not a memory-normalized advantage.

## 3. One policy per level captures every best per-swap rate found

**Novelty estimate: low-to-medium, as a restricted search-space reduction
conjecture.** [Li et al. (2021)](https://arxiv.org/abs/2005.04946) found little
benefit from nonuniform cutoffs in their symmetric chains. Here nonuniform
postselection can matter greatly, but variation *between levels* appears
sufficient on the tested homogeneous balanced trees.

![Per-level versus per-swap best rates, and sizes of the three policy spaces. Overlapping points include the two optimizer seeds. Equality for eight links is a search observation, not an exhaustive certificate.](pauliAD-overnight-report/3-level-compression.png){width=100%}

The best level and per-swap rates agree at every tested setting: 16 distinct
size/lifetime/source points, each searched with two seeds. For four links,
all 27 per-swap schedules were evaluated, so equality is established
numerically on those eight parameter points. For eight links, the 27 level
schedules were exhaustive, but per-swap GP searched only a fraction of 2187
schedules and was seeded with the level optimum. Equality there is weaker
evidence.

**Working hypothesis:** use one policy per nesting level before searching
spatial asymmetry. This reduces the eight-link policy space by 81×. Symmetry
of the hardware alone is not a proof that a symmetry-breaking policy cannot
win; heterogeneous or unbalanced chains remain untested.

## 4. Optimizing $q$ gives small matched gains; larger apparent gains can come from the tree

**Novelty estimate: low as a physical principle; useful as a negative result
and an attribution check.** [Wang et al.](https://arxiv.org/abs/1309.3818)
establish PE robustness for single-pair correlation measures. They do not
imply a repeater SKR advantage. The present contribution is testing that
tradeoff under multilevel waiting and protocol optimization.

![Left: rate change from the original joint winner to its Bell-source control with exactly the same operations. Negative values mean Bell sources improve that protocol. Right: a larger apparent search gain is explained by a different tree.](pauliAD-overnight-report/4-q-attribution.png){width=100%}

All positive-key **reported joint winners** use $q\geq0.495$. For four links
at $T_1=600$, $q\simeq0.4952$ gives only **0.0771%** over the matched Bell
protocol, consistently across seeds. No positive joint winner uses uniform
Bell-outcome rejection.

For eight links, $T_1=200$, seed 7, the reported joint rate exceeds the Bell
search winner by **13.88%**. Both reported winners actually use Bell sources,
all outcomes, and one terminal recurrence: the difference is the swap tree.
The original joint candidate had $q=0.4969$; resetting it to $1/2$ raises its
rate further, from $0.02680256$ to $0.02685390$. Thus the large gain cannot
be credited to PE sources. Different seeds also find different trees, so
the best-found rates should not be called global optima.

## Evidence limits and next decision

The batch completed 48 jobs and 8120 distinct backend evaluations. Of these,
7928 converged; 136 lacked convergence and 56 failed physicality checks.
They were excluded, not treated as zero key. All reported winners passed
coverage and successive-horizon checks. Two seeds are search replicates,
not a statistical confidence interval. The coarse four-point lifetime grid
demonstrates no new positive-key boundary or threshold scaling law.

**Priority for follow-up:** refine the source-dependent level schedules
around the key boundary, then compare them jointly with terminal recurrence
at fixed resource assumptions. A broad claim that PE sources improve the
optimized network, or that per-level policies are always optimal, is not
supported by this batch. The two leading findings merit further study;
none of these novelty estimates establishes a first-in-literature claim.

**Provenance.** All plots use existing files in
`results/pauliAD/overnight/`; no evaluator was run for this report. Exact
numbers, matched trial IDs, figure data and input hashes are saved in
`pauliAD-overnight-report/`, beside this document. Regenerate figures from
the repository root:

```bash
python -m src.analysis.plot_pauli_ad_overnight
```

Literature checked on 27 October 2026.

