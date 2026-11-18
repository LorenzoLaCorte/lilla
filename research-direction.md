# Research direction: jointly optimizing repeater protocols under non-unital noise

Research assessment, 14 November 2026. This note proposes experiments and optimizer changes; it does not report new optimization runs. Local evidence comes from the QCNC and TQE manuscript sources, `lilla-overleaf/main.tex`, this repository, and the archived PDFs in `src/analysis-i`.

## What the source parameter means

The metropolitan report uses

\[
|\psi_q\rangle=\sqrt{1-q}|00\rangle+\sqrt q|11\rangle,
\qquad \rho_{q,v}=v|\psi_q\rangle\langle\psi_q|+(1-v)I/4.
\]

Here `q` controls Schmidt imbalance and `v` controls visibility. Its source preparation cost and generation probability are independent of `q`. Consequently, those experiments investigate source shape under stochastic storage and processing, rather than an experimentally calibrated elementary generation rate–fidelity curve. Varying `q` can still change the end-to-end delivery rate through state-dependent acceptance and retries.

In Promponas et al., **Rate-Fidelity Tradeoffs in All-Photonic and Memory-Equipped Quantum Switches**, Sec. II-C, `beta` is a physical source control: both generation probability `p_i(beta)` and Werner parameter `w_i(beta)` depend on it. It is not an objective weight. Their abstraction does not supply a universal map from Schmidt imbalance to generation probability. [Paper](https://arxiv.org/html/2603.02610v1#S2.SS3).

Use distinct names for distinct decisions:

| Decision | Suggested name | Meaning |
|---|---|---|
| Source shape | `source_q` | Population of `11` in the pure reference source |
| Physical source brightness | `source_brightness` | Hardware control determining generation probability and delivered state |
| Memory cutoff | `memory_cutoff` | Maximum allowed readiness-time difference at a join |
| Swap acceptance | `bell_outcomes` | Which available corrected BSM branches are retained |
| Application quality requirement | `min_fidelity` | Constraint on the delivered ensemble |

Do not identify brightness with `source_q` without a source-specific derivation. A model may have both controls, or a characterized one-dimensional relationship between them.

For the above mixed source, its initial target fidelity is

\[
F_0(q,v)=v\left(\tfrac12+\sqrt{q(1-q)}\right)+(1-v)/4.
\]

Holding `v` fixed therefore does not hold initial fidelity fixed. Under amplitude damping toward `0`, `q` and `1-q` also describe physically different orientations. Searching only `q <= 1/2` deliberately restricts the source to ground-biased states; it is not a symmetry argument establishing the unrestricted optimum.

## References directly supporting the physics

1. **Bose, Vedral and Knight (1999), Purification via entanglement swapping and conserved entanglement**, Physical Review A 60, 194. Equations (2)–(3) give the Bell and non-Bell output branches and their probabilities for two identical partially entangled inputs. This is a direct reference for the quality–success-probability trade-off behind the existing `psi_pair` policy. Their parameterization is `cos(theta)|00> + sin(theta)|11>`; set `q = sin(theta)^2`. [Full text](https://arxiv.org/pdf/quant-ph/9812013).

2. **Solís-Prosser, Delgado, Jiménez and Neves (2014), Deterministic and probabilistic entanglement swapping of nonmaximally entangled states assisted by optimal quantum state discrimination**, Physical Review A 89, 012337. Connects deterministic swapping to minimum-error discrimination and probabilistic enhancement to maximum-confidence discrimination. Useful if extending beyond acceptance of a fixed Bell measurement. [Paper](https://arxiv.org/abs/1402.1244).

3. **Wang, Tang, Yuan and Kuang (2013), Nonmaximally entangled states can be better for quantum correlation distribution and storage**. Studies partially entangled two-qubit states under local amplitude damping, including concurrence and fully entangled fraction. Their `u` is the `00` population, so `q = 1-u`. This supports source-shape optimization under non-unital noise; it does not establish a repeater secret-key-rate advantage or a generation probability law. [Full text](https://arxiv.org/pdf/1309.3818).

4. **Masajada, Fellous-Asiani and Streltsov, Optimizing entanglement distribution via noisy quantum channels**, Physical Review A 113, 052414 (2026), initially arXiv:2506.06089 (2025). Certain combinations of amplitude damping and depolarization permit entanglement distribution only for weakly entangled inputs. Relevant motivation for extending the evaluator beyond Bell-diagonal states; its task is distinct from optimizing stochastic repeater throughput. [Published paper](https://doi.org/10.1103/hdzn-fwpj), [preprint](https://arxiv.org/abs/2506.06089).

5. **Guerra, Sarubi, Chaves and Maziero, Entanglement distribution in quantum networks via swapping of partially entangled pure states**, Annalen der Physik 538, e00475 (2026), arXiv:2508.04536. An especially close network reference: partially entangled pure inputs, multiple topologies, and probabilities of producing maximally entangled outputs. This is prior art to address when framing the postselection contribution. [Paper](https://arxiv.org/abs/2508.04536).

6. **Humphreys et al. (2018), Deterministic delivery of remote entanglement on a quantum network**, Nature 558, 268–273. Equations (4)–(5) and the following discussion give a physical single-click source trade-off. In the low-detection-efficiency limit, brightness `alpha` yields `p_gen ≈ 2 p_det alpha` and conditional Bell fidelity `1-alpha`, at a fixed attempt period. The remote state is a mixture of a Bell state and a product state, rather than the pure Schmidt-imbalanced family above. Its initial spin–optical state is partially entangled. [Full text](https://arxiv.org/pdf/1712.07567), [publication](https://doi.org/10.1038/s41586-018-0200-5).

These references support PE concentration and brightness-dependent generation, but do not specify a universal `p_gen(q)`. The following experimental source provides a more relevant, device-specific route.

## Experimental calibration of a coupled source model

Follow-up to the request for an experimental source reference: **Ma et al., Nonlinearity symmetry breaking for generating tunable quantum entanglement in semiconductor metasurfaces**, Science Advances 11, eadu4133 (2025). The [preprint](https://arxiv.org/html/2409.10845v1) has a different title. Its Fig. 3B measures the two cross-polarized pair rates versus pump wavelength; Fig. 3C gives their population ratio, and Fig. 3A,D characterize the state. This is substantially closer to the requested source model than the references above.

Taking the pump wavelength as the control `b`, let `A(b)` and `B(b)` be the background- and efficiency-corrected rates for `HV` and `VH`. In the ideal coherent two-component approximation, a fixed local basis/phase correction maps the state to the project's pure family, with

\[
q(b)=\frac{A(b)}{A(b)+B(b)},\qquad R_{\rm pair}(b)=A(b)+B(b).
\]

Here the basis convention maps `VH` to `00` and `HV` to `11`. The paper's `alpha/beta` denotes a **population** ratio, so `q=r/(1+r)`, rather than `r^2/(1+r^2)`.

For rare events, independently successful pair delivery/loading with probability `eta_link(b)` gives

\[
p_{\rm gen}(b)\simeq R_{\rm pair}(b)\eta_{\rm link}(b)\Delta t\ll1.
\]

This is a proposed conversion from measured source data to the simulator, not a measured memory-link success law. Fit/interpolate both rate curves over the measured range. Detected coincidences must not be identified directly with heralded memory loading. The experiment has finite mixedness; neither an exactly pure state nor the particular white-noise mixture used in the current reports is established by the source reference. Full tomography can inform a better noise model. A coupled rate/quality curve also need not be globally monotone.

## Two analytic checks explaining the mechanisms

For two identical pure sources, ideal complete BSM, immediate correction, and no storage noise:

| Accepted outcomes | Intrinsic probability | Fidelity to corrected `Phi+` |
|---|---:|---:|
| All four | `1` | `1/2 + 2q(1-q)` |
| Both Psi outcomes | `2q(1-q)` | `1` |

These follow from the branch formulas of Bose et al. For `q=1/4`, acceptance of Psi gives unit fidelity with probability `3/8`, while accepting all outcomes gives fidelity `7/8`. This directly expresses a concentration trade-off. At `q=1/2`, all branches are Bell states, so rejecting any branch offers no quality benefit. Independent hardware failure must be applied separately.

For a different mechanism, let each qubit of one pure source undergo amplitude damping with excited-state survival probability `s`. Direct application of the Kraus maps gives

\[
F_{\Phi^+}(q,s)=\tfrac12-q s(1-s)+s\sqrt{q(1-q)}.
\]

For fixed `0 < s < 1`, maximizing this particular fidelity within the family gives

\[
q_* = \frac12\left(1-\frac{1-s}{\sqrt{1+(1-s)^2}}\right)<\frac12.
\]

This illustrates why an initially less entangled, ground-biased state can finish with higher Bell fidelity. It is not a repeater SKR optimum: random storage durations, cutoff rejections, distillation and postselection change the problem. At `s=0` every member has fidelity `1/2`, so the optimum is degenerate.

The swap probabilities/fidelities were checked against `accepted_bell_swap` at `q=0.1, 0.25, 0.5`; the damping fidelity was checked independently with dense Kraus maps at `s=0.2, 0.7, 1`. These checks did not rerun archived network experiments.

## Cutoffs and communication

Li, Coopmans and Elkouss compare uniform and level-dependent cutoffs in Sec. V-C/Fig. 7 of **Efficient optimization of cut-offs in quantum repeater chains**. Uniform cutoffs give only a small or negligible SKR reduction in the studied symmetric regimes. The larger search family contains uniform cutoffs, so uniformity is a useful low-dimensional approximation rather than a superior optimum. Their numerical study excludes distillation and communication latency. [Preprint](https://arxiv.org/abs/2005.04946), [journal PDF, Sec. V](https://pure.tudelft.nl/ws/portalfiles/portal/114421259/Efficient_Optimization_of_Cutoffs_in_Quantum_Repeater_Chains.pdf).

Start with one shared readiness-difference cutoff. Include a genuine no-cutoff candidate, then check the restriction against separate cutoffs by span, depth, or operation type on small chains and selected heterogeneous cases. On an unbalanced tree these parameterizations differ. A readiness-difference cutoff is not a hard bound on total storage over the entire history of a pair; accepted pairs additionally age during communication.

The current repository already charges communication on successful attempts, failed operations and cutoff aborts, and ages both inputs. Elementary generation includes heralding. The next scientific task is to test whether including this timing changes optimized protocol decisions. Compare both evaluation of old choices under the timing model and reoptimization with timing enabled.

For physical calibration, derive delays from distance, propagation speed, operation location and the chosen time unit. The implementation's segment-count delay is a particular timing abstraction. If an elementary attempt includes local overhead, its duration need not equal the elementary propagation time. Likewise, deferred Pauli-frame correction and a protocol that waits for every acknowledgement can have different timing requirements. Specify the supported control model before interpreting a measured rate reduction as universal.

## Objective and experiment design

For continuity with TQE/QCNC, retain pooled asymptotic BB84 SKR as the main objective:

\[
R_{\mathrm{key}}=\frac{[1-h_2(e_X)-h_2(e_Z)]_+}{\mathbb E[T]}.
\]

Compute errors from the delivered state. In a non-Werner model a single fidelity does not determine both errors. Report generation rate, fidelity, both errors and SKR together. Keep the pooled-ensemble convention explicit; sorting by available classical records would define a different processing strategy.

For an application-facing rate–fidelity study, additionally solve

\[
\max_\theta\; 1/\mathbb E[T_\theta]
\quad\text{subject to}\quad F(\bar\rho_\theta)\ge F_{\min}
\]

for several `F_min`. This produces an interpretable frontier of best-found feasible policies without equating a physical source setting with an objective weight. An average-fidelity constraint does not guarantee every delivered pair meets the threshold. A scalar objective weight is optional and must be fixed externally for each optimization; optimizing that weight as a free control does not specify an application preference.

Run a staged comparison:

1. Reproduce selected TQE/QCNC regimes using current operations and clearly documented noise/timing settings.
2. Jointly optimize swap tree, distillation count/placement and one memory cutoff. Compare to no cutoff and to optimizing these decisions separately.
3. Repeat with and without communication, and with non-unital versus matched unital noise. Reevaluate winners across models as well as reoptimizing within each model.
4. Add a shared `source_q`; include the Bell-source boundary as an anchor and reevaluate each source-tuned winner at `q=1/2` with identical operations.
5. Add categorical postselection, initially uniform and then grouped by level/span. Compare fixed PE sources separately from freely selectable sources.
6. If claiming a physical source generation trade-off, repeat with a characterized source model returning generation probability, delivered state and attempt duration together.

Use brute-force checks on small protocol families, repeated BO seeds, equal evaluation budgets, incumbent baselines and convergence checks for finalists. Under-covered evaluations are unresolved, rather than evidence of zero physical rate. Include preparation of extra distillation copies and all rejection/retry latency. State memory-resource assumptions when comparing protocols using different numbers of copies.

The October metropolitan report provides useful swap-only hypotheses: concentrate fixed PE sources at the first level and accept all corrected outcomes later; its reported freely tuned `q` gains are small. Those observations do not establish superiority over optimally used Bell sources, and changed timing requires reevaluation.

The current README states that real-X distillation now implements BBPSSW with input/output Werner twirls. Older raw-CNOT or output-only-twirl distillation results must be recomputed before being used as BBPSSW evidence. The Overleaf real-X derivation and archived reports should also be aligned with the chosen operation. Omitting a standalone twirling-loss study is compatible with accurately specifying the twirls required by the distillation protocol.

## Concrete optimizer work

The production starting point is `src/optimization/gp_asymmetric.py`.

- Expose `RealXState`, its initial coordinates and Pauli/amplitude-damping rates in the CLI and parameter construction. The current CLI accepts only Werner, Bell and Pauli-Fourier states.
- Preserve the existing tree/placement encoding, adding a separately named `memory_cutoff` mapped to the evaluator's `mt_cut` or `cutoff`. The existing GP variable `tau` already controls distillation placement spread.
- Start with a shared `source_q`; construct its real-X coordinates, optionally with visibility. Add a separate source-model interface if probability or attempt time depends on a physical source setting.
- Expose global `bell_outcomes`, then explicitly route a schedule to individual operations. The current asymmetric evaluator passes the same parameter value to every swap; per-operation policies require additional plumbing.
- Replace the protocol-only global cache with a run-scoped cache keyed by the decoded operations, cutoff, source, acceptance schedule, hardware, noise, timing, numerical policy and objective definition. Changing `q` or cutoff while keeping the tree must trigger a new evaluation.
- Save full candidate descriptions and separate physical metrics from numerical status. Current result files retaining only protocol tuples and SKR would lose the new settings.
- Adapt truncation to cutoff/postselection retry tails, retaining convergence diagnostics. The current objective catches failed coverage and caches a zero score; that must not become a reported zero-rate physical result.
- Keep source `q`, brightness, application fidelity thresholds, noise parameters and existing encoding variables distinct in configuration and logs.

The proposed central research question is: **When do non-unital memory noise and finite communication latency change the best-found joint choices of swap tree, cutoff and distillation placement, and when do source shaping and outcome selection add a matched performance gain?** This is a concrete extension of both original papers. A claim of literature priority would require a broader review.
