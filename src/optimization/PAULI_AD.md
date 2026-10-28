# Postselection and source/protocol optimization

Run from the repository root, after `source venv/bin/activate`. These scripts
use the real-X Pauli/AD backend directly, with **no visualization dependency**.
They do not change the existing optimizers or backend.

## Start here

A short, complete batch tests both studies and writes plots and summaries:

```bash
python -m src.scripts.gp_pauliAD --preset smoke --output results/pauliAD/my-smoke
```

A suggested first overnight batch (48 independent jobs):

```bash
caffeinate -i python -m src.scripts.gp_pauliAD \
  --preset overnight --study both --nodes 5 9 --t1 25 74 200 600 \
  --seeds 7 42 --gp-calls 100 --workers 2 \
  --output results/pauliAD/overnight
```

`caffeinate -i` prevents idle sleep on macOS; omit it on Linux. This command
runs in the foreground. Each job writes its own live log. To detach it, prefix
the command with `nohup`, redirect its output to a log, and append `&`.
Do not start a second copy in the same output directory.

Use `--dry-run` first to print the exact job list. There is no fixed runtime
promise: each job has a default one-hour timeout, preserves partial trials, and
the runner continues after failures/timeouts. Use one worker for lower memory
use. Each subprocess limits BLAS/Numba threading to one thread. The runner
prints job completion, while detailed progress is in each `run.log` and
`console.log`.

Ctrl-C stops the active subprocesses and cancels queued jobs. Repeat the
**same command plus `--resume`** to recover an interrupted batch, using the
same code and Python environment.
Resume replays the seeded GP proposals using saved evaluations; it does not
restart the expensive evaluations. Budgets, seeds, physics, and numeric
tolerances must match. Increasing the budget currently requires a new output
directory. Failed candidates remain failed on replay; to retry with a larger
horizon, start a new run with the larger cap. The default overnight preset
also includes three-node chains; the command above concentrates on four and
eight links. Add `3` to `--nodes` for the two-link comparison.

To rebuild batch tables/plots without evaluating anything:

```bash
python -m src.scripts.gp_pauliAD --summarize-only --output results/pauliAD/overnight
```

## 1. Where should we postselect?

### Hardware follow-up to the first overnight report

The new `hardware` preset addresses the report's four questions. It reuses
`reproduce_results.py`'s C/D generation probabilities, swap success (0.85)
and source visibilities (0.960/0.958). Slots are 250/100 microseconds,
respectively, adopting 50/20 km and propagation speed `2e8` m/s.
Its default `network` profile uses effective `T1=2.4 s`, `T2=1.71 s` from
[Bradley et al. (2022)](https://doi.org/10.1038/s41534-022-00637-w), measured
during optical network operations after charge/resonance conditioning.
This is an optimistic, hardware-informed model; it does not include the
cost of those experimental checks or constitute a calibrated device model.

```bash
source venv/bin/activate
caffeinate -i python -m src.scripts.gp_pauliAD \
  --preset hardware --hardware-regimes C --nodes 5 9 \
  --memory-scales 0.5 1 2 --relaxation-models ad unital \
  --seeds 7 --gp-calls 60 --max-dists 2 --max-total-dists 2 \
  --workers 2 --case-timeout 7200 \
  --output results/pauliAD/hardware-followup
```

This is **144 jobs**, with 60 GP proposals per search plus anchors and
matched controls. `--dry-run` prints the commands. Repeat the same command
with `--resume` after interruption. Runtime depends strongly on retry
probabilities and horizon; start with one worker if RAM is constrained.
The two-million-element FFT work arrays at the largest horizon can be large.
No complete overnight batch is launched automatically.

The measured lifetimes do not identify the relaxation fixed point.
`ad` assumes decay toward `|0>`; `unital` has unbiased longitudinal
relaxation with the same T1/T2, implemented by existing Pauli rates
`(1/T2,1/T2,1/T1)` and zero AD. Treating these as separate models prevents
claiming a ground-state AD advantage from T1/T2 measurements alone.
`--memory-scales` multiplies both lifetimes: smaller means stronger noise.
The alternative `--memory-profiles benchmark` sets local `T2=2*t_coh`
to match the old homogeneous scalar link XX decay, with explicitly assumed
`T1=10*T2`; it does not equate the full channels or claim measured values.

Choose any subset using `--sections placement distillation asymmetry source`:

| Section | Evaluation | Figures |
|---|---|---|
| `placement` | Bell and PE (`q=.25`), benchmark visibility, balanced tree, no distillation; uniform/level/node policies | Bell row then PE row: schedules + gain ratios; companion absolute-rate plot |
| `distillation` | Bell only; best swaps-only vs best with distillation; perfect, benchmark-mixed, source-AD deliveries | SKR vs lifetime with winning K; best sampled rates separated by total K |
| `asymmetry` | Same balanced tree, either one 5x slower elementary link or one 4x shorter-lived internal memory | Uniform/level/node SKR for each separate defect |
| `source` | Bell vs optimized q, both protocols optimized; perfect and source-AD deliveries | SKR vs lifetime and matched-protocol SKR vs q |

Default `--source-dampings .05 .15` are independent AD probabilities per
qubit **at delivery**, after any visibility mixing. They model an input
imperfection, without adding preparation latency. Source cost remains
independent of q. The source and distillation sections include perfect, historical mixed and
both damped inputs.
The default q grid is `.1 .2 .3 .4 .45 .475 .49 .5`. Both the Bell and joint
winning protocols are profiled; Bell points update the Bell incumbent too.
`comparisons.csv` separates these matched source gains from reoptimized gains.

`--max-dists` bounds recurrence per vertex; `--max-total-dists` also bounds
the total across the tree. The hardware defaults are two for each. All
distillation cases include a swaps-only search and terminal-recurrence
anchors; count-specific curves are best sampled, not certified optima.
Both searches optimize the tree and uniform acceptance. Gate infidelity,
operation duration and classical delays remain omitted. State-dependent
distillation success, hardware swap failures and all retries are included.
Extra distillation hardware success can be set with `--p-distillation`.

For a larger follow-up, use a **new directory** with `--hardware-regimes C D`,
`--memory-scales .125 .25 .5 1 2 4` and `--seeds 7 42`. For asymmetry,
`--asymmetry-geometries bottleneck left_comb` optionally combines the defects
or uses a homogeneous comb. On a comb every height contains only one swap,
so level and node policy classes coincide by construction; the heterogeneous
balanced cases are the meaningful compression test.

The hardware preset starts at horizon 16384 and allows 1048576, with the same
strict coverage and two-successive-horizon metric checks. Increase
`--max-horizon` in a fresh output directory if `unresolved.csv` shows missing
coverage/confirmation. Old cached failures are not silently retried.

At the batch root read `summary.txt`, `summary.csv`, `comparisons.csv`,
`distillation_counts.csv`, `q_profiles.csv`, and `unresolved.csv` alongside
the numbered PNG/PDF figures. Tables retain SKR/wait in both slots and seconds,
winning operation sequences, policies, q, recurrence count and convergence.
Plots keep different hardware/model/source conditions separate; shaded seed
ranges describe optimizer variability, not confidence intervals. Zero-key
schedules are blank and zero-baseline ratios are omitted; absolute-rate plots
retain those cases. Regenerate without evaluations with:

```bash
python -m src.scripts.gp_pauliAD --summarize-only \
  --output results/pauliAD/hardware-followup
```

An executed 18-job, two-link software check covering all four sections and
both relaxation models is saved under `results/pauliAD/hardware-smoke-20261028/`.
It uses one GP proposal, so it is not a scientific optimization result.
The report's full modelling rationale is in
[`pauliAD-overnight-report.md`](../analysis/pauliAD-overnight-report.md).

### Original placement study

`src.optimization.optimize_postselection` fixes the source and swap tree,
without distillation, and compares nested strategy classes:

| Class | Acceptance choice |
|---|---|
| `uniform` | The same policy at every swap: all corrected, Psi pair, or Phi pair |
| `level` | One independent policy per output-subtree height |
| `node` | One independent policy per physical swap location |

Elementary links have height zero; the first swapping level has height one.
For unbalanced trees this is subtree height, rather than distance from the root.
Every class includes the uniform schedules. Node searches also receive the
best level schedule. The GP anchors include final-only Psi selection, motivated
by the fifth notebook experiment.

```bash
python -m src.optimization.optimize_postselection \
  --nodes 5 --tree balanced --q 0.5 --t1-slots 74 \
  --output results/pauliAD/placement-N5-T74
```

This is an especially useful sanity check: selecting Psi only at the root
improves SKR by approximately 45% over accepting all outcomes. The script
enumerates all 27 per-swap schedules here, rather than assuming that example
is optimal.

Defaults: exhaustive uniform search, exhaustive larger spaces up to 243
schedules, GP otherwise. A balanced eight-link tree has 27 level schedules
and 2187 per-swap schedules. Use `--method exhaustive` or `--max-exact 2187`
to certify the latter by enumeration; use `--method gp` to compare GP search
seeds even in small spaces. The reported certificate applies only when every
candidate converges, within the stated numerical tolerances.

`--tree left` and `--tree right` select combs. `--protocol s0 s2 s1` supplies
a fixed tree explicitly. Policies are stored in **physical boundary order**:
`policies[i]` is the acceptance choice at `s{i}`, regardless of execution
order. For the balanced four-link tree, `s1` is the root. `summary.json` maps
these boundaries to physical endpoint spans, heights and depths.

The key questions are: how large is `node_vs_uniform`, does `level` capture
most of it, and does nonuniform selection enable key where uniform selection
does not? Compare absolute gains as well as percentages. A small GP gain is
not an upper bound on untested strategies' gains. Homogeneous and heterogeneous
hardware should be assessed separately.

## 2. Which source and protocol should we use?

`src.optimization.gp_pauliAD` builds on `gp_asymmetric`'s existing
`gamma, K, eta, tau` encoding for tree shape and distillation placement.
It adds a global source parameter and uniform swap acceptance:

\[
|\psi_q\rangle=\sqrt{1-q}|00\rangle+\sqrt q|11\rangle,
\qquad q\in[q_{\min},1/2].
\]

All elementary sources have the same q; q=1/2 is the Bell source. With
`--visibility v`, the source instead is
`v |psi_q><psi_q| + (1-v) I/4`. This is not a comparison at equal initial
fidelity. Preparation time/success probability are assumed independent of q.

```bash
python -m src.optimization.gp_pauliAD \
  --nodes 5 --t1-slots 100 --max-dists 1 \
  --gp-calls 100 --seed 42 --output results/pauliAD/protocol-N5-T100
```

Four matched searches identify what produces the gain:

| Search | Source | Acceptance | Tree and distillation |
|---|---|---|---|
| `bell_all` | q=1/2 | all corrected | optimized |
| `bell_policy` | q=1/2 | optimized uniform policy | optimized |
| `q_all` | optimized q | all corrected | optimized |
| `joint` | optimized q | optimized uniform policy | optimized |

Each gets `--gp-calls` proposals **plus explicit anchors**. Joint search includes
the compatible ablation winners, so it cannot lose to its evaluated controls.
The original joint winner is also evaluated with the **same tree** and (i)
Bell sources, (ii) all outcomes, (iii) both, and (iv) distillation removed.
These up-to-four additional evaluations isolate the source, branch and
distillation effects from a better tree found by the GP. They appear in
`matched_controls` in each job's summary and in batch `matched_controls.csv`.
An improving control is promoted to the reported joint winner.
`--joint-only` skips the ablations for a cheaper exploratory run. Use
`--max-dists 0` to isolate source/acceptance/tree effects first. The distillation
bound is per tree vertex, including leaves and the final link; hardware success
is controlled by `--p-distillation`. The recurrence is the backend's existing
bilateral-CNOT recurrence, without Werner twirling or new filtering operations.

The original encoding samples placements through NumPy's global RNG. Here a
stable hash of the encoding and replicate seed gives a local sampling seed;
the caller's RNG state is restored. Repeated encodings within a replicate are
deterministic, and the saved actual operation sequence fully specifies the
evaluated protocol. Multiple seeds explore different placement realizations.
The inherited tree enumeration is limited to at most 11 nodes in this script.
GP results are **best found**, not globally optimal certificates.

Initially compare `joint` against `bell_policy`, not only against fixed q=1/4.
Plot winning q against T1 and network size; identify whether gains come from
source imbalance, branch selection, or distillation. Check whether the winner
sits at the q search boundary before extending `--q-min` below its default 0.1.
The search does not assume that the best q varies monotonically with noise.

## Physics, convergence, and failures

- `--nodes N` means N-1 elementary links. Default generation is 0.2, swap and
  distillation hardware success are 1. Operations are instantaneous. Noise acts
  on both endpoints of a waiting link at every level. Rejection regenerates
  the consumed resources. There is no additional fixed preparation damping.
- Lifetimes are in slots. Default T1=100 and T2=2*T1 (pure AD).
  `--t2-slots` adds pure dephasing, with
  `kappa=1/T1`, `Gamma_X=Gamma_Y=1/T2-1/(2*T1)`, `Gamma_Z=0`.
  Infinite lifetimes are written `inf`. The runner's `--t2-ratio 1` provides
  a useful mixed relaxation/dephasing control.
- Individual scripts accept one `--p-gen` per link and one `--t1-slots` or
  `--t2-slots` per node. Alternatively provide `--pauli-rates GX GY GZ`
  and `--ad-rates kappa`; triples/scalars broadcast, or give a flattened
  triple and an AD rate per node. These are Pauli **mode-decay rates**;
  the backend applies Pauli noise after AD. Raw rates and lifetimes cannot
  be combined. Noncommuting general Pauli/AD layers retain that fixed order.
- No cutoffs by default. `--memory-cutoff-slots` fixes one shared cutoff;
  neither study optimizes it. The batch runner deliberately uses no cutoffs.
- The objective is pooled asymptotic X/Z BB84 SKR, `SKF / mean waiting time`.
  Postselection probability is already included in the accepted-delivery
  waiting distribution. It is not multiplied in a second time. No tail
  extrapolation is used. Waiting-time normalization and the residual-tail
  convention match the repository's utility functions.
- Each candidate starts at horizon 256, doubles up to 262144, and must have
  coverage >=0.999999 at **two successive horizons**, with SKR, SKF, waiting
  time, eX and eZ agreeing to `rtol=1e-5, atol=1e-9`. All are adjustable in
  individual scripts. Failures and insufficient horizons remain explicit
  missing results; their diagnostics are saved and they cannot win.
  An unresolved result now identifies insufficient coverage, a missing second
  covered horizon, or unstable metrics. Supplied states retain the same strict
  physicality checks. Selected swap probabilities are evaluated using sums of
  nonnegative population products to avoid cancellation in rare Bell branches.
  FFT PSD roundoff is repaired before conditioning, only within its existing
  probability-weighted machine-precision error budget; PMFs are preserved and
  larger violations still trigger the causal fallback. Invariant conditional
  states use a single scalar renewal, preserving their rare bins exactly.
  Only subnormal bins (below the
  smallest normal float, about 2.23e-308) retain a physical placeholder state
  instead of conditioning correlators that have lost relative precision; their
  PMF is preserved. This affects at most horizon times 2.23e-308 total mass.
  The original overnight batch used a 65536 cap. Use a new output directory
  with these defaults; reproducing its old resume plan requires explicitly
  passing `--max-horizon 65536` and retaining the original code/environment.
- To avoid a flat objective where all samples give zero key, the GP ranks
  zero-key states by their unclipped BB84 margin using a small positive
  penalty. Positive-key candidates always have negative objective (-SKR),
  and invalid candidates have penalty 1. Headline results always report
  actual SKR, not this feasibility guide. Refine marginal positive-key
  transitions; a sensible exploratory SKF threshold is 1e-9.

## What to read after the run

At the **batch root**, start with `summary.txt`, then `comparisons.csv` and
`summary.csv`. `postselection.png/.pdf` compare strategy classes and absolute
nonuniform gains; `protocol.png/.pdf` show rates and winning q (where SKF is
above 1e-9). Lines are seed
medians with seed ranges shaded, not confidence intervals. `manifest.json`
records job exit status/timing and `plan.json` records exact commands.

Inside **each job**:

| File | Purpose |
|---|---|
| `config.json`, `environment.json` | Physics, convergence settings, seed, packages and source hashes |
| `trials.jsonl` | Append-and-fsync audit trail, proposal encoding, actual protocol, policies, metrics and convergence history |
| `trials.csv` | Flat readable trial table, refreshed every ten requests and at exit |
| `summary.txt`, `summary.json` | Class/ablation winners, comparison gains, and certification status |
| `best.json`, `best.npz` | Best candidate and its completion PMF plus six-coordinate conditional states |
| `search.png` | Running best SKR, q versus SKR, and SKF versus waiting time |
| `best_protocol.png` | Winning tree with distillation counters and acceptance labels |
| `best_delivery.png` | Completion-time PMF and Phi+ fidelity distribution |
| `run.log`, `console.log` | Live trial/backend diagnostics; console log is batch-only |

Physical candidates are cached across stages. Cached requests remain in the
trace with `cached=true`; distinguish proposal count from distinct backend
evaluations. Restart tolerates an incomplete final JSONL record. A per-job
filesystem lock prevents concurrent writers. A stopped job still has trial
records; a hard kill may leave plots/CSV behind the journal until resume.

The recorded baseline example is under `results/pauliAD/working-example-20261026/`; the
four-link distillation smoke is under
`results/pauliAD/protocol-distillation-example/`. These are small checks, not
completed overnight studies.

Focused scientific and restart checks:

```bash
pytest -q src/tests/test_pauli_ad_optimization.py
```
