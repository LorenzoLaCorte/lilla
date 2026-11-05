# Metropolitan optimization evidence — 30 October 2026

Input: `results/pauliAD/metropolitan-followup/`, 288 completed jobs.
The input directory is unchanged. Interpretation is in the
[main report](../pauliAD-overnight-report.md).

| File | Contents |
|---|---|
| `audit.json` | Request, candidate, convergence, profile and timing totals |
| `inputs.sha256.json` | Hashes of 1,442 inputs, including all 288 journals |
| `evaluations.csv` | All 26,736 distinct candidate rows summed within jobs |
| `winners.csv` | 672 stage winners, with certification, errors and protocols |
| `q_profiles.csv` | 1,581 fixed-protocol q-profile rows, including unresolved points |
| `matched-controls.csv` | Stored Bell/all-outcome/no-distillation controls |
| `jobs.csv` | Job parameters, times and subprocess elapsed times |
| `placement.csv` | Complete uniform/level/node placement winners |
| `distillation-by-count.csv` | Best sampled rate by total recurrence count |
| `matched-terminal-recurrence.csv` | Matched balanced/all-corrected terminal-recurrence controls |
| `asymmetry-gaps.csv` | Uniform/level/node rates and observed differences |
| `source-gains.csv` | Reoptimized joint-q versus internal Bell-search comparisons |
| `matched-source-gains.csv` | Matched q-versus-Bell comparisons on fixed protocols |
| `convergence.csv` | Candidate convergence by study and size |
| Numbered PNG/PDF files | Main and supplementary report figures |

`skr_bits_per_second` uses the 35-microsecond slot; `skr` uses bits/slot.
`wait_seconds` differs from `mean_wait` in slots. `K` counts total recurrence
rounds, including leaves and internal vertices.

`benchmark_fidelity` labels the nominal Bell-reference regime. The actual
source uses `source_visibility`, `source_damping` and candidate `q`.
Perfect and source-AD controls repeat across nominal regimes. There are
17,304 distinct hardware/candidate descriptions across the 26,736 within-job
evaluations. Neither count implies independent physical observations or
independent optimizer replicates.

Only `status=converged` rows establish reported rates. Unresolved rows keep
their truncated diagnostics but cannot win. Ratios against zero baselines
are undefined. Zero-key q choices can reflect the margin tie breaker and
must not be interpreted as useful source optima.

Regenerate without backend evaluation or optimization:

```bash
source venv/bin/activate
python -m src.analysis.plot_pauli_ad_metropolitan_report
```

The generator checks each converged record's final two coverage values,
metric stability, entropy margin and SKR formula. To build the main PDF:

```bash
cd src/analysis
pandoc pauliAD-overnight-report.md --pdf-engine=xelatex \
  -V geometry:margin=1.7cm -V fontsize=10pt \
  -o pauliAD-overnight-report.pdf
```

Historical reports are `pauliAD-ideal-hardware-report.md/.pdf` and
`pauliAD-long-distance-report.md/.pdf`, with separate preserved evidence.
The metropolitan reference note records the pilot and calibration, rather
than the completed optimization.
