# Hardware-informed overnight evidence

Input: `results/pauliAD/hardware-followup/`, completed 29 October 2026.
The input directory is unchanged. Every converged candidate has zero SKF/SKR;
signed BB84 margins are diagnostics, not positive-key fractions.

| File | Contents |
|---|---|
| `audit.json` | Job, request, candidate, profile and timing totals |
| `inputs.sha256.json` | Hashes of 722 input files |
| `evaluations.csv` | One candidate row per job, including unresolved diagnostics |
| `winners.csv` | All 336 reported stage winners and certification flags |
| `q_profiles.csv` | 804 stored matched-profile rows, including unresolved points |
| `matched-recurrence.csv` | 144 matched balanced/all-corrected terminal-recurrence controls |
| `distillation-by-count-scale2.csv` | Best sampled margin by source, model, size and count at s=2 |
| `asymmetry-gaps.csv` | Observed node/level/uniform margin differences |
| `source-margins.csv` | Bell-versus-joint source-search margin comparisons |
| `convergence.csv` | Converged/unresolved candidate counts by study and size |
| Numbered PNG/PDF files | Six figures embedded in the report |

Candidate IDs deduplicate requests **within each job**. Physical experiments
may repeat across jobs. Invalid or unresolved diagnostics do not establish
converged rates, even if their truncated `skr` field is zero.

Regenerate data/figures from the repository root, without backend evaluations:

```bash
source venv/bin/activate
python -m src.analysis.plot_pauli_ad_hardware_report
```

The script independently checks every converged record's two-horizon coverage,
metric stability, signed margin and rate formula. To build the main PDF:

```bash
cd src/analysis
pandoc pauliAD-overnight-report.md --pdf-engine=xelatex \
  -V geometry:margin=1.7cm -V fontsize=10pt \
  -o pauliAD-overnight-report.pdf
```

The first ideal-hardware report is archived separately as
`src/analysis/pauliAD-ideal-hardware-report.md` and `.pdf`. Its original
figures and evidence remain in `src/analysis/pauliAD-overnight-report/`.
