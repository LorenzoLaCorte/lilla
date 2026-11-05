# Frozen evidence for twirled-model optimization

Snapshot: **2 November 2026, 08:50:56 CET / 07:50:56 UTC**.
Main interpretation: [twirled-model-optimization.md](../twirled-model-optimization.md).

`evidence.json` contains all extracted comparisons and fixed-policy
diagnostics from all **192 completed jobs**, numerical-status counts, seven
representative protocols/comparisons and their convergence histories, and
**1,348 input checksums**, including all per-model journals. All 48,747
converged evaluations passed the stored convergence/scalar-arithmetic audit.
There are also 14,845 failed and 14,168 unconverged evaluations. These remain
missing, never zero. The raw experiment inputs and outputs were not modified.

The four PNG/PDF figure pairs use this frozen evidence. Rates in the bar
charts are bits/second, converted with the saved 35-microsecond slot.
Both complete and partial comparisons are explicitly labelled; there is
no fully resolved joint-space comparison in this snapshot.

The eight-link ranking-reversal chart contains four resolved scores per
noise regime. The 100% loss is a strict, post hoc pairwise witness and a
common-subset result, not a certified optimum over the full joint space.
The source-fidelity-matched Werner prediction is positive for its selected
protocol, so this witness is not an all-zero surrogate tie.

The prior 158-job report, PDF, figures and evidence are preserved in
[the interim archive](../twirled-model-optimization-20261101/README.md).

Replot from the same evidence without reading the live batch or evaluating
protocols:

```bash
source venv/bin/activate
python -m src.analysis.twirled_model_report --replot
```

For a later extraction, select a **new output directory** with `--output`
and revise the associated report. A new extraction does not update the
manually written prose and tables automatically.
