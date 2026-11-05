# Frozen evidence for twirled-model optimization

Snapshot: **1 November 2026, 13:54:48 CET / 12:54:48 UTC**.
Main interpretation: [archived report](../report.md).

`evidence.json` contains all extracted comparisons and fixed-policy
diagnostics from 158 completed jobs, numerical-status counts, representative
protocols and their convergence histories, and 170 input checksums. Active
jobs are omitted. Live batch manifests may have advanced since extraction.
The inputs and the running experiment were not modified.

The two PNG/PDF figure pairs use this frozen evidence. Rates in the bar
chart are bits/second, converted with the saved 35-microsecond slot.
Both complete and partial comparisons are explicitly labelled; there is
no fully resolved joint-space comparison in this snapshot.

Replot from the same evidence without reading the live batch or evaluating
protocols:

```bash
source venv/bin/activate
python -m src.analysis.twirled_model_report --replot
```

For a later extraction, select a **new output directory** with `--output`
and revise the associated report. A new extraction does not update the
manually written prose and tables automatically.
