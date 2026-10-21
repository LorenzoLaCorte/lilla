# Outcome postselection: two small experiments

From the repository root:

```bash
source venv/bin/activate
python -m src.analysis.postselection
python -m src.analysis.postselection_example
```

Each script writes a new directory under `results/analysis/` containing
`postselection.png`, `postselection.pdf`, `sweep.csv`, and `config.json`.
The CSV includes rate, unclipped BB84 score, waiting time, eX/eZ and coverage.
Every point is checked at T and 2T. Saved configurations can be imported into
the Streamlit app. Original run bundles and older study outputs remain intact.

## Four links with waiting-time relaxation

Parameters for `postselection.py` are directly in `main()`:
four balanced links, Werner F0=0.97, p_gen=0.2, p_swap=1, no distillation or
cutoff, full real-X, and T2=2*T1. It varies T1 from 5 to 4000 slots and compares
the three global outcome policies. The saved-noise study, saved-run audit,
command-line options, automatic boundary refinement and large reports were
removed. T=1024 and 2048 agree for this example.

The earlier refined positive-key thresholds for this same pure-relaxation
slice were approximately 236 slots for all outcomes, 357 for Phi/ Psi pairs.
The simplified plot reproduces the narrower positive-key region under global
rejection. This finding is specific to this initial-state/protocol family.

Filtering benefit means an improvement in the **BB84 score**, not a reduction
in waiting time. Under strong damping, Phi selection can give a less negative
score, but it remains below zero. Its shorter waiting time than Psi selection
comes from higher accepted-outcome probability. These are distinct effects.

Phi and Psi are not identical. At a fixed swap, if
`b = zR(left)*zL(right)`, their probabilities are `(1+b)/2` and `(1-b)/2`, and
their conditioned XX correlators are `XX(left)*XX(right)/(1+b)` and
`XX(left)*XX(right)/(1-b)`. Near the useful-key region in the homogeneous
study, products of the damping-generated biases are small, so differences are
small compared with their shared retry penalty. Under strong damping the
score/waiting curves separate, while clipping both rates to zero conceals this
in the SKR panel. Local biases can also differ even when BB84 rates are close.

## A case where filtering gives positive key

`postselection_example.py` uses two prescribed input states,
each obtained by applying equal local amplitude damping to a perfect Phi+
pair, with survival eta. Their real-X coordinates are

```text
(1, 1-eta, 1-eta, eta²+(1-eta)², eta, -eta).
```

Both pairs are already relaxed before swapping. Generation and swap hardware
are deterministic, with no additional waiting noise, distillation or cutoff.
The backend still accounts for rejected outcomes and full-reset retries.
This differs from the four-link experiment's freshly generated Werner pairs.
The reported rates exclude the prior input-preparation time; including a fixed
finite preparation cost would reduce the rate without changing positivity.

At eta=0.9145:

| Accepted outcomes | Unclipped BB84 score | SKR (bits/model slot) | Mean (slots) |
|---|---:|---:|---:|
| All corrected | -0.00365726 | 0 | 1.000000 |
| Phi pair | -0.01145921 | 0 | 1.985486 |
| Psi pair | +0.00445348 | 0.00221046 | 2.014728 |

Psi improves eX from 0.08184488 to 0.07876555, while slightly worsening eZ
from 0.14415223 to 0.14521377. The entropy benefit outweighs the penalty:
the pooled score crosses zero despite a longer waiting time. Phi does the
opposite transverse filtering and remains below threshold.

The existing low-level maps give onset eta values of approximately
0.91383123 (Psi), 0.91503786 (all), and 0.91615407 (Phi). Thus a small interval
exists where only Psi yields positive key. Since eta=exp(-tau/T1), for a fixed
prior storage duration tau this corresponds to required T1 of about 11.10*tau
for Psi versus 11.26*tau for all outcomes. This is a controlled example of a
regime extension, not evidence of an extension in the original four-link slice.

The generated example `config.json` selects Psi at eta=0.9145 and can be loaded
into the app. The full curve covers eta=0.905 to 0.925. The plot displays
T1 in units of the fixed prior storage time tau, using T1/tau=-1/log(eta),
with T2=2*T1 (pure relaxation). The CSV retains the original eta values.

## Checks

```bash
source venv/bin/activate
python -m pytest -q src/tests/analysis src/tests/pauli_adc/test_numeric.py src/tests/pauli_adc/test_metrics.py src/tests/pauli_adc/test_protocol_metamorphic.py
python -m compileall -q src/analysis
git diff --check
```

23 tests passed. Controls cover the absence of immediate filtering benefit
with one fresh Werner input, noiseless Werner reduction, insufficient horizon,
and the positive Psi case against the direct swap map and geometric retry law.
