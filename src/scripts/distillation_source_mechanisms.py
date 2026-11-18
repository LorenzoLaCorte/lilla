"""Short state-level diagnostics for harmful raw recurrence and PE sources.

Run: python -m src.scripts.distillation_source_mechanisms

Each ideal round consumes two independent identical copies of the previous
conditioned ensemble, accepts both 00/11 outcomes, and averages those branches.
This diagnostic has no storage noise or timing; use the comparison runner's
pure_pe preset to include preparation, memory noise, cutoffs and retries.
"""

from argparse import ArgumentParser
import csv
import json
import os
from pathlib import Path

import numpy as np

from src.core.pauli_adc import (
    DISTILLATION_STRATEGIES, bell_weights_to_coordinates, coordinates_to_matrix,
    phi_plus_fidelity, recurrence_distillation,
)
from src.scripts.compare_distillation_strategies import LABELS, ROOT, source_coordinates


def state_metrics(coordinates):
    """Dense diagnostics, including entanglement negativity independently of F."""
    rho = np.asarray(coordinates_to_matrix(coordinates), dtype=complex)
    partial_transpose = rho.reshape(2, 2, 2, 2).transpose(0, 3, 2, 1).reshape(4, 4)
    eigenvalues = np.linalg.eigvalsh(rho)
    pt_eigenvalues = np.linalg.eigvalsh(partial_transpose)
    return dict(
        fidelity=phi_plus_fidelity(coordinates),
        negativity=float(-np.minimum(pt_eigenvalues, 0.).sum()),
        purity=float(np.trace(rho @ rho).real),
        minimum_eigenvalue=float(eigenvalues[0]),
        iz=float(coordinates[1]), zi=float(coordinates[2]),
        zz=float(coordinates[3]), xx=float(coordinates[4]), yy=float(coordinates[5]),
        population_00=float(rho[0, 0].real), population_01=float(rho[1, 1].real),
        population_10=float(rho[2, 2].real), population_11=float(rho[3, 3].real),
        coherence_00_11=float(rho[0, 3].real),
    )


def diagnostic_sources():
    cases = [dict(source_id="dephased_bell_f095", family="dephased_bell", q=None,
                  visibility=None, coordinates=bell_weights_to_coordinates([.95, 0, 0, .05])),
             dict(source_id="werner_f095", family="werner", q=None, visibility=14/15,
                  coordinates=source_coordinates("werner", 14/15, None))]
    for q in (.1, .2, .25, .35, .5):
        cases.append(dict(source_id=f"pure_pe_q{q:g}", family="pure_pe", q=q,
                          visibility=1., coordinates=source_coordinates("biased_x", 1., q)))
    return cases


def run_diagnostics(rounds=5):
    rows = []
    for source in diagnostic_sources():
        for strategy in DISTILLATION_STRATEGIES:
            coordinates = np.array(source["coordinates"], dtype=float)
            for round_number in range(rounds+1):
                probability = None
                if round_number:
                    coordinates, probability = recurrence_distillation(
                        coordinates, coordinates, strategy=strategy, conditioned=True,
                    )
                rows.append(dict(source_id=source["source_id"], family=source["family"],
                                 q=source["q"], visibility=source["visibility"],
                                 strategy=strategy, round=round_number,
                                 acceptance_probability=probability,
                                 **state_metrics(coordinates)))
    return rows


def plot_diagnostics(output, rows):
    os.environ.setdefault("XDG_CACHE_HOME", str(output / ".cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".cache" / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import seaborn as sns
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False})
    palette = sns.color_palette("colorblind")
    fig, axes = plt.subplots(2, 2, figsize=(180/25.4, 105/25.4), sharex=True)
    for column, (source_id, label) in enumerate((
        ("dephased_bell_f095", r"Dephased Bell mixture ($F_0=0.95$)"),
        ("pure_pe_q0.25", r"Pure PE source ($q=1/4$)"),
    )):
        for strategy, color, marker, style in zip(
            DISTILLATION_STRATEGIES, palette, ("o", "s", "^", "D"), ("-", "--", ":", "-.")
        ):
            selected = [r for r in rows if r["source_id"] == source_id and r["strategy"] == strategy]
            for row_number, metric in enumerate(("fidelity", "negativity")):
                axes[row_number, column].plot(
                    [r["round"] for r in selected], [r[metric] for r in selected],
                    label=LABELS[strategy], color=color, marker=marker, ls=style,
                    ms=4 if strategy != "none" else 2.5,
                    markerfacecolor="none" if strategy == "bell_diagonal" else color,
                    lw=1.,
                )
        axes[0, column].text(.03, .04, label, transform=axes[0, column].transAxes)
        axes[0, column].set_ylim(.48, 1.02)
        axes[1, column].set_ylim(-.02, .52)
        axes[1, column].set_xlabel("Recurrence rounds")
        axes[1, column].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[0, 0].set_ylabel(r"$\Phi^+$ fidelity")
    axes[1, 0].set_ylabel("Entanglement negativity")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .93))
    for suffix in ("pdf", "png"):
        fig.savefig(output / ("source_mechanisms."+suffix), dpi=220)
    plt.close(fig)


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results" / "distillation_strategies" / "mechanisms")
    args = parser.parse_args(argv)
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = run_diagnostics(args.rounds)
    (output / "source_mechanisms.json").write_text(json.dumps(rows, indent=2, allow_nan=False)+"\n")
    with (output / "source_mechanisms.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_diagnostics(output, rows)
    report = """# Distillation source mechanisms

The slides' `Twirling before recurrence` example uses the Bell-diagonal mixture
0.95 Phi+ + 0.05 Phi-, not a pure partially entangled (PE) state. Raw CNOT
recurrence gives fidelity 0.905, versus approximately 0.96496 after an input
Werner twirl. The BBPSSW output twirl preserves that conditional fidelity.
A Bell-diagonal twirl alone leaves this already Bell-diagonal input unchanged.

For a pure source sqrt(1-q)|00> + sqrt(q)|11>, let C=2 sqrt(q(1-q)). Raw
recurrence accepts with probability one but squares C, keeping the same
00/11 populations. Fidelity changes from (1+C)/2 to (1+C^2)/2 and negativity
from C/2 to C^2/2. Both decrease for nonmaximal entangled inputs. At q=1/4,
fidelity goes from 0.9330127 to 0.875, and negativity from 0.4330127 to 0.375.
Repeated identical-copy rounds give C_r=C_0^(2^r), approaching a separable
diagonal mixture. q=1/2 is the fixed perfect-Bell control.

For pure PE inputs, auxiliary outcome 11 produces a perfect Bell pair with
probability 2q(1-q). Outcome 00 produces a less balanced pure state with
probability (1-q)^2+q^2. The current raw protocol accepts and averages both;
it therefore does not implement the concentration protocol accepting only 11.

Bell-diagonal twirling and raw recurrence have equal Bell populations and
negativity in these noise-free examples, despite different local biases for
PE sources. They can separate after non-unital storage. Werner twirling also
changes the correlation anisotropy; DEJMPS changes which parity is tested.

Every round consumes independent identical copies of the previous conditioned
ensemble. The plots show conditional state quality, not key-rate superiority:
acceptance probabilities are saved separately. Gate, storage, preparation and
retry timing are absent here; the comparison runner's pure_pe preset includes
those stochastic network effects under the existing timing model.
"""
    (output / "summary.md").write_text(report)
    print(f"Saved {len(rows)} state-level measurements and plots in {output}")


if __name__ == "__main__":
    main()
