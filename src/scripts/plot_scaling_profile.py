"""Plot runtime composition at the largest profiled chain for each p_swap.

    python -m src.scripts.plot_scaling_profile

Reads profiling.json only; no simulations or profiles are rerun. Writes a
single-column runtime_breakdown.pdf/.png and the six plotted rows as CSV.
Panels show measured seconds on separate linear axes, without normalization.
Scalar retry setup and its state-dependent application are shown separately.
"""

from argparse import ArgumentParser
import csv
import json
import math
from pathlib import Path


SETUP_PHASES = {"shared_retry_resolvent"}
APPLICATION_PHASES = {"batched_renewal_application"}
STATE_PHASES = {"state_join", "waiting_noise_aggregation", "quantum_map"}
GROUP_LABELS = ("Retry setup", "Apply retries", "State/noise maps", "Other")
GROUP_COLUMNS = ("retry_setup_seconds", "retry_application_seconds", "state_seconds", "other_seconds")


def summarize(document):
    """Group disjoint measured seconds, with unassigned work counted as other."""
    largest = {}
    for case in document["records"]:
        p_swap = case["p_swap"]
        if p_swap not in largest or case["nesting_level"] > largest[p_swap]["nesting_level"]:
            largest[p_swap] = case
    if not largest:
        raise ValueError("No completed profiles to plot")
    rows = []
    for p_swap, case in sorted(largest.items()):
        for backend in ("Werner", "X-state"):
            data = case["backends"][backend]
            phases = {name: values["seconds"] for name, values in data["phases"].items()}
            if "legacy_scalar_renewals" in phases:
                raise ValueError("Runtime breakdown requires profiles of the shared-retry implementation")
            if any(not math.isfinite(value) or value < 0 for value in phases.values()):
                raise ValueError(f"Invalid phase timings for {p_swap}, {backend}")
            total = sum(phases.values())
            if total <= 0 or not math.isclose(total, data["profiled_seconds"], rel_tol=1e-9):
                raise ValueError(f"Phase timings do not sum to the profiled total for {p_swap}, {backend}")
            setup = sum(value for name, value in phases.items() if name in SETUP_PHASES)
            application = sum(value for name, value in phases.items() if name in APPLICATION_PHASES)
            state = sum(value for name, value in phases.items() if name in STATE_PHASES)
            other = sum(value for name, value in phases.items()
                        if name not in SETUP_PHASES | APPLICATION_PHASES | STATE_PHASES)
            rows.append({
                "p_swap": p_swap, "nodes": case["nodes"], "horizon": case["horizon"],
                "backend": backend, "profiled_seconds": total,
                "reference_median_seconds": data["reference_median_seconds"],
                "retry_setup_seconds": setup, "retry_application_seconds": application,
                "state_seconds": state, "other_seconds": other,
            })
    return rows


def render(rows, folder):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt, cycler
    import seaborn as sns

    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "runtime_breakdown.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    palette = sns.color_palette("colorblind")
    with plt.rc_context({
        "axes.prop_cycle": cycler(color=palette),
        "font.family": "sans-serif", "font.size": 7,
        "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "legend.fontsize": 7, "axes.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": None,
    }):
        panel_count = len(rows) // 2
        fig, axes = plt.subplots(panel_count, 1, squeeze=False,
                                 figsize=(88 / 25.4, (40 * panel_count + 10) / 25.4),
                                 constrained_layout=True)
        colors = (palette[0], palette[2], palette[1], "0.8")
        for panel, ax in enumerate(axes[:, 0]):
            pair = rows[2 * panel:2 * panel + 2]
            maximum = max(row["profiled_seconds"] for row in pair)
            bottoms = [0.0, 0.0]
            for column, label, color in zip(GROUP_COLUMNS, GROUP_LABELS, colors):
                heights = [row[column] for row in pair]
                ax.bar([0, 1], heights, bottom=bottoms, width=0.55, color=color, label=label)
                for x, bottom, height in zip([0, 1], bottoms, heights):
                    if height >= 0.1 * maximum:
                        ax.text(x, bottom + height / 2, f"{height:.3g}", ha="center", va="center",
                                fontsize=6, color="white" if column == "retry_setup_seconds" else "black")
                bottoms = [bottom + height for bottom, height in zip(bottoms, heights)]
            ax.set_xticks([0, 1], [row["backend"] for row in pair])
            ax.set_xlim(-0.6, 1.6)
            ax.set_ylim(0, 1.08 * maximum)
            ax.set_ylabel("Time (s)")
            row = pair[0]
            ax.set_xlabel(rf"$p_{{\mathrm{{swap}}}}={row['p_swap']:g},\quad N={row['nodes']}$")
        axes[0, 0].legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2,
                          frameon=False, borderaxespad=0, columnspacing=1.0)
        for suffix in ("pdf", "png"):
            fig.savefig(folder / f"runtime_breakdown.{suffix}", dpi=300)
        plt.close(fig)


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path(__file__).resolve().parents[2] / "results/benchmarks/chain_scaling/profiling.json")
    parser.add_argument("--output-dir", type=Path, help="Default: the input file's directory.")
    args = parser.parse_args(argv)
    rows = summarize(json.loads(args.input.read_text()))
    folder = args.output_dir or args.input.parent
    render(rows, folder)
    print(f"Saved runtime_breakdown.pdf, .png, and .csv in {folder}; no simulations run.")


if __name__ == "__main__":
    main()
