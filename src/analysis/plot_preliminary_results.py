"""Generate a standalone three-panel research figure from saved scan artifacts.

Run after activating venv: MPLCONFIGDIR=/tmp/lilla-mpl python -m
src.analysis.plot_preliminary_results
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT/name).read_text())


def plot_policy_lines(ax, rows, *, four_links=False):
    colors = {"none": "#1764a0", "root": "#d8781a"}
    for dist in ("none", "root"):
        for model, style in (("physical", "-"), ("werner", "--")):
            chosen = [r for r in rows if r["candidate"]["distillation"] == dist
                      and r["candidate"]["cutoff"] is not None
                      and (not four_links or r["candidate"]["tree"] == [[0, 1], [2, 3]])]
            chosen.sort(key=lambda r: r["candidate"]["cutoff"])
            y = [r["metrics" if four_links else "predictions"][model]
                 ["rate_bits_per_slot"] for r in chosen]
            ax.plot([r["candidate"]["cutoff"] for r in chosen], y,
                    style, color=colors[dist], marker="o", ms=3,
                    lw=1.7, label=f"{dist} · {model}")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16] + ([] if four_links else [32]))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Shared cutoff (slots)")
    ax.set_ylabel("Pooled BB84 rate (bits/slot)")
    ax.grid(alpha=.18)


def main():
    two = next(r for r in read("distillation_family_results.json")["results"]
               if r["case"]["family"] == "phase_error"
               and r["case"]["fidelity"] == .95
               and r["case"]["t1_slots"] == 30)
    four = read("four_link_phase_32768.json")["results"][0]
    threshold = read("key_threshold_results.json")

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), constrained_layout=True)
    plot_policy_lines(axes[0], two["rows"])
    axes[0].set_title("a  Two links: phase-biased input")
    axes[0].plot([1], [.027110963143756276], "*", ms=12, color="#1764a0")
    axes[0].plot([4], [.011029148011783839], "*", ms=12, color="#d8781a")
    axes[0].legend(fontsize=7, frameon=False, loc="upper right")

    points = sorted(threshold["probes"], key=lambda p: p["t1_slots"])
    colors = {"full_real_x": "#1764a0", "local_channel_twirl": "#d8781a"}
    for model, label in (("full_real_x", "Physical T1/T2"),
                         ("local_channel_twirl", "Pauli-twirled channel")):
        xs, ys = [], []
        for point in points:
            rows = [r for r in point["rows"] if r["model"] == model]
            xs.append(point["t1_slots"])
            ys.append(max(r["rate_bits_per_slot"] for r in rows))
        axes[1].plot(xs, ys, "-o", color=colors[model], ms=3, lw=1.7,
                     label=label)
        axes[1].axvline(threshold["cutoff_one_thresholds"][model],
                        color=colors[model], alpha=.5, ls=":")
    axes[1].set_xlabel(r"Memory lifetime $T_1$ (slots)")
    axes[1].set_ylabel("Best rate over tested cutoffs (bits/slot)")
    axes[1].set_title("b  Three links: key threshold")
    axes[1].legend(fontsize=7, frameon=False, loc="upper left")
    axes[1].grid(alpha=.18)

    plot_policy_lines(axes[2], four["rows"], four_links=True)
    axes[2].set_title("c  Four links: balanced tree")
    axes[2].plot([1], [.0018800655527829954], "*", ms=12, color="#1764a0")
    axes[2].plot([4], [.0008614642720735014], "*", ms=12, color="#d8781a")
    axes[2].legend(fontsize=7, frameon=False, loc="upper right")
    for extension in ("pdf", "png"):
        fig.savefig(ROOT/f"preliminary_results.{extension}", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
