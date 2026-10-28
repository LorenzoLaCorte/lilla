"""Report figures from saved overnight trials only; never runs an evaluator.

Run: python -m src.analysis.plot_pauli_ad_overnight
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import NullFormatter, ScalarFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/pauliAD/overnight"
OUTPUT = Path(__file__).with_name("pauliAD-overnight-report")
BLUE, ORANGE, GRAY = "#0072B2", "#D55E00", "#999999"


def load(study, nodes, t1, q=None, seed=7):
    label = str(q) if study == "postselection" else "opt"
    path = RUN / f"{study}_N{nodes}_T1_{t1}_q_{label}_seed{seed}/summary.json"
    return json.loads(path.read_text())


def save(fig, name):
    fig.savefig(OUTPUT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False})
    evidence = {}
    # 1. Same-level policy schedules, and gains over each source's own baseline.
    examples = [(5, 74, .5), (5, 200, .25), (9, 600, .25)]
    labels = ["4 links, Bell, $T_1=74$", "4 links, PE, $T_1=200$", "8 links, PE, $T_1=600$"]
    schedules = np.array([[0, 1, 2], [1, 0, 2], [1, 0, 0]])
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.25), gridspec_kw={"width_ratios": [1.05, 1.3]},
                             constrained_layout=True)
    axes[0].imshow(schedules, cmap=ListedColormap([BLUE, ORANGE, "#eeeeee"]), vmin=0, vmax=2, aspect="auto")
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, ("all", r"$\Psi$", "—")[schedules[i, j]], ha="center", va="center",
                         color="white" if schedules[i, j] < 2 else "#777777")
    axes[0].set(xticks=range(3), xticklabels=["1 (first)", "2", "3"], yticks=range(3),
                yticklabels=labels, xlabel="Swapping level", title="Best level schedule (exhaustive)")
    for i, (nodes, t1, q) in enumerate(examples):
        s = load("postselection", nodes, t1, q)
        baseline, hybrid = [s["results"][k]["best"]["result"] for k in ("uniform", "level")]
        ratio = hybrid["skr"]/baseline["skr"]
        evidence[f"placement_{nodes}_{t1}_{q}"] = dict(uniform=baseline, hybrid=hybrid, rate_ratio=ratio)
        axes[1].barh(i+.16, 1., height=.28, color=GRAY, label="Best uniform" if i == 0 else None)
        axes[1].barh(i-.16, ratio, height=.28, color=ORANGE, label="Best level" if i == 0 else None)
        axes[1].text(ratio+.16, i-.16, f"{ratio:.2f}×", va="center", fontsize=9)
    axes[1].set(yticks=range(3), yticklabels=[], ylim=(2.5, -.5), xlim=(0, 14),
                xlabel="SKR / best uniform SKR", title="Retries included; fixed source and tree")
    axes[1].legend(loc="upper right", fontsize=8)
    save(fig, "1-placement")

    # 2. Exact matched Bell-source protocols present in the existing trial logs.
    matched = []
    for t1 in (74, 200, 600):
        p = RUN / f"protocol_N5_T1_{t1}_q_opt_seed7/trials.jsonl"
        trials = [json.loads(line) for line in p.read_text().splitlines()]
        for distillation in (False, True):
            protocol = ["s0", "s2", "s1"] + (["d3"] if distillation else [])
            found = [r for r in trials if r["candidate"]["protocol"] == protocol
                     and r["candidate"]["q"] == .5
                     and set(r["candidate"]["policies"]) == {"all_corrected"}
                     and r["result"]["status"] == "converged"]
            assert found, (t1, protocol)
            matched.append(dict(T1=t1, distillation=distillation, request_id=found[0]["request_id"],
                                **found[0]["result"]))
    pd.DataFrame(matched).to_csv(OUTPUT / "matched-distillation.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.2), constrained_layout=True)
    positions = np.arange(3)
    for distilled, offset, color, label in [(False, -.18, GRAY, "Swaps only"),
                                           (True, .18, BLUE, "One terminal recurrence")]:
        group = [r for r in matched if r["distillation"] == distilled]
        axes[0].bar(positions+offset, [r["skr"] for r in group], .34, color=color, label=label)
        axes[1].plot(positions, [r["skf"] for r in group], "o-", color=color, label=label)
    for i in range(3):
        a, b = matched[2*i:2*i+2]
        axes[0].text(i, max(a["skr"], b["skr"])+.003, f"{b['skr']/a['skr']:.2f}×",
                     ha="center", fontsize=8)
    for ax in axes:
        ax.set(xticks=positions, xticklabels=[74, 200, 600], xlabel="$T_1$ (slots)")
    axes[0].set(ylabel="SKR (bits/slot)", ylim=(0, .095), title="4 Bell links, identical balanced tree")
    axes[1].set(ylabel="SKF", ylim=(0, 1.08), title="Quality gain eventually stops paying for retries")
    axes[0].legend(fontsize=8, loc="upper left")
    save(fig, "2-terminal-distillation")
    evidence["matched_distillation"] = matched

    # 3. Level policies match every best per-node rate, with different guarantees.
    table = pd.read_csv(RUN / "summary.csv")
    placement = table[table.study == "postselection"]
    pairs = placement.pivot(index=["nodes", "t1", "q", "seed"], columns="stage", values="skr").reset_index()
    assert np.allclose(pairs.node, pairs.level, rtol=0, atol=1e-14)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.2), constrained_layout=True)
    for nodes, marker, color in [(5, "o", BLUE), (9, "x", ORANGE)]:
        sub = pairs[pairs.nodes == nodes]
        axes[0].scatter(sub.level, sub.node, label=f"{nodes-1} links", marker=marker,
                        color=color, s=35, alpha=.7)
    axes[0].plot([0, .08], [0, .08], "--", color=".5", linewidth=1)
    axes[0].set(xlabel="Best level SKR (bits/slot)", ylabel="Best per-swap SKR (bits/slot)")
    axes[0].legend(fontsize=8)
    counts = [(4, 3, 9, 27), (8, 3, 27, 2187)]
    for i, (label, color) in enumerate([("Uniform", GRAY), ("Per level", BLUE), ("Per swap", ORANGE)]):
        axes[1].bar(np.arange(2)+(i-1)*.23, [row[i+1] for row in counts], .22, label=label, color=color)
    axes[1].set(xticks=[0, 1], xticklabels=["4 links", "8 links"], yscale="log",
                ylabel="Number of possible schedules", title="Same best-found rate, smaller search")
    axes[1].legend(fontsize=8)
    save(fig, "3-level-compression")
    evidence["level_vs_node"] = dict(comparisons=len(pairs), largest_rate_difference=float(abs(pairs.node-pairs.level).max()),
                                     independent_hardware_source_points=len(pairs.drop(columns="seed").drop_duplicates()))

    # 4. Use matched q counterfactuals, instead of attributing a better tree to PE.
    q_rows = []
    for summary in sorted(RUN.glob("protocol_*/summary.json")):
        s = json.loads(summary.read_text())
        controls = s["matched_controls"]
        original = controls["original_joint_winner"]
        if original["result"]["skr"] <= 0:
            continue
        bell = controls["same_protocol_bell"]
        _, n, _, t1, _, _, seed = summary.parent.name.split("_")
        q_rows.append(dict(nodes=int(n[1:]), T1=float(t1), seed=int(seed[4:]), q=original["candidate"]["q"],
                           gain_percent=100*(original["result"]["skr"]/bell["result"]["skr"]-1)))
    pd.DataFrame(q_rows).to_csv(OUTPUT / "matched-q.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.2), constrained_layout=True)
    for nodes, marker, color in [(5, "o", BLUE), (9, "x", ORANGE)]:
        rows = [r for r in q_rows if r["nodes"] == nodes]
        axes[0].scatter([r["T1"] for r in rows], [r["gain_percent"] for r in rows],
                        color=color, marker=marker, label=f"{nodes-1} links", s=35)
    axes[0].axhline(0, color=".6", linewidth=1)
    axes[0].set(xscale="log", xlabel="$T_1$ (slots)", ylabel="PE gain over matched Bell (%)",
                title="Original joint winner; same tree and operations")
    axes[0].set_xticks([74, 200, 600])
    axes[0].xaxis.set_major_formatter(ScalarFormatter())
    axes[0].xaxis.set_minor_formatter(NullFormatter())
    axes[0].legend(fontsize=8)
    s = load("protocol", 9, 200, seed=7)
    labels = ["Bell GP\nwinner", "Original\njoint winner", "Same joint\nprotocol, Bell"]
    records = [s["results"]["bell_all"]["best"], s["matched_controls"]["original_joint_winner"],
               s["matched_controls"]["same_protocol_bell"]]
    axes[1].bar(range(3), [r["result"]["skr"] for r in records], color=[GRAY, ORANGE, BLUE])
    axes[1].set(xticks=range(3), xticklabels=labels, ylabel="SKR (bits/slot)", ylim=(0, .034),
                title="8 links, $T_1=200$, seed 7")
    for i, r in enumerate(records):
        axes[1].text(i, r["result"]["skr"]+.001, f"q={r['candidate']['q']:.4f}", ha="center", fontsize=8)
    save(fig, "4-q-attribution")
    evidence["matched_q"] = q_rows
    (OUTPUT / "evidence.json").write_text(json.dumps(evidence, indent=2)+"\n")
    inputs = [RUN / "summary.csv", RUN / "audit.json", *RUN.glob("*/summary.json"), *RUN.glob("*/trials.jsonl")]
    (OUTPUT / "inputs.sha256.json").write_text(json.dumps(
        {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}, indent=2)+"\n")


if __name__ == "__main__":
    main()
