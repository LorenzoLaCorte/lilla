"""Read-only audit and figures for the completed metropolitan optimization.

Run: python -m src.analysis.plot_pauli_ad_metropolitan_report
Never calls the backend or an optimizer; original run files are unchanged.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

from src.analysis.plot_pauli_ad_hardware_report import entropy, flatten

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/"results/pauliAD/metropolitan-followup"
OUTPUT = Path(__file__).with_name("pauliAD-metropolitan-report")
BLUE, ORANGE, GRAY, GREEN = "#0072B2", "#D55E00", "#777777", "#009E73"


def read(path):
    return json.loads(path.read_text())


def collect():
    plan, manifest = read(RUN/"plan.json"), read(RUN/"manifest.json")
    assert len(plan) == len(manifest) == 288
    assert {r["status"] for r in manifest} == {"complete"}
    evaluations, winners, profiles, controls, jobs = [], [], [], [], []
    inputs = [RUN/"plan.json", RUN/"manifest.json"]
    requests = cached = 0
    physical = set()
    for case in plan:
        directory = Path(case["directory"])
        files = [directory/f for f in ("config.json", "environment.json", "runner_status.json", "summary.json", "trials.jsonl")]
        inputs.extend(files)
        config, env, status, summary = map(read, files[:4])
        meta = {k: case[k] for k in ("name", "section", "nodes", "regime", "memory_scale", "relaxation_model",
                                    "source_state", "geometry", "slot_seconds", "seed", "distance_km", "total_span_km")}
        meta.update(links=case["nodes"]-1, benchmark_fidelity=case["link_fidelity"],
                    source_visibility=config["hardware"]["visibility"], source_damping=config["hardware"]["source_damping"])
        assert np.isclose(case["slot_seconds"], 35e-6) and case["distance_km"] == 5
        records = [json.loads(line) for line in files[-1].read_text().splitlines()]
        unique = {r["candidate_id"]: r for r in records}
        requests += len(records)
        cached += sum(r["cached"] for r in records)
        for record in unique.values():
            result = record["result"]
            if result["status"] == "converged":
                n, history = config["numerics"], result["history"]
                assert len(history) >= 2
                assert min(history[-1]["coverage"], history[-2]["coverage"]) >= n["coverage"]
                keys = ("skr", "skf", "mean_wait", "eX", "eZ")
                assert np.allclose([history[-1][k] for k in keys], [history[-2][k] for k in keys], rtol=n["rtol"], atol=n["atol"])
                margin = 1-float(entropy(result["eX"]))-float(entropy(result["eZ"]))
                coordinates = result["pooled_coordinates"]
                assert np.allclose([result["eX"], result["eZ"]],
                                   [(1-coordinates[4])/2, (1-coordinates[3])/2], atol=1e-10, rtol=0)
                assert np.isclose(result["fidelity"], (1+coordinates[3]+coordinates[4]-coordinates[5])/4, atol=1e-10)
                assert abs(margin-result["bb84_margin"]) < 1e-12
                assert np.isclose(result["skr"], max(0, margin)/result["mean_wait"], atol=1e-12)
            physical.add(json.dumps([config["hardware"], record["candidate"]], sort_keys=True))
            evaluations.append(flatten(record, meta))
        for stage, info in summary["results"].items():
            record = info["best"]
            assert record and record["candidate_id"] in unique and record["result"]["status"] == "converged"
            winners.append(dict(**flatten(record, meta), stage=stage, certified=info.get("certified", False),
                                method=info.get("method", "GP best found"), space=info.get("space_size"),
                                converged_candidates=info.get("converged_candidates")))
        for reference, rows in summary.get("q_profiles", {}).items():
            profiles.extend(dict(**flatten(r, meta), reference=reference) for r in rows)
        for label, record in summary.get("matched_controls", {}).items():
            controls.append(dict(**flatten(record, meta), control=label))
        jobs.append(dict(**meta, elapsed_seconds=status["elapsed_seconds"], start=env["started"],
                         end=datetime.fromtimestamp(files[2].stat().st_mtime, timezone.utc).isoformat()))
    e, w, q, c, j = map(pd.DataFrame, (evaluations, winners, profiles, controls, jobs))
    audit = dict(jobs=len(plan), statuses=dict(Counter(r["status"] for r in manifest)),
                 requests=requests, cached_requests=cached, distinct_within_job_evaluations=len(e),
                 distinct_hardware_candidate_pairs=len(physical), candidate_statuses=e.status.value_counts().to_dict(),
                 unresolved_reasons=e[e.status != "converged"].reason.value_counts().to_dict(),
                 positive_evaluations=int(((e.status == "converged") & (e.skr > 0)).sum()),
                 stage_winners=len(w), positive_stage_winners=int((w.skr > 0).sum()),
                 q_profile_rows=len(q), q_profile_statuses=q.status.value_counts().to_dict(),
                 start=min(j.start), end=max(j.end), worker_seconds=float(j.elapsed_seconds.sum()))
    audit["wall_seconds"] = (datetime.fromisoformat(audit["end"])-datetime.fromisoformat(audit["start"])).total_seconds()
    for frame, name in ((e, "evaluations"), (w, "winners"), (q, "q_profiles"), (c, "matched-controls"), (j, "jobs")):
        frame.to_csv(OUTPUT/(name+".csv"), index=False)
    (OUTPUT/"audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    (OUTPUT/"inputs.sha256.json").write_text(json.dumps({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                         for p in inputs}, indent=2)+"\n")
    return e, w, q, c, audit


def save(fig, name):
    fig.savefig(OUTPUT/(name+".png"), dpi=200, bbox_inches="tight")
    fig.savefig(OUTPUT/(name+".pdf"), bbox_inches="tight")
    plt.close(fig)


def memory_axis(ax):
    ax.set(xscale="log", xticks=[.5, 1, 2], xticklabels=["0.5", "1", "2"], xlabel="Memory scale s")
    ax.minorticks_off()


def placement(w):
    p = w[w.section == "placement"]
    p.to_csv(OUTPUT/"placement.csv", index=False)
    for regime in ("metro97", "metro99"):
        for model in ("ad", "unital"):
            sub = p[(p.regime == regime) & (p.relaxation_model == model)]
            fig, axes = plt.subplots(2, 2, figsize=(9, 5.3), constrained_layout=True)
            for i, source_q in enumerate((.5, .25)):
                data = sub[sub.q == source_q]
                conditions = [(n, s) for n in (5, 9) for s in (.5, 1., 2.)]
                matrix = np.full((6, 3), np.nan)
                for row, (n, s) in enumerate(conditions):
                    level = data[(data.nodes == n) & (data.memory_scale == s) & (data.stage == "level")].iloc[0]
                    if level.skr <= 0:
                        continue
                    policies = json.loads(level.policies)
                    # Balanced power-of-two trees: boundary 2^h-1 has height h+1.
                    for h in range(int(np.log2(n-1))):
                        matrix[row, h] = ("all_corrected", "psi_pair", "phi_pair").index(policies[2**h-1])
                axes[i, 0].imshow(np.ma.masked_invalid(matrix), aspect="auto", vmin=0, vmax=2,
                                  cmap=ListedColormap([BLUE, ORANGE, GREEN]))
                for row in range(6):
                    for column in range(3):
                        v = matrix[row, column]
                        axes[i, 0].text(column, row, ("all", "Ψ", "Φ")[int(v)] if np.isfinite(v) else "—",
                                        color="white" if np.isfinite(v) else GRAY, ha="center", va="center")
                labels = [f"{n-1} links, s={s:g}" for n, s in conditions]
                axes[i, 0].set(xticks=[0, 1, 2], xticklabels=[1, 2, 3], yticks=range(6), yticklabels=labels,
                               xlabel="Swapping level (first → last)", title=("Bell" if i == 0 else "PE, q=0.25")+": best level schedule")
                for row, (n, s) in enumerate(conditions):
                    values = data[(data.nodes == n) & (data.memory_scale == s)].set_index("stage")
                    uniform, level = values.loc["uniform"], values.loc["level"]
                    if uniform.skr > 0:
                        ratio = level.skr/uniform.skr
                        axes[i, 1].barh(row+.13, 1., .23, color=GRAY, label="Best uniform" if row == 0 else None)
                        axes[i, 1].barh(row-.13, ratio, .23, color=ORANGE, label="Best level" if row == 0 else None)
                        axes[i, 1].text(ratio+.06, row-.13, f"{ratio:.2f}×", va="center", fontsize=7)
                    elif level.skr > 0:
                        axes[i, 1].text(.05, row, f"Key activated: 0 → {level.skr_bits_per_second:.3f} bits/s", va="center", fontsize=7, color=ORANGE)
                    else:
                        axes[i, 1].text(.05, row, "No key", va="center", color=GRAY, fontsize=7)
                axes[i, 1].set(yticks=range(6), yticklabels=[], ylim=(5.5, -.5), xlim=(0, 3.2),
                               xlabel="SKR / same-source best uniform SKR", title="Retry costs included")
                if axes[i, 1].get_legend_handles_labels()[0]:
                    axes[i, 1].legend(fontsize=7, loc="lower right")
            save(fig, f"1-placement-{regime}-{model}")
        fig, axes = plt.subplots(2, 2, figsize=(8.8, 4.8), constrained_layout=True)
        for i, source_q in enumerate((.5, .25)):
            for column, n in enumerate((5, 9)):
                ax = axes[i, column]
                for stage, color in (("uniform", GRAY), ("level", BLUE), ("node", ORANGE)):
                    for model, style in (("ad", "-"), ("unital", "--")):
                        sub = p[(p.regime == regime) & (p.q == source_q) & (p.nodes == n) & (p.stage == stage) & (p.relaxation_model == model)].sort_values("memory_scale")
                        ax.plot(sub.memory_scale, sub.skr_bits_per_second, style, marker="o", markersize=3, color=color, label=f"{stage}, {model}")
                memory_axis(ax)
                ax.set(ylabel="SKR (bits/s)", ylim=(0, None), title=("Bell" if i == 0 else "PE, q=0.25")+f"; {n-1} links")
        axes[0, 1].legend(fontsize=6, ncol=2)
        save(fig, f"1-rates-{regime}")


def distillation(e, w):
    d = w[w.section == "distillation"]
    families = [[("metro97", "perfect", "Perfect", BLUE)],
                [("metro97", "benchmark", "F0=0.97", BLUE), ("metro99", "benchmark", "F0=0.99", ORANGE)],
                [("metro97", "ad0.05", "AD 5%", GREEN), ("metro97", "ad0.15", "AD 15%", ORANGE)]]
    for model in ("ad", "unital"):
        fig, axes = plt.subplots(2, 3, figsize=(9.5, 4.7), constrained_layout=True)
        for row, n in enumerate((5, 9)):
            for column, family in enumerate(families):
                ax = axes[row, column]
                for regime, state, label, color in family:
                    for stage, style in (("bell_swaps", "--"), ("bell_policy", "-")):
                        sub = d[(d.regime == regime) & (d.source_state == state) & (d.nodes == n) & (d.stage == stage) & (d.relaxation_model == model)].sort_values("memory_scale")
                        ax.plot(sub.memory_scale, sub.skr_bits_per_second, style, marker="o", markersize=3, color=color,
                                label=f"{label}, {'swaps' if stage == 'bell_swaps' else 'best'}")
                        if stage == "bell_policy":
                            for _, r in sub[sub.skr > 0].iterrows():
                                ax.annotate(f"K={r.K}", (r.memory_scale, r.skr_bits_per_second), xytext=(1, 5), textcoords="offset points", fontsize=6)
                memory_axis(ax)
                ax.set(ylim=(0, None), title=f"{n-1} links; {('perfect source', 'mixed sources', 'damped sources')[column]}")
                ax.legend(fontsize=6, loc="upper left" if column in (1, 2) else "lower right")
                if column == 0:
                    ax.set_ylabel("Best found SKR (bits/s)")
        save(fig, "2-distillation-"+model)
    valid = e[(e.section == "distillation") & (e.status == "converged")]
    best = valid.sort_values(["skr", "bb84_margin"]).groupby(["name", "K"]).tail(1)
    best.to_csv(OUTPUT/"distillation-by-count.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(8.7, 4.5), constrained_layout=True)
    for row, n in enumerate((5, 9)):
        for column, regime in enumerate(("metro97", "metro99")):
            ax = axes[row, column]
            for k, color in ((0, GRAY), (1, BLUE), (2, ORANGE)):
                sub = best[(best.regime == regime) & (best.nodes == n) & (best.source_state == "benchmark") & (best.relaxation_model == "ad") & (best.K == k)].sort_values("memory_scale")
                ax.plot(sub.memory_scale, sub.skr_bits_per_second, "o-", color=color, label=f"K={k}")
            memory_axis(ax)
            ax.set(ylabel="Best sampled SKR (bits/s)", ylim=(0, None), title=f"{n-1} links; F0={.97 if regime == 'metro97' else .99}")
    axes[0, 1].legend(fontsize=7)
    save(fig, "2-counts")
    matched = []
    for n in (5, 9):
        balanced = ["s0", "s2", "s1"] if n == 5 else ["s0", "s2", "s1", "s4", "s6", "s5", "s3"]
        for k in (0, 1, 2):
            rows = valid[(valid.nodes == n) & (valid.protocol == json.dumps(balanced+[f"d{n-2}"]*k))
                         & (valid.policies == json.dumps(["all_corrected"]*(n-2)))]
            matched.extend(rows.to_dict("records"))
    pd.DataFrame(matched).to_csv(OUTPUT/"matched-terminal-recurrence.csv", index=False)


def asymmetry(w):
    a = w[w.section == "asymmetry"]
    gaps = a.pivot(index=["regime", "nodes", "memory_scale", "relaxation_model", "geometry"], columns="stage", values="skr_bits_per_second").reset_index()
    gaps["node_minus_level"] = gaps.node-gaps.level
    gaps.to_csv(OUTPUT/"asymmetry-gaps.csv", index=False)
    assert abs(gaps.node_minus_level).max() < 1e-10
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 2.7), constrained_layout=True)
    for ax, n in zip(axes, (5, 9)):
        for geometry, color in (("slow_link", ORANGE), ("weak_memory", BLUE)):
            for model, style in (("ad", "-"), ("unital", "--")):
                sub = gaps[(gaps.regime == "metro99") & (gaps.nodes == n) & (gaps.geometry == geometry) & (gaps.relaxation_model == model)].sort_values("memory_scale")
                ax.plot(sub.memory_scale, sub.node, style, marker="o", markersize=3, color=color, label=f"{geometry}, {model}")
        memory_axis(ax)
        ax.set(ylabel="SKR (bits/s)", ylim=(0, None), title=f"{n-1} links; F0=0.99; all classes coincide")
    axes[0].legend(fontsize=6)
    save(fig, "3-asymmetry")


def sources(w, q):
    s = w[w.section == "source"]
    pairs = s.pivot(index=["name", "regime", "nodes", "memory_scale", "relaxation_model", "source_state"], columns="stage", values="skr_bits_per_second").reset_index()
    pairs["relative_gain"] = pairs.joint/pairs.bell_policy.where(pairs.bell_policy > 0)-1
    pairs.to_csv(OUTPUT/"source-gains.csv", index=False)
    matched = []
    for (name, reference), rows in q.groupby(["name", "reference"]):
        valid = rows[rows.status == "converged"]
        bell = valid[valid.q == .5]
        if valid.empty or bell.empty:
            continue
        baseline = bell.iloc[0]
        best = valid.sort_values(["skr", "bb84_margin"]).iloc[-1]
        matched.append(dict(name=name, reference=reference, nodes=best.nodes, regime=best.regime,
                            memory_scale=best.memory_scale, relaxation_model=best.relaxation_model,
                            source_state=best.source_state, q=best.q if best.skr > 0 else None,
                            skr=best.skr_bits_per_second, bell_skr=baseline.skr_bits_per_second,
                            relative_gain=best.skr/baseline.skr-1 if baseline.skr > 0 else None,
                            protocol=best.protocol, policies=best.policies))
    pd.DataFrame(matched).to_csv(OUTPUT/"matched-source-gains.csv", index=False)
    states = [("metro97", "benchmark", "$F_0=0.97$"),
              ("metro99", "benchmark", "$F_0=0.99$"), ("metro99", "ad0.05", "Source AD 5%")]
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 4.6), constrained_layout=True)
    for row, n in enumerate((5, 9)):
        for column, (regime, state, label) in enumerate(states):
            ax = axes[row, column]
            for stage, color, marker in (("bell_policy", BLUE, "o"), ("joint", ORANGE, "x")):
                sub = s[(s.regime == regime) & (s.nodes == n) & (s.source_state == state) & (s.relaxation_model == "ad") & (s.stage == stage)].sort_values("memory_scale")
                ax.plot(sub.memory_scale, sub.skr_bits_per_second, "-", marker=marker, markersize=4, color=color,
                        label="Bell" if stage == "bell_policy" else "Optimized q")
                if stage == "joint":
                    for _, r in sub[sub.skr > 0].iterrows():
                        ax.annotate(f"q={r.q:.3f}", (r.memory_scale, r.skr_bits_per_second), xytext=(1, 5), textcoords="offset points", fontsize=6)
            memory_axis(ax)
            ax.set(ylim=(0, None), title=f"{n-1} links; {label}")
            if column == 0:
                ax.set_ylabel("SKR (bits/s)")
    axes[0, 0].legend(fontsize=7)
    save(fig, "4-source-rates")
    # Fully matched source comparison with no tree/operation changes.
    example = q[(q.regime == "metro99") & (q.nodes == 5) & (q.memory_scale == .5)
                & (q.source_state == "benchmark") & (q.relaxation_model == "ad")
                & (q.reference == "joint_winner") & (q.status == "converged")].sort_values("q")
    bell = example.loc[example.q == .5].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 2.6), constrained_layout=True)
    axes[0].plot(example.q, example.skr_bits_per_second, "o-", color=BLUE)
    axes[0].set(xlabel="q; same physical protocol", ylabel="SKR (bits/s)", title="4 links, F0=0.99, s=0.5, AD")
    zoom = example[example.q >= .45]
    axes[1].plot(zoom.q, 100*(zoom.skr_bits_per_second/bell.skr_bits_per_second-1), "o-", color=ORANGE)
    axes[1].axhline(0, color=GRAY, linewidth=.8)
    axes[1].set(xlabel="q (zoom near Bell)", ylabel="Matched SKR gain (%)", title="Maximum gain is about 0.184%")
    save(fig, "4-matched-q")


def convergence(e):
    counts = e.groupby(["section", "nodes", "status"]).size().unstack(fill_value=0)
    counts.to_csv(OUTPUT/"convergence.csv")
    fig, ax = plt.subplots(figsize=(8.7, 2.6), constrained_layout=True)
    positions = np.arange(len(counts))
    ax.bar(positions, counts.converged, color=BLUE, label="Converged")
    bottom = counts.converged.copy()
    for status, color in (("unconverged", ORANGE), ("failed", GRAY)):
        if status in counts:
            ax.bar(positions, counts[status], bottom=bottom, color=color, label=status)
            bottom += counts[status]
    ax.set(xticks=positions, xticklabels=[f"{a[:4]}\n{n-1} links" for a, n in counts.index],
           ylabel="Distinct evaluations within jobs", title="Convergence audit; completed jobs are not optimality certificates")
    ax.legend(fontsize=7)
    save(fig, "5-convergence")


def main():
    OUTPUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
    e, w, q, c, audit = collect()
    placement(w)
    distillation(e, w)
    asymmetry(w)
    sources(w, q)
    convergence(e)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
