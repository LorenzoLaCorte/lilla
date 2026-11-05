"""Audit and plot saved hardware results; never call an optimizer/evaluator.

python -m src.analysis.plot_pauli_ad_hardware_report
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/"results/pauliAD/hardware-followup"
OUTPUT = Path(__file__).with_name("pauliAD-hardware-report")
BLUE, ORANGE, GRAY = "#0072B2", "#D55E00", "#777777"


def read(path):
    return json.loads(path.read_text())


def entropy(p):
    p = np.asarray(p, dtype=float)
    result = np.zeros_like(p)
    inside = (p > 0) & (p < 1)
    result[inside] = -p[inside]*np.log2(p[inside])-(1-p[inside])*np.log2(1-p[inside])
    return result


def flatten(record, meta):
    r, c = record["result"], record["candidate"]
    metrics = r if r["status"] == "converged" else r.get("diagnostics", {})
    row = dict(**meta, candidate_id=record["candidate_id"], request_id=record["request_id"],
               status=r["status"], reason=r.get("convergence_reason"), horizon=r["horizon"],
               q=c["q"], K=sum(s.startswith("d") for s in c["protocol"]),
               protocol=json.dumps(c["protocol"]), policies=json.dumps(c["policies"]))
    row.update({k: metrics.get(k) for k in ("skr", "skf", "bb84_margin", "eX", "eZ", "fidelity", "coverage", "mean_wait")})
    row["wait_seconds"] = metrics.get("mean_wait", np.nan)*meta["slot_seconds"]
    row["skr_bits_per_second"] = metrics.get("skr", np.nan)/meta["slot_seconds"]
    for label, value in zip(("cII", "cIZ", "cZI", "cZZ", "cXX", "cYY"), metrics.get("pooled_coordinates", [])):
        row[label] = value
    return row


def collect():
    plan, manifest = read(RUN/"plan.json"), read(RUN/"manifest.json")
    assert len(plan) == len(manifest) == 144
    assert {r["status"] for r in manifest} == {"complete"}
    evaluations, winners, profiles, jobs, inputs = [], [], [], [], [RUN/"plan.json", RUN/"manifest.json"]
    requests = cached = 0
    for case in plan:
        directory = Path(case["directory"])
        files = [directory/f for f in ("config.json", "environment.json", "runner_status.json", "summary.json", "trials.jsonl")]
        inputs.extend(files)
        config, env, status, summary = map(read, files[:4])
        meta = {k: case[k] for k in ("name", "section", "nodes", "memory_scale", "relaxation_model", "source_state", "geometry", "slot_seconds", "seed")}
        meta["links"] = case["nodes"]-1
        records = [json.loads(line) for line in files[-1].read_text().splitlines()]
        unique = {r["candidate_id"]: r for r in records}
        requests += len(records)
        cached += sum(r["cached"] for r in records)
        for record in unique.values():
            result = record["result"]
            if result["status"] == "converged":
                n = config["numerics"]
                history = result["history"]
                assert len(history) >= 2
                assert min(history[-1]["coverage"], history[-2]["coverage"]) >= n["coverage"]
                keys = ("skr", "skf", "mean_wait", "eX", "eZ")
                assert np.allclose([history[-1][k] for k in keys], [history[-2][k] for k in keys], rtol=n["rtol"], atol=n["atol"])
                margin = 1-float(entropy(result["eX"]))-float(entropy(result["eZ"]))
                assert abs(margin-result["bb84_margin"]) < 1e-12
                assert np.isclose(result["skr"], max(0, margin)/result["mean_wait"], atol=1e-12)
            evaluations.append(flatten(record, meta))
        for stage, info in summary["results"].items():
            record = info["best"]
            assert record and record["candidate_id"] in unique
            winners.append(dict(**flatten(record, meta), stage=stage, certified=info.get("certified", False),
                                method=info.get("method", "GP best found"), space=info.get("space_size"),
                                converged_candidates=info.get("converged_candidates")))
        for reference, records in summary.get("q_profiles", {}).items():
            profiles.extend(dict(**flatten(r, meta), reference=reference) for r in records)
        jobs.append(dict(**meta, elapsed_seconds=status["elapsed_seconds"],
                         start=env["started"], end=datetime.fromtimestamp(files[2].stat().st_mtime, timezone.utc).isoformat()))
    e, w, q, j = map(pd.DataFrame, (evaluations, winners, profiles, jobs))
    valid = e[e.status == "converged"]
    assert valid.skr.eq(0).all() and valid.bb84_margin.lt(0).all()
    assert w.skr.eq(0).all() and w.status.eq("converged").all()
    audit = dict(jobs=len(plan), job_statuses=dict(Counter(r["status"] for r in manifest)),
                 requests=requests, cached_requests=cached, distinct_within_job_evaluations=len(e),
                 candidate_statuses=e.status.value_counts().to_dict(),
                 unresolved_reasons=e[e.status != "converged"].reason.value_counts().to_dict(),
                 winner_rows=len(w), positive_winner_rows=int((w.skr > 0).sum()),
                 q_profile_rows=len(q), q_profile_statuses=q.status.value_counts().to_dict(),
                 maximum_converged_margin=float(valid.bb84_margin.max()),
                 maximum_unconverged_diagnostic_margin=float(e[e.status != "converged"].bb84_margin.max()),
                 worker_seconds=float(j.elapsed_seconds.sum()),
                 start=min(j.start), end=max(j.end))
    audit["wall_seconds"] = (datetime.fromisoformat(audit["end"])-datetime.fromisoformat(audit["start"])).total_seconds()
    for frame, name in ((e, "evaluations"), (w, "winners"), (q, "q_profiles"), (j, "jobs")):
        frame.to_csv(OUTPUT/(name+".csv"), index=False)
    (OUTPUT/"audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    (OUTPUT/"inputs.sha256.json").write_text(json.dumps({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                         for p in inputs}, indent=2)+"\n")
    return e, w, q, audit


def save(fig, name):
    fig.savefig(OUTPUT/(name+".png"), dpi=200, bbox_inches="tight")
    fig.savefig(OUTPUT/(name+".pdf"), bbox_inches="tight")
    plt.close(fig)


def memory_axis(ax):
    ax.set(xscale="log", xticks=[.5, 1, 2], xticklabels=["0.5", "1", "2"], xlabel="Memory scale s")
    ax.minorticks_off()


def placement(w):
    p = w[w.section == "placement"]
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5), constrained_layout=True, sharey=True)
    for i, source_q in enumerate((.5, .25)):
        for column, nodes in enumerate((5, 9)):
            ax = axes[i, column]
            for model, style in (("ad", "-"), ("unital", "--")):
                for stage, color, marker in (("uniform", GRAY, "o"), ("level", BLUE, "s"), ("node", ORANGE, "x")):
                    sub = p[(p.q == source_q) & (p.nodes == nodes) & (p.relaxation_model == model) & (p.stage == stage)].sort_values("memory_scale")
                    ax.plot(sub.memory_scale, sub.bb84_margin, style, marker=marker, markersize=4, color=color,
                            label=f"{stage}, {model}", alpha=.85)
            ax.axhline(0, color="black", linewidth=.8)
            memory_axis(ax)
            ax.set(title=("Bell" if source_q == .5 else "PE, q=0.25")+f"; {nodes-1} links", ylim=(-1.03, .06))
            if column == 0:
                ax.set_ylabel("Signed BB84 margin m (not SKF)")
    axes[0, 1].legend(fontsize=6, ncol=2, loc="upper left")
    save(fig, "1-placement")


def distillation(e):
    v = e[(e.section == "distillation") & (e.status == "converged") & (e.memory_scale == 2)]
    best = v.sort_values("bb84_margin").groupby(["nodes", "relaxation_model", "source_state", "K"]).tail(1)
    best.to_csv(OUTPUT/"distillation-by-count-scale2.csv", index=False)
    states = ("perfect", "benchmark", "ad0.05", "ad0.15")
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 4.6), sharey=True, constrained_layout=True)
    for i, model in enumerate(("ad", "unital")):
        for j, nodes in enumerate((5, 9)):
            ax = axes[i, j]
            for k, color in ((0, GRAY), (1, BLUE), (2, ORANGE)):
                sub = best[(best.relaxation_model == model) & (best.nodes == nodes) & (best.K == k)].set_index("source_state")
                ax.bar(np.arange(4)+(k-1)*.23, [sub.loc[s, "bb84_margin"] for s in states], .22, color=color, label=f"K={k}")
            ax.axhline(0, color="black", linewidth=.8)
            ax.set(xticks=np.arange(4), xticklabels=["Perfect", "Mixed", "AD 5%", "AD 15%"],
                   title=f"{nodes-1} Bell links; {model}; s=2", ylim=(-1.04, .04))
            if j == 0:
                ax.set_ylabel("Best sampled signed margin m")
    axes[0, 1].legend(ncol=3, fontsize=7, loc="lower left")
    save(fig, "2-distillation")


def matched_recurrence(e):
    # Match the actual operations, not just separate search winners.
    records = []
    for nodes in (5, 9):
        balanced = ["s0", "s2", "s1"] if nodes == 5 else ["s0", "s2", "s1", "s4", "s6", "s5", "s3"]
        sub = e[(e.section == "distillation") & (e.nodes == nodes) & (e.status == "converged")]
        for k in (0, 1, 2):
            match = sub[(sub.protocol == json.dumps(balanced+[f"d{nodes-2}"]*k)) &
                        (sub.policies == json.dumps(["all_corrected"]*(nodes-2)))]
            records.extend(match.to_dict("records"))
    frame = pd.DataFrame(records)
    assert len(frame) == 144
    frame.to_csv(OUTPUT/"matched-recurrence.csv", index=False)
    f = frame[(frame.nodes == 5) & (frame.relaxation_model == "ad") & (frame.source_state == "perfect") & (frame.memory_scale == 2)].sort_values("K")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.5), constrained_layout=True)
    for key, color, label in (("eX", ORANGE, "$e_X$"), ("eZ", BLUE, "$e_Z$")):
        axes[0].plot(f.K, f[key], "o-", color=color, label=label)
    axes[0].set(ylabel="Pooled error probability", ylim=(0, .52), title="Bit errors fall; phase errors grow")
    axes[0].legend(fontsize=8)
    for key, color, label in (("cXX", ORANGE, "$c_{XX}$"), ("cZZ", BLUE, "$c_{ZZ}$"), ("fidelity", GRAY, "$F_{\Phi^+}$")):
        axes[1].plot(f.K, f[key], "o-", color=color, label=label)
    axes[1].set(ylabel="Correlation / fidelity", ylim=(0, 1.05), title="A better margin need not mean better fidelity")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set(xticks=[0, 1, 2], xlabel="Terminal recurrence rounds K")
    save(fig, "2-matched-recurrence")


def asymmetry(w):
    a = w[w.section == "asymmetry"]
    pairs = a.pivot(index=["nodes", "memory_scale", "relaxation_model", "geometry"], columns="stage", values="bb84_margin").reset_index()
    pairs["node_minus_level"] = pairs.node-pairs.level
    pairs["level_minus_uniform"] = pairs.level-pairs.uniform
    pairs.to_csv(OUTPUT/"asymmetry-gaps.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.6), constrained_layout=True)
    for ax, nodes in zip(axes, (5, 9)):
        for geometry, color in (("slow_link", ORANGE), ("weak_memory", BLUE)):
            for model, style in (("ad", "-"), ("unital", "--")):
                sub = pairs[(pairs.nodes == nodes) & (pairs.geometry == geometry) & (pairs.relaxation_model == model)].sort_values("memory_scale")
                ax.plot(sub.memory_scale, sub.node_minus_level, style, marker="o", color=color, label=f"{geometry}, {model}")
        memory_axis(ax)
        ax.set(title=f"{nodes-1} links; observed diagnostic gap", ylabel="$m_{node}-m_{level}$", ylim=(-.00012, .0029))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1].legend(fontsize=6, loc="upper left")
    save(fig, "3-asymmetry")


def sources(q, w):
    v = q[q.status == "converged"]
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 2.6), constrained_layout=True)
    axes[0].scatter(v.q, v.skr_bits_per_second, s=12, color=BLUE, alpha=.3)
    axes[0].set(xlabel="Profiled q", ylabel="SKR (bits/s)", title="All converged q profiles: zero key", ylim=(-.015, .015), yticks=[0])
    example = v[(v.nodes == 9) & (v.source_state == "perfect") & (v.memory_scale == .5) & (v.reference == "joint_winner")]
    for model, color in (("ad", ORANGE), ("unital", BLUE)):
        sub = example[example.relaxation_model == model].sort_values("q")
        axes[1].plot(sub.q, sub.bb84_margin, "o-", markersize=3, color=color, label=model)
    axes[1].set_yscale("symlog", linthresh=1e-7)
    axes[1].set(xlabel="q, fixed protocol within each curve", ylabel="Signed margin m",
                title="Margin improvement is not a rate gain")
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[1].legend(fontsize=7)
    classical = example[(example.relaxation_model == "ad") & (example.q == .1)].iloc[0]
    axes[2].bar([0, 1, 2], [classical.cXX, classical.cZZ, classical.fidelity], color=[ORANGE, BLUE, GRAY])
    axes[2].set(xticks=[0, 1, 2], xticklabels=["$c_{XX}$", "$c_{ZZ}$", "$F_{\Phi^+}$"], ylim=(0, 1.15), title="q=0.1: almost classical")
    axes[2].text(0, .035, f"{classical.cXX:.1e}", ha="center", fontsize=7)
    save(fig, "4-source")
    (OUTPUT/"classical-example.json").write_text(json.dumps(classical.to_dict(), indent=2)+"\n")
    # Human-readable paired controls; no division by zero in rate comparisons.
    source_w = w[w.section == "source"]
    pairs = source_w.pivot(index=["name", "nodes", "memory_scale", "relaxation_model", "source_state"], columns="stage", values="bb84_margin")
    pairs.to_csv(OUTPUT/"source-margins.csv")


def convergence(e):
    counts = e.groupby(["section", "nodes", "status"]).size().unstack(fill_value=0)
    counts["unresolved_fraction"] = counts.get("unconverged", 0)/counts.sum(axis=1)
    counts.to_csv(OUTPUT/"convergence.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.6), constrained_layout=True, gridspec_kw={"width_ratios": [1.8, 1]})
    positions = np.arange(len(counts))
    axes[0].bar(positions, counts.converged, color=BLUE, label="Converged")
    axes[0].bar(positions, counts.unconverged, bottom=counts.converged, color=ORANGE, label="Horizon unresolved")
    axes[0].set(xticks=positions, xticklabels=[f"{section[:4]}\n{nodes-1} links" for section, nodes in counts.index],
                ylabel="Distinct evaluations within jobs", title="Job completion is not candidate convergence")
    axes[0].legend(fontsize=7)
    reasons = e[e.status != "converged"].reason.value_counts()
    axes[1].bar(range(len(reasons)), reasons.values, color=[ORANGE, GRAY])
    axes[1].set(xticks=range(len(reasons)), xticklabels=["Needs second\ncovered horizon" if k == "confirmation" else "Coverage\nbelow target" for k in reasons.index], title="No physicality failures")
    for i, value in enumerate(reasons.values):
        axes[1].text(i, value+15, str(value), ha="center", fontsize=8)
    save(fig, "5-convergence")


def main():
    OUTPUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
    e, w, q, audit = collect()
    placement(w)
    distillation(e)
    matched_recurrence(e)
    asymmetry(w)
    sources(q, w)
    convergence(e)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
