"""Freeze completed twirled-model comparisons and draw report figures; no evaluation."""
from __future__ import annotations

from argparse import ArgumentParser
import collections
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CASES = {
    "cutoff_ad": ("L4_phase_error_F0.97_ad_s0.2_5km_68aa81d4", "swaps", "werner_source"),
    "cutoff_pauli": ("L4_phase_error_F0.97_pauli_s0.2_5km_6fc1f3f0", "swaps", "werner_source"),
    "joint_werner": ("L4_phase_error_F0.97_ad_s1_5km_36d8a11f", "joint", "werner_source"),
    "joint_channel": ("L8_werner_F0.99_ad_s0.2_5km_74aa41d7", "joint", "channel_twirl"),
    "zero_key_ad": ("L8_phase_error_F0.97_ad_s1_5km_b8f9638b", "joint", "werner_source"),
    "zero_key_pauli": ("L8_phase_error_F0.97_pauli_s1_5km_d3b1a836", "joint", "werner_source"),
    "large_positive_loss": ("L8_phase_error_F0.99_ad_s0.2_5km_ecd0ccf1", "joint", "werner_source"),
}


def audit_result(result, numerics):
    """Check saved scalar arithmetic and the actual convergence conditions."""
    if result["status"] != "converged":
        assert "skr" not in result  # Missing results cannot masquerade as zero.
        return
    history = result["history"]
    assert len(history) >= 2 and history[-1]["horizon"] == result["horizon"]
    assert history[-1]["horizon"] > history[-2]["horizon"]
    assert min(v["coverage"] for v in history[-2:]) >= numerics["coverage"]
    for k in ("skr", "skf", "mean_wait", "eX", "eZ"):
        assert math.isclose(history[-1][k], history[-2][k], rel_tol=numerics["rtol"], abs_tol=numerics["atol"])
    def entropy(p):
        assert 0 <= p <= 1
        return -p*math.log2(p)-(1-p)*math.log2(1-p) if 0 < p < 1 else 0.
    margin = 1-entropy(result["eX"])-entropy(result["eZ"])
    assert abs(margin-result["bb84_margin"]) < 1e-12
    assert abs(max(0., margin)-result["skf"]) < 1e-12
    assert math.isclose(result["skr"], result["skf"]/result["mean_wait"], rel_tol=1e-9, abs_tol=1e-12)
    x = result["pooled_coordinates"]
    assert abs(x[0]-1) < 1e-10
    assert abs((1-x[4])/2-result["eX"]) < 1e-10
    assert abs((1-x[3])/2-result["eZ"]) < 1e-10
    assert abs((1+x[3]+x[4]-x[5])/4-result["fidelity"]) < 1e-10


def snapshot(input_path, output):
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}

    def read(path):
        raw = path.read_bytes()
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    timestamp = datetime.now(timezone.utc).isoformat()
    plan = read(input_path/"plan.json")["cases"]
    manifest = read(input_path/"manifest.json")
    batch = read(input_path/"batch.json")
    completed = {r["name"] for r in manifest if r["status"] == "complete"}
    comparisons, diagnostics, jobs, examples = [], [], [], {}
    counts = collections.Counter()
    errors, horizon_reasons = collections.Counter(), collections.Counter()
    for c in plan:
        if c["name"] not in completed:
            continue
        s = read(Path(c["directory"])/"summary.json")
        meta, slot = s["spec"]["metadata"], s["spec"]["metadata"]["slot_seconds"]
        jobs.append(dict(name=c["name"], metadata=meta, candidates=s["candidates"],
                         numerical_status=s["numerical_status"]))
        all_records = {}
        for model, status_counts in s["numerical_status"].items():
            counts.update(status_counts)
            path = Path(c["directory"])/model/"trials.jsonl"
            raw = path.read_bytes()
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
            records = list(map(json.loads, raw.splitlines()))
            all_records[model] = {r["candidate_id"]: r for r in records}
            assert len(records) == len(all_records[model]) == s["candidates"]
            assert collections.Counter(r["result"]["status"] for r in records) == collections.Counter(status_counts)
            for r in records:
                v = r["result"]
                audit_result(v, s["spec"]["numerics"])
                if v["status"] == "failed":
                    errors[v["error"]] += 1
                elif v["status"] == "unconverged":
                    horizon_reasons[v.get("convergence_reason", "unknown")] += 1
        for r in s["comparisons"]:
            flat = {k: v for k, v in r.items() if k not in ("physical_winner_metrics", "selected_physical_metrics")}
            comparisons.append(dict(case=c["name"], links=meta["links"], nodes=meta["nodes"],
                                    source=meta["source"], noise=meta["noise"],
                                    memory_scale=meta["memory_scale"], fidelity=meta["fidelity_reference"], **flat))
            for label, (name, space, model) in CASES.items():
                if (c["name"], r["space"], r["model"]) != (name, space, model):
                    continue
                chosen = dict(r, case=name, spec=s["spec"])
                for key in ("physical_winner_metrics", "selected_physical_metrics"):
                    v = chosen[key]
                    assert v["status"] == "converged"
                    assert abs(v["skr"]-v["skf"]/v["mean_wait"]) < 1e-12
                    assert min(h["coverage"] for h in v["history"][-2:]) >= s["spec"]["numerics"]["coverage"]
                    chosen[key+"_readable"] = dict(skr_bits_second=v["skr"]/slot, skf=v["skf"],
                        wait_seconds=v["mean_wait"]*slot, fidelity=v["fidelity"], eX=v["eX"], eZ=v["eZ"],
                        bb84_margin=v["bb84_margin"],
                        last_skr_change=abs(v["history"][-1]["skr"]-v["history"][-2]["skr"]), horizon=v["horizon"])
                # Distinguish the common-subset choice from the model's own
                # argmax over all its converged candidates in the requested space.
                from src.optimization.twirled_regret import in_space
                records = {}
                for m in ("full", model):
                    valid = [v for v in all_records[m].values() if v["result"]["status"] == "converged"
                             and in_space(v["candidate"], space)]
                    winner = min(valid, key=lambda v: (-v["result"]["skr"], v["candidate_id"]))
                    records[m] = dict(id=winner["candidate_id"], skr=winner["result"]["skr"])
                chosen["own_converged_argmax"] = records
                chosen["selection_matches_own_argmax"] = records[model]["id"] == r["selected_id"]
                chosen["reference_matches_own_argmax"] = records["full"]["id"] == r["physical_winner_id"]
                chosen["cross_model_scores"] = {
                    label: {m: all_records[m][key]["result"] for m in ("full", model)}
                    for label, key in (("full_choice", r["physical_winner_id"]), ("approx_choice", r["selected_id"]))}
                examples[label] = chosen
        diagnostics += [dict(case=c["name"], links=meta["links"], source=meta["source"], noise=meta["noise"], **r)
                        for r in s["diagnostics"]]
    counts_table = []
    for space in ("swaps", "distillation", "postselection", "joint"):
        for model in ("bell_source", "werner_source", "channel_twirl", "bell_channel_twirl", "bell_each_stage"):
            rows = [r for r in comparisons if r["space"] == space and r["model"] == model]
            complete = [r for r in rows if r["complete"]]
            positive = [r for r in complete if r.get("physical_best", 0) > 0]
            loss = [r for r in positive if not r["all_zero_prediction"] and r["robust_relative_loss"] > 1e-4]
            counts_table.append(dict(space=space, model=model, completed=len(rows), complete=len(complete),
                                     positive_reference=len(positive), loss_above_001_percent=len(loss)))
    sizes = []
    for links in (2, 3, 4, 8):
        names = {c["name"] for c in plan if c["spec"]["metadata"]["links"] == links}
        times = [r["elapsed_seconds"] for r in manifest if r["name"] in names and r["status"] == "complete"]
        sizes.append(dict(links=links, completed=len(times), planned=len(names),
                          mean_job_seconds=sum(times)/len(times) if times else None))
    paper = Path.home()/"Research/qcnc2025-paper/main.tex"
    hashes[str(paper)] = hashlib.sha256(paper.read_bytes()).hexdigest()
    result = dict(extracted_at_utc=timestamp, input=str(input_path.relative_to(ROOT)),
                  batch_started=batch["started"], batch_status=batch["status"],
                  batch_ended=batch.get("ended"), failure_messages=dict(errors), horizon_reasons=dict(horizon_reasons),
                  completed_cases=len(jobs), planned_cases=len(plan), completed_case_evaluation_status=dict(counts),
                  sizes=sizes, certification=counts_table, examples=examples,
                  comparisons=comparisons, diagnostics=diagnostics, jobs=jobs, input_sha256=hashes)
    (output/"evidence.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


def plots(data, output):
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/lilla-twirl-report-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    examples = data["examples"]
    labels = ("cutoff_pauli", "cutoff_ad", "joint_werner", "joint_channel")
    titles = ("4 links, Pauli: Werner cutoff", "4 links, AD: Werner cutoff",
              "4 links, AD: Werner joint (partial)", "8 links, AD: channel twirl (partial)")
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.5), constrained_layout=True)
    for ax, key, title in zip(axes.flat, labels, titles):
        r = examples[key]
        values = [r["physical_winner_metrics_readable"]["skr_bits_second"],
                  r["selected_physical_metrics_readable"]["skr_bits_second"]]
        ax.bar([0, 1], values, color=["#0072B2", "#D55E00"], width=.55)
        ax.set_xticks([0, 1], ["Full-model choice", "Approx.-model choice"])
        ax.tick_params(axis="x", labelsize=8)
        ax.set(title=title, ylabel="Physical SKR (bits/s)", ylim=(0, 1.3*max(values)))
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:.5g}", ha="center", va="bottom", fontsize=9)
        ax.text(.98, .94, f"Loss: {100*r['relative_loss']:.2f}%\n{r['paired']}/{r['candidates']} pairs resolved",
                ha="right", va="top", transform=ax.transAxes, fontsize=8)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    for ext in ("png", "pdf"):
        fig.savefig(output/f"physical-rate-comparisons.{ext}", dpi=170)
    plt.close(fig)

    # Size trends only within a fixed hardware/source/noise/figure-of-merit family.
    fig, ax = plt.subplots(figsize=(6.8, 3.1), constrained_layout=True)
    for noise, color in (("pauli", "#0072B2"), ("ad", "#D55E00")):
        rows = [r for r in data["comparisons"] if r["space"] == "swaps" and r["model"] == "werner_source"
                and r["source"] == "phase_error" and r["fidelity"] == .97 and r["memory_scale"] == .2
                and r["noise"] == noise and r["complete"] and r.get("physical_best", 0) > 0]
        rows.sort(key=lambda r: r["links"])
        ax.plot([r["links"] for r in rows], [100*r["relative_loss"] for r in rows], "o-", color=color, label=noise)
    ax.set(xlabel="Elementary links (N−1)", ylabel="Physical SKR loss (%)")
    ax.set_title("Phase-error source; fidelity-matched Werner approximation\n"
                 "F0=0.97, memory scale 0.2; cutoff-only space", fontsize=10)
    ax.set_xticks([2, 3, 4, 8])
    ax.legend()
    ax.grid(alpha=.2)
    for ext in ("png", "pdf"):
        fig.savefig(output/f"cutoff-loss-by-size.{ext}", dpi=170)
    plt.close(fig)

    if "zero_key_ad" in examples and "zero_key_pauli" in examples:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
        for ax, key, title in zip(axes, ("zero_key_pauli", "zero_key_ad"), ("Pauli storage", "AD plus dephasing")):
            r = examples[key]
            for offset, model, label, color in ((-.18, "full", "Full-model execution", "#0072B2"),
                                               (.18, "werner_source", "Werner-source prediction", "#D55E00")):
                values = [r["cross_model_scores"][choice][model]["skr"]/r["spec"]["metadata"]["slot_seconds"]
                          for choice in ("full_choice", "approx_choice")]
                ax.bar([offset, 1+offset], values, width=.34, color=color, label=label)
                for i, value in enumerate(values):
                    ax.text(i+offset, value+.025, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
            ax.set_xticks([0, 1], ["No distillation\ncutoff 544", "One round on each half\nno cutoff"])
            ax.tick_params(axis="x", labelsize=8)
            ax.set(title=f"{title}: {r['paired']}/{r['candidates']} pairs resolved", ylabel="SKR (bits/s)", ylim=(0, 1.7))
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
        axes[1].legend(fontsize=7, loc="upper right")
        fig.suptitle("Eight-link phase-error source: predicted and physical rankings reverse", fontsize=11)
        for ext in ("png", "pdf"):
            fig.savefig(output/f"eight-link-ranking-reversal.{ext}", dpi=170)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.8, 3.2), constrained_layout=True)
        for noise, color in (("pauli", "#0072B2"), ("ad", "#D55E00")):
            rows = [r for r in data["comparisons"] if r["space"] == "joint" and r["model"] == "werner_source"
                    and r["source"] == "phase_error" and r["fidelity"] == .97 and r["memory_scale"] == 1
                    and r["noise"] == noise and not r.get("all_zero_prediction", True)
                    and r.get("physical_best", 0) > 0]
            rows.sort(key=lambda r: r["links"])
            ax.plot([r["links"] for r in rows], [100*r["relative_loss"] for r in rows], "o--",
                    markerfacecolor="white", color=color, label=noise)
        ax.set(xlabel="Elementary links (N−1)", ylabel="Converged-subset SKR loss (%)", ylim=(-3, 106))
        ax.set_title("Phase-error source, F0=0.97, reference memories\nJoint spaces are incomplete at every size", fontsize=10)
        ax.set_xticks([2, 3, 4, 8])
        ax.legend()
        ax.grid(alpha=.2)
        for ext in ("png", "pdf"):
            fig.savefig(output/f"joint-loss-by-size.{ext}", dpi=170)
        plt.close(fig)


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=ROOT/"results/twirled-regret/noise-size")
    p.add_argument("--output", type=Path, default=Path(__file__).with_name("twirled-model-optimization"))
    p.add_argument("--replot", action="store_true", help="Use frozen evidence, without rereading the live batch")
    args = p.parse_args()
    if args.replot:
        data = json.loads((args.output/"evidence.json").read_text())
    else:
        data = snapshot(args.input.resolve(), args.output)
    plots(data, args.output)
    print(json.dumps({k: data[k] for k in ("extracted_at_utc", "completed_cases", "planned_cases",
                                         "completed_case_evaluation_status", "sizes")}))


if __name__ == "__main__":
    main()
