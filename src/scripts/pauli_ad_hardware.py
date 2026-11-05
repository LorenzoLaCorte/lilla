"""Hardware follow-up plan and read-only figures for the four report questions.

Used by ``python -m src.scripts.gp_pauliAD --preset hardware``.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

from src.scripts.reproduce_results import DEFAULT_HOMOGENEOUS_CASES

SECTIONS = ("placement", "distillation", "asymmetry", "source")
MEMORY_REFERENCE = "https://doi.org/10.1038/s41534-022-00637-w"
FIBER_REFERENCE = "https://doi.org/10.1103/PhysRevX.4.041038"
FIDELITY_REFERENCES = {
    "metro97": "https://doi.org/10.1038/s41467-025-57557-4",
    "metro99": "https://doi.org/10.1103/PRXQuantum.5.030330",
}


def metropolitan_link(distance_km=5., fidelity=.97):
    """Optimistic hybrid link, not a measured integrated memory interface.

    Fit a distance-independent attempt prefactor to the old C/D benchmarks.
    New passive fiber: 5 km / 1.5 dB including .2 dB connectors (Agarwal 2014).
    The two midpoint arms have total photon path L; do not square the total
    path transmission again. A local 10 us overhead is explicitly assumed.
    """
    if not np.isfinite(distance_km) or distance_km <= 0 or not 0 < fidelity <= 1:
        raise ValueError("metropolitan length and fidelity must be positive and finite")
    c = next(c for c in DEFAULT_HOMOGENEOUS_CASES if c.label.startswith("C_"))
    d = next(c for c in DEFAULT_HOMOGENEOUS_CASES if c.label.startswith("D_"))
    fitted_alpha = 10*np.log10(d.p_gen/c.p_gen)/(50.-20.)
    prefactor = float(c.p_gen*10**(fitted_alpha*50/10))
    alpha, insertion, local_seconds = .26, .2, 10e-6
    loss = alpha*distance_km+insertion
    return dict(distance_km=float(distance_km), fiber_loss_db_per_km=alpha,
                insertion_loss_db=insertion, channel_loss_db=loss,
                channel_transmission=10**(-loss/10), p_gen_zero_loss=prefactor,
                calibration_loss_db_per_km=float(fitted_alpha), local_attempt_seconds=local_seconds,
                slot_seconds=distance_km*1000/2e8+local_seconds,
                probability=prefactor*10**(-loss/10), link_fidelity=fidelity,
                fiber_reference=FIBER_REFERENCE)


def hardware(regime, profile, nodes, scale, geometry="balanced", *, metro_link_km=5.):
    """Reuse benchmark probabilities/visibility; all lifetimes below are per qubit.

    Bradley et al. report effective longitudinal/transverse decay during network
    operation: 2.4 s and 1.71 s. Their relaxation is NOT evidence for zero-
    temperature AD; our non-unital AD model remains an explicit approximation.
    """
    metro = regime in FIDELITY_REFERENCES
    case = next(c for c in DEFAULT_HOMOGENEOUS_CASES if c.label.startswith(("C" if metro else regime)+"_"))
    optical = metropolitan_link(metro_link_km, .97 if regime == "metro97" else .99) if metro else {}
    slot_seconds = optical["slot_seconds"] if metro else {"C": 50_000., "D": 20_000.}[regime]/2e8
    if profile == "network":
        t1, t2 = 2.4/slot_seconds, 1.71/slot_seconds
    elif profile == "benchmark":
        if metro:
            raise ValueError("metropolitan regimes use the network memory profile; historical benchmark units apply only to C/D")
        # Match the scalar homogeneous Werner cXX decay exp(-t/t_coh):
        # two local transverse decays give exp(-2t/T2). T1 is an assumption.
        t2 = 2*case.t_coh
        t1 = 10*t2
    else:
        raise ValueError(f"unknown memory profile {profile}")
    p_gen = [float(optical["probability"] if metro else case.p_gen)]*(nodes-1)
    t1s, t2s = [float(t1*scale)]*nodes, [float(t2*scale)]*nodes
    if geometry in ("slow_link", "bottleneck"):
        p_gen[-1] /= 5
    if geometry in ("weak_memory", "bottleneck"):
        # Internal node, not just an end memory; breaks exchange symmetry.
        t1s[-2] /= 4
        t2s[-2] /= 4
    result = dict(regime=regime, memory_profile=profile, memory_scale=scale,
                slot_seconds=slot_seconds, p_gen=p_gen, p_swap=case.p_swap,
                visibility=(4*optical["link_fidelity"]-1)/3 if metro else float(case.w0), t1_slots=t1s, t2_slots=t2s,
                reference=MEMORY_REFERENCE if profile == "network" else "src/scripts/reproduce_results.py",
                geometry=geometry)
    if metro:
        result.update({k: v for k, v in optical.items() if k not in ("probability", "slot_seconds")})
        result.update(total_span_km=metro_link_km*(nodes-1), fidelity_reference=FIDELITY_REFERENCES[regime])
    return result


def cases(args):
    nodes = args.nodes or [5, 9]
    seeds = args.seeds or [7, 42]
    plan = []
    for regime in args.hardware_regimes:
        for profile in args.memory_profiles:
            for node in nodes:
                for scale in args.memory_scales:
                    specs = []
                    if "placement" in args.sections:
                        specs += [("placement", "balanced", q, "benchmark", 0.) for q in (.5, .25)]
                    if "distillation" in args.sections:
                        specs += [("distillation", "balanced", .5, state, damping)
                                  for state, damping in [("perfect", 0.), ("benchmark", 0.),
                                      *[(f"ad{d:g}", d) for d in args.source_dampings]]]
                    if "asymmetry" in args.sections:
                        specs += [("asymmetry", geometry, .5, "benchmark", 0.)
                                  for geometry in args.asymmetry_geometries]
                    if "source" in args.sections:
                        specs += [("source", "balanced", None, state, damping)
                                  for state, damping in [("perfect", 0.), ("benchmark", 0.),
                                      *[(f"ad{d:g}", d) for d in args.source_dampings]]]
                    for section, geometry, q, state, damping in specs:
                        h = hardware(regime, profile, node, scale, geometry,
                                     metro_link_km=getattr(args, "metro_link_km", 5.))
                        visibility = h["visibility"] if state == "benchmark" else 1.
                        for seed in seeds:
                            name = (f"{section}_{regime}_{profile}_N{node}_scale{scale:g}_"
                                    f"{geometry}_{state}_q{q if q is not None else 'opt'}_seed{seed}")
                            directory = args.output/name
                            placement = section in ("placement", "asymmetry")
                            module = "optimize_postselection" if placement else "gp_pauliAD"
                            command = [sys.executable, "-m", f"src.optimization.{module}",
                                       "--nodes", str(node), "--p-gen", *map(str, h["p_gen"]),
                                       "--p-swap", str(h["p_swap"]),
                                       "--p-distillation", str(args.p_distillation),
                                       "--t1-slots", *map(str, h["t1_slots"]),
                                       "--t2-slots", *map(str, h["t2_slots"]),
                                       "--visibility", str(visibility), "--source-damping", str(damping),
                                       "--initial-horizon", str(args.initial_horizon),
                                       "--max-horizon", str(args.max_horizon),
                                       "--gp-calls", str(args.gp_calls), "--initial-points", str(args.initial_points),
                                       "--seed", str(seed), "--output", str(directory)]
                            if placement:
                                command += ["--q", str(q), "--tree", "left" if geometry == "left_comb" else "balanced",
                                            "--method", args.post_method, "--max-exact", str(args.max_exact)]
                            else:
                                command += ["--search-mode", "bell" if section == "distillation" else "source",
                                            "--max-dists", str(args.max_dists),
                                            "--max-total-dists", str(args.max_total_dists), "--q-min", str(args.q_min)]
                                if section == "source":
                                    command += ["--q-grid", *map(str, args.q_grid)]
                            if args.resume:
                                command.append("--resume")
                            base = dict(name=name, study="postselection" if placement else "protocol",
                                             section=section, nodes=node, t1=h["t1_slots"][0], q=q, seed=seed,
                                             directory=str(directory), command=command, **h,
                                             source_state=state, source_damping=damping, source_visibility=visibility)
                            for model in args.relaxation_models:
                                new = dict(base, relaxation_model=model)
                                new["name"] = name+"_"+model
                                new["directory"] = str(args.output/new["name"])
                                new["command"] = list(command)
                                new["command"][new["command"].index("--output")+1] = new["directory"]
                                if model == "unital":
                                    # Exact unital Bloch decay with these T1/T2,
                                    # supported by the existing Pauli backend.
                                    first = new["command"].index("--t1-slots")
                                    last = new["command"].index("--visibility")
                                    rates = [v for t1, t2 in zip(h["t1_slots"], h["t2_slots"])
                                             for v in (1/t2, 1/t2, 1/t1)]
                                    new["command"][first:last] = ["--pauli-rates", *map(str, rates), "--ad-rates", "0"]
                                plan.append(new)
    return plan


def write_table(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def saved_trials(directory):
    unique = {}
    path = Path(directory)/"trials.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # An interrupted append may leave the final line incomplete.
            unique[record["candidate_id"]] = record
    return unique


def summarize(plan, output):
    """Only read saved converged results. Missing values never become zero SKR."""
    from src.scripts.gp_pauliAD import write_json
    rows, comparisons, profiles, diagnostics, counts = [], [], [], [], []
    for case in plan:
        # Include evaluated failures from interrupted/timed-out jobs too, even
        # if the job has not yet written its final optimization summary.
        unique = saved_trials(case["directory"])
        for record in unique.values():
            r = record["result"]
            if r["status"] != "converged":
                diagnostics.append(dict(name=case["name"], candidate_id=record["candidate_id"],
                                        status=r["status"], reason=r.get("convergence_reason", "evaluation"),
                                        error=r.get("error"), horizon=r.get("horizon")))
        path = Path(case["directory"])/"summary.json"
        if not path.exists():
            continue
        saved = json.loads(path.read_text())
        meta = {k: case[k] for k in ("name", "section", "nodes", "seed", "regime", "memory_profile",
                                    "memory_scale", "slot_seconds", "geometry", "source_state", "source_damping", "relaxation_model")}
        meta.update({k: case[k] for k in ("distance_km", "total_span_km", "channel_loss_db", "link_fidelity", "p_gen_zero_loss") if k in case})
        def row(record):
            if record is None:
                return dict(status="no_valid_result")
            c, r = record["candidate"], record["result"]
            result = dict(status=r["status"], q=c["q"], protocol=json.dumps(c["protocol"]),
                          policies=json.dumps(c["policies"]), distillation_rounds=sum(s.startswith("d") for s in c["protocol"]))
            if r["status"] == "converged":
                result.update({k: r[k] for k in ("skr", "skf", "mean_wait", "fidelity", "coverage", "horizon")})
                result.update(skr_bits_per_second=r["skr"]/case["slot_seconds"],
                              mean_wait_seconds=r["mean_wait"]*case["slot_seconds"])
            return result
        for stage, stage_result in saved["results"].items():
            rows.append(dict(**meta, stage=stage, certified=stage_result.get("certified", False), **row(stage_result["best"])))
        pairs = (("level", "uniform"), ("node", "level"), ("node", "uniform")) if case["study"] == "postselection" else (
            (("bell_policy", "bell_swaps"),) if case["section"] == "distillation" else (("joint", "bell_policy"),))
        for target, baseline in pairs:
            a, b = saved["results"][target]["best"], saved["results"][baseline]["best"]
            if a is None or b is None:
                continue
            ar, br = a["result"]["skr"], b["result"]["skr"]
            comparisons.append(dict(**meta, target=target, baseline=baseline, skr=ar, baseline_skr=br,
                                    absolute_gain=ar-br, relative_gain=ar/br-1 if br > 0 else None,
                                    activates_key=ar > 0 and br == 0, q=a["candidate"]["q"]))
        for reference, records in saved.get("q_profiles", {}).items():
            for record in records:
                profiles.append(dict(**meta, reference=reference, **row(record)))
            valid = [r for r in records if r["result"]["status"] == "converged"]
            bell = next((r for r in valid if r["candidate"]["q"] == .5), None)
            if valid and bell:
                best = max(valid, key=lambda r: r["result"]["skr"])
                ar, br = best["result"]["skr"], bell["result"]["skr"]
                comparisons.append(dict(**meta, target="matched_q_"+reference, baseline="same_protocol_q0.5",
                                        skr=ar, baseline_skr=br, absolute_gain=ar-br,
                                        relative_gain=ar/br-1 if br > 0 else None,
                                        activates_key=ar > 0 and br == 0, q=best["candidate"]["q"]))
        if case["section"] == "distillation":
            for rounds in sorted({sum(s.startswith("d") for s in r["candidate"]["protocol"]) for r in unique.values()}):
                valid = [r for r in unique.values() if r["result"]["status"] == "converged"
                         and sum(s.startswith("d") for s in r["candidate"]["protocol"]) == rounds]
                if valid:
                    counts.append(dict(**meta, **row(max(valid, key=lambda r: r["result"]["skr"]))))
    write_table(output/"summary.csv", rows)
    write_table(output/"comparisons.csv", comparisons)
    write_table(output/"q_profiles.csv", profiles)
    write_table(output/"distillation_counts.csv", counts)
    write_table(output/"unresolved.csv", diagnostics)
    write_json(output/"summary-data.json", rows)
    lines = ["Hardware follow-up: four report questions", f"Completed summaries: {len({r['name'] for r in rows})}/{len(plan)} jobs.",
             "SKR includes postselection retries. Rates saved in bits/slot AND bits/second.",
             "Memory scales multiply both local T1 and T2. Scale=1 is the stated reference.",
             "Mixed benchmark sources use visibility w0, not Bell fidelity w0.",
             "Best found is not certified optimal. Seed ranges measure optimizer variation.",
             "Missing/unconverged evaluations are excluded, not zero key.",
             f"Unresolved distinct candidate evaluations: {len(diagnostics)} (see unresolved.csv).",
             "q_profiles.csv and matched_q comparisons isolate source effects at fixed operations.", ""]
    for c in comparisons:
        gain = f"{100*c['relative_gain']:.4g}%" if c["relative_gain"] is not None else "undefined (zero baseline)"
        lines.append(f"{c['name']} {c['target']} / {c['baseline']}: {gain}; "
                     f"SKR={c['skr']:.8g} vs {c['baseline_skr']:.8g}; q={c['q']:.6g}; activates_key={c['activates_key']}")
    (output/"summary.txt").write_text("\n".join(lines)+"\n")
    if rows:
        try:
            plot(rows, comparisons, profiles, output, counts)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            write_json(output/"plot_status.json", dict(status="failed", error=str(exc)))
            print(f"WARNING: hardware tables saved; plotting failed: {exc}", flush=True)
        else:
            write_json(output/"plot_status.json", dict(status="complete"))


def plot(rows, comparisons, profiles, output, counts=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator
    import pandas as pd
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    table = pd.DataFrame(rows)
    data = table[table.status == "converged"]
    if data.empty:
        return
    colors = dict(uniform=".5", level="#0072B2", node="#D55E00",
                  bell_swaps=".5", bell_policy="#0072B2", joint="#D55E00")
    labels = dict(uniform="Uniform", level="Per level", node="Per swap",
                  bell_swaps="Swaps only", bell_policy="Best Bell protocol found", joint="Best q/protocol found")
    def memory_axis(ax, scales):
        scales = sorted(set(scales))
        if scales:
            ax.set_xscale("log")
            ax.set_xlim(float(min(scales))*.8, float(max(scales))*1.25)
            ax.set_xticks(scales, [f"{v:g}" for v in scales])
            ax.xaxis.set_minor_locator(NullLocator())
    def curves(ax, sub, stages):
        for stage in stages:
            frame = sub[sub.stage == stage]
            if frame.empty:
                continue
            grouped = frame.groupby("memory_scale").skr_bits_per_second.agg(["median", "min", "max"])
            ax.plot(grouped.index, grouped["median"], ".-", label=labels[stage], color=colors[stage])
            ax.fill_between(grouped.index, grouped["min"], grouped["max"], alpha=.12, color=colors[stage])
        ax.set(xscale="log", xlabel="Memory lifetime / reference lifetime", ylabel="SKR (bits/s)")
        memory_axis(ax, sub.memory_scale)
        ax.set_ylim(0, ax.get_ylim()[1]*1.08)
        if ax.lines:
            ax.legend(fontsize=7)
    def save(fig, stem):
        fig.savefig(output/(stem+".png"), dpi=170)
        fig.savefig(output/(stem+".pdf"))
        plt.close(fig)
    # Never connect points across different hardware, source states or geometry.
    for (regime, profile, model, nodes), group in data.groupby(["regime", "memory_profile", "relaxation_model", "nodes"]):
        suffix = f"{regime}-{profile}-{model}-N{nodes}"
        place = group[group.section == "placement"]
        if not place.empty:
            # Requested layout: Bell first row, PE second row; schedule + gain.
            fig, axes = plt.subplots(2, 2, figsize=(9, 5.8), constrained_layout=True)
            for i, q in enumerate((.5, .25)):
                sub = place[place.q == q]
                axes[i, 0].set_title(("Bell" if q == .5 else "PE, q=0.25")+": best level schedule")
                scales = sorted(sub.memory_scale.unique())
                heights = int(np.ceil(np.log2(nodes-1)))
                matrix = np.full((len(scales), heights), np.nan)
                for j, scale in enumerate(scales):
                    candidates = sub[(sub.memory_scale == scale) & (sub.stage == "level")]
                    if candidates.empty:
                        continue
                    # Show the highest-SKR seed; rate panels retain all seed ranges.
                    winner = candidates.loc[candidates.skr.idxmax()]
                    if winner.skr <= 0:
                        continue  # A zero-key policy is not scientifically preferred.
                    from src.optimization.pauli_ad import protocol_nodes
                    policies = json.loads(winner.policies)
                    for node in protocol_nodes(json.loads(winner.protocol), nodes-1):
                        matrix[j, node["height"]-1] = ("all_corrected", "psi_pair", "phi_pair").index(policies[node["boundary"]])
                if scales:
                    from matplotlib.colors import ListedColormap
                    axes[i, 0].imshow(np.ma.masked_invalid(matrix), aspect="auto", vmin=0, vmax=2,
                                      cmap=ListedColormap(["#0072B2", "#D55E00", "#009E73"]))
                    for j in range(len(scales)):
                        for k in range(heights):
                            value = matrix[j, k]
                            axes[i, 0].text(k, j, ("all", "Ψ", "Φ")[int(value)] if np.isfinite(value) else "—",
                                            ha="center", va="center", color="white" if np.isfinite(value) else ".4")
                    axes[i, 0].set(xticks=range(heights), xticklabels=range(1, heights+1),
                                   yticks=range(len(scales)), yticklabels=[f"{v:g}×" for v in scales],
                                   xlabel="Swapping level (first → last)", ylabel="Memory scale")
                pairs = sub.pivot(index=["seed", "memory_scale"], columns="stage", values="skr")
                for stage in ("level", "node"):
                    if stage not in pairs or "uniform" not in pairs:
                        continue
                    ratio = (pairs[stage]/pairs.uniform.where(pairs.uniform > 0)).reset_index(name="ratio")
                    grouped = ratio.groupby("memory_scale").ratio.agg(["median", "min", "max"])
                    axes[i, 1].plot(grouped.index, grouped["median"], ".-", color=colors[stage], label=labels[stage])
                    axes[i, 1].fill_between(grouped.index, grouped["min"], grouped["max"], color=colors[stage], alpha=.12)
                axes[i, 1].axhline(1, color=".5", linestyle=":")
                axes[i, 1].set(xscale="log", xlabel="Memory lifetime / reference lifetime",
                               ylabel="SKR / same-source uniform SKR", title="Retry cost included; zero baselines omitted")
                memory_axis(axes[i, 1], place.memory_scale)
                if axes[i, 1].get_legend_handles_labels()[0]:
                    axes[i, 1].legend(fontsize=7)
            save(fig, "1-placement-"+suffix)
            fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.9), constrained_layout=True)
            for ax, q in zip(axes, (.5, .25)):
                curves(ax, place[place.q == q], ("uniform", "level", "node"))
                ax.set_title("Bell" if q == .5 else "PE, q=0.25")
            save(fig, "1-absolute-rates-"+suffix)
        for section, stages in (("distillation", ("bell_swaps", "bell_policy")),
                                ("asymmetry", ("uniform", "level", "node")),
                                ("source", ("bell_policy", "joint"))):
            sub = group[group.section == section]
            keys = ["geometry"] if section == "asymmetry" else ["source_state"]
            panels = list(sub.groupby(keys[0]))
            if not panels:
                continue
            fig, axes = plt.subplots(1, len(panels), figsize=(4.3*len(panels), 2.9), squeeze=False, constrained_layout=True)
            for ax, (label, frame) in zip(axes[0], panels):
                curves(ax, frame, stages)
                ax.set_title(str(label))
                if section == "source":
                    winners = frame[(frame.stage == "joint") & (frame.skr > 0)].groupby("memory_scale").agg(
                        rate=("skr_bits_per_second", "median"), q=("q", "median"))
                    for scale, winner in winners.iterrows():
                        ax.annotate(f"q={winner.q:.3f}", (scale, winner.rate), xytext=(0, 5), textcoords="offset points", fontsize=7)
                elif section == "distillation":
                    winners = frame[(frame.stage == "bell_policy") & (frame.skr > 0)].groupby("memory_scale").agg(
                        rate=("skr_bits_per_second", "median"), rounds=("distillation_rounds", lambda v: "/".join(str(int(k)) for k in sorted(set(v)))))
                    for scale, winner in winners.iterrows():
                        ax.annotate(f"K={winner['rounds']}", (scale, winner.rate), xytext=(0, 5), textcoords="offset points", fontsize=7)
            save(fig, {"distillation": "2-distillation-", "asymmetry": "3-asymmetry-", "source": "4-source-"}[section]+suffix)
        if counts:
            k = pd.DataFrame(counts)
            k = k[(k.regime == regime) & (k.memory_profile == profile) & (k.relaxation_model == model) & (k.nodes == nodes)]
            panels = list(k.groupby("source_state"))
            if panels:
                fig, axes = plt.subplots(1, len(panels), figsize=(4.3*len(panels), 2.9), squeeze=False, constrained_layout=True)
                for ax, (state, frame) in zip(axes[0], panels):
                    for rounds, subset in frame.groupby("distillation_rounds"):
                        med = subset.groupby("memory_scale").skr_bits_per_second.median()
                        ax.plot(med.index, med.values, ".-", label=f"K={rounds}")
                    ax.set(xscale="log", xlabel="Memory lifetime / reference lifetime", ylabel="Best sampled SKR (bits/s)", title=state)
                    memory_axis(ax, frame.memory_scale)
                    ax.set_ylim(bottom=0)
                    ax.legend(fontsize=7)
                save(fig, "2-distillation-counts-"+suffix)
        if profiles:
            p = pd.DataFrame(profiles)
            p = p[(p.regime == regime) & (p.memory_profile == profile) & (p.relaxation_model == model)
                  & (p.nodes == nodes) & (p.status == "converged")]
            for state, frame in p.groupby("source_state"):
                fig, ax = plt.subplots(figsize=(5.8, 3), constrained_layout=True)
                for (scale, reference), subset in frame.groupby(["memory_scale", "reference"]):
                    median = subset.groupby("q").skr_bits_per_second.median()
                    ax.plot(median.index, median.values, ".-", label=f"{scale:g}×, {reference}")
                ax.set(xlabel="Source excitation weight q", ylabel="SKR (bits/s)", title="Matched protocols: "+state)
                ax.set_ylim(bottom=0)
                ax.legend(fontsize=6, ncol=2)
                save(fig, f"4-q-profiles-{suffix}-{state}")
