"""Run from the repository root with python -m streamlit run src/visualization/app.py."""
from copy import deepcopy
import json
from pathlib import Path
import sys

# Streamlit executes this file as a script, so make repository imports explicit.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import streamlit as st

from src.core.pauli_adc.state import coordinates_to_matrix
from src.visualization.comparisons import (candidate_key, comparison_plan, comparison_rows,
                                          parameter_differences, scenario_key, selection_loss)
from src.visualization.config import (DEFAULT, MODELS, OUTCOMES, converted, expand_config,
                                      canonical, load_config)
from src.visualization.evaluate import EvaluationService, source_fingerprint
from src.visualization.metrics import DISPLAY_THRESHOLD, bell_populations
from src.visualization.plots import (matrix_figure, run_label, state_figure, time_figures,
                                     tree_figure)
from src.visualization.protocols import locations, ordered_trees, parse_tree
from src.visualization.storage import (DEFAULT_DIRECTORY, config_json, figures_html, load_run,
                                       per_time_csv, save_run, summary_csv)

PRESETS = Path(__file__).with_name("presets")
st.set_page_config(page_title="Pauli–ADC explorer", layout="wide")
st.title("Local Pauli–ADC explorer")
st.caption("One fixed recursive protocol per evaluation. AD then Pauli memory noise, immediate corrections, destructive full-reset retries. Times are generation-attempt slots.")

if "service" not in st.session_state:
    st.session_state.service = EvaluationService()
    st.session_state.draft = deepcopy(DEFAULT)
    st.session_state.epoch = 0
    st.session_state.last = None
    st.session_state.recent_comparison = []


def set_draft(config):
    expand_config(config)
    st.session_state.draft = deepcopy(config)
    st.session_state.epoch += 1
    st.rerun()


def evaluate_and_save(config):
    with st.spinner("Evaluating the explicit protocol…"):
        run = st.session_state.service.evaluate(config)
        destination = save_run(run)
    st.session_state.last = run
    st.success(f"Saved {destination.name}; {run.metadata['wall_seconds']:.3f} s"
               + (" (memory cache)" if run.metadata["cache_hit"] else ""))
    return run


def json_input(label, value, key):
    return st.text_area(label, config_json(value), key=key, height=80)


def parse_json(text):
    return json.loads(text, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))


def node_values(value, count):
    return value if isinstance(value, list) else [value]*count


def cleaned_column(frame, name):
    return [None if pd.isna(v) else float(v) for v in frame[name]]


with st.expander("Load a scenario or configuration", expanded=False):
    preset = st.selectbox("Scenario preset", sorted(PRESETS.glob("*.json")),
                          format_func=lambda p: p.stem.replace("_", " "))
    if st.button("Load preset"):
        set_draft(load_config(preset.read_text()))
    uploaded = st.file_uploader("Import config JSON", type=["json"])
    if st.button("Import configuration", disabled=uploaded is None):
        try:
            set_draft(load_config(uploaded.getvalue().decode("utf-8")))
        except (ValueError, KeyError, TypeError) as error:
            st.error(str(error))
    raw_config = st.text_area("Or paste a complete version-1 config", "", height=100)
    if st.button("Load pasted configuration"):
        try:
            set_draft(load_config(raw_config))
        except (ValueError, KeyError, TypeError) as error:
            st.error(str(error))

c = deepcopy(st.session_state.draft)
epoch = st.session_state.epoch
key = lambda name: f"draft-{epoch}-{name}"
# Structural controls rerender available tree locations; they never evaluate.
s = int(st.number_input("Elementary links (physical nodes = links + 1)", min_value=1,
                        max_value=8, value=c["links"], key=key("links")))
trees = ordered_trees(s)
index = trees.index(c["protocol"]["tree"]) if c["protocol"]["tree"] in trees else 0
tree = st.selectbox("Ordered swap tree", trees, index=index,
                    format_func=lambda v: json.dumps(v), key=key(f"tree-{s}"))
heterogeneous = st.toggle("Per-link / per-node tables", value=isinstance(c["hardware"]["p_gen"], list),
                          key=key("heterogeneous"))
initial_mode = st.selectbox("Initial-state mode", ["werner", "bell_weights", "real_x"],
                           index=["werner", "bell_weights", "real_x"].index(c["initial_state"]["kind"]),
                           key=key("initial-mode"))
memory_mode = st.selectbox("Memory controls", ["t1_t2", "pauli_adc"],
                          index=["t1_t2", "pauli_adc"].index(c["memory"]["kind"]), key=key("memory-mode"))
variant = st.selectbox("Convenient protocol variant", ["Custom", "Root distillation", "Cutoff 10",
                      "Root distillation + cutoff 10", "Phi pair", "Psi pair"], key=key("variant"))
st.caption("Structural controls refresh the form. Use Preview configuration to inspect it; Evaluate runs the simulation.")

with st.form(key("evaluation")):
    name = st.text_input("Run name", c["name"])
    slot_text = st.text_input("Slot duration in seconds (optional JSON number; null = slots only)",
                              json.dumps(c["slot_duration_seconds"]))
    p_swap = st.number_input("Swap hardware success", 0.0, 1.0, float(c["hardware"]["p_swap"]))
    p_dist = st.number_input("Distillation hardware success", 0.0, 1.0,
                            float(c["hardware"]["p_distillation"]))
    if heterogeneous:
        _, original, _ = expand_config(converted(c, model="full_real_x"))
        pgen = original["p_gen"] if s == c["links"] else [0.2]*s
        fidelities = node_values(c["initial_state"].get("fidelity", 0.98), s)
        if len(fidelities) != s:
            fidelities = [0.98]*s
        link_data = {"link": list(range(s)), "p_gen": pgen}
        if initial_mode == "werner":
            link_data["fidelity"] = fidelities
        link_frame = st.data_editor(pd.DataFrame(link_data), disabled=["link"],
                                    hide_index=True, key=key(f"link-table-{s}-{initial_mode}"))
        st.caption("Link i connects physical nodes i and i+1. Each node's rate is shared by its memories.")
    else:
        pgen_text = json_input("Generation success (scalar or per-link JSON list)", c["hardware"]["p_gen"], key("pgen"))
        if initial_mode == "werner":
            fidelity_text = json_input("Werner fidelity F0 (scalar or per-link list)",
                                       c["initial_state"].get("fidelity", 0.98), key("fidelity"))
    if initial_mode != "werner":
        initial_values = c["initial_state"].get("values", [0.98, 0.02/3, 0.02/3, 0.02/3]
                         if initial_mode == "bell_weights" else [1, 0, 0, 1, 1, -1])
        if np.asarray(initial_values).shape[-1] != (4 if initial_mode == "bell_weights" else 6):
            initial_values = [1, 0, 0, 0] if initial_mode == "bell_weights" else [1, 0, 0, 1, 1, -1]
        state_text = json_input("Bell order: phi+, psi+, psi-, phi-; real-X order: II, IZ, ZI, ZZ, XX, YY",
                                initial_values, key("state-values"))
    if memory_mode == "t1_t2":
        t1, t2 = c["memory"].get("t1_slots", 2000), c["memory"].get("t2_slots", 1000)
        if heterogeneous:
            t1s, t2s = node_values(t1, s+1), node_values(t2, s+1)
            if len(t1s) != s+1 or len(t2s) != s+1:
                t1s, t2s = [2000.0]*(s+1), [1000.0]*(s+1)
            node_frame = st.data_editor(pd.DataFrame({"physical_node": range(s+1),
                            "T1_slots": t1s, "T2_slots": t2s}, dtype=object),
                            column_config={"T1_slots": st.column_config.NumberColumn(),
                                           "T2_slots": st.column_config.NumberColumn()},
                            disabled=["physical_node"], hide_index=True, key=key(f"node-table-{s}"))
            st.caption("Blank lifetime cells mean infinite lifetime (JSON null). T2 must be at most 2*T1.")
        else:
            t1_text = json_input("T1 in slots (null = infinite)", t1, key("t1"))
            t2_text = json_input("T2 in slots (null = infinite)", t2, key("t2"))
    else:
        _, physical_seed, _ = expand_config(converted(c, model="full_real_x"))
        if heterogeneous:
            gamma_seed = physical_seed["pauli_mode_decay_rates"] if s == c["links"] else [[0.00075, 0.00075, 0]]*(s+1)
            kappa_seed = physical_seed["amplitude_damping_rate"] if s == c["links"] else [0.0005]*(s+1)
            rate_frame = st.data_editor(pd.DataFrame({"physical_node": range(s+1),
                           "GammaX": [g[0] for g in gamma_seed], "GammaY": [g[1] for g in gamma_seed],
                           "GammaZ": [g[2] for g in gamma_seed], "kappa": kappa_seed}),
                           disabled=["physical_node"], hide_index=True, key=key(f"rate-table-{s}"))
            st.caption("All rates are per slot. Pauli MODE decay rates obey the triangle inequalities.")
        else:
            gamma_text = json_input("Pauli MODE decay rates: shared [GammaX,GammaY,GammaZ] or per-node rows",
                         c["memory"].get("pauli_mode_decay_rates", physical_seed["pauli_mode_decay_rates"]), key("gamma"))
            kappa_text = json_input("Amplitude damping rate: scalar or per-node list",
                         c["memory"].get("amplitude_damping_rate", physical_seed["amplitude_damping_rate"]), key("kappa"))
    spans = list(locations(parse_tree(tree, s)))
    default_dists = [span for span, count in c["protocol"]["distillation_rounds"].items()
                     if count and span in spans]
    dists = st.multiselect("One recurrence round at these link spans", spans, default=default_dists,
                           help="Two independent copies of the selected subtree are prepared.")
    cutoff_text = st.text_input("Shared memory cutoff in slots (null = No cutoff)",
                               json.dumps(c["protocol"]["memory_cutoff_slots"]))
    override_text = json_input("Optional overrides by operation ID (e.g. swap:0:3 or dist:0:3)",
                               c["protocol"]["cutoff_overrides"], key("overrides"))
    outcomes = st.selectbox("Global Bell-outcome acceptance", OUTCOMES,
                           index=OUTCOMES.index(c["protocol"]["bell_outcomes"]))
    model = st.selectbox("Prediction model", MODELS, index=MODELS.index(c["prediction_model"]),
                        format_func=lambda m: "Full real-X" if m == MODELS[0] else "Local Pauli channel twirl")
    horizon = int(st.number_input("Horizon T (bins 0,...,T-1)", min_value=2,
                                  value=c["numerics"]["t_trunc"], step=1))
    coverage_target = st.number_input("Coverage target", 0.0, 1.0,
                                      float(c["numerics"]["coverage_target"]), format="%.6f")
    preview = st.form_submit_button("Preview configuration")
    submitted = st.form_submit_button("Evaluate", type="primary")

try:
    draft = deepcopy(c)
    draft.update(name=name, links=s, slot_duration_seconds=parse_json(slot_text), prediction_model=model,
                 hardware={"p_gen": cleaned_column(link_frame, "p_gen") if heterogeneous else parse_json(pgen_text),
                           "p_swap": p_swap, "p_distillation": p_dist},
                 numerics={"t_trunc": horizon, "coverage_target": coverage_target})
    draft["initial_state"] = ({"kind": "werner", "fidelity": cleaned_column(link_frame, "fidelity")
                               if heterogeneous else parse_json(fidelity_text)} if initial_mode == "werner"
                              else {"kind": initial_mode, "values": parse_json(state_text)})
    draft["memory"] = ({"kind": "t1_t2", "t1_slots": cleaned_column(node_frame, "T1_slots")
                        if heterogeneous else parse_json(t1_text), "t2_slots": cleaned_column(node_frame, "T2_slots")
                        if heterogeneous else parse_json(t2_text)} if memory_mode == "t1_t2" else
                       {"kind": "pauli_adc", "pauli_mode_decay_rates": rate_frame[["GammaX", "GammaY", "GammaZ"]].values.tolist()
                        if heterogeneous else parse_json(gamma_text),
                        "amplitude_damping_rate": cleaned_column(rate_frame, "kappa")
                        if heterogeneous else parse_json(kappa_text)})
    protocol = {"tree": tree, "distillation_rounds": {span: 1 for span in dists},
                "memory_cutoff_slots": parse_json(cutoff_text), "cutoff_overrides": parse_json(override_text),
                "bell_outcomes": outcomes}
    if "Root distillation" in variant:
        protocol["distillation_rounds"][f"0:{s-1}"] = 1
    if "cutoff 10" in variant.lower():
        protocol["memory_cutoff_slots"] = 10
    if variant in ("Phi pair", "Psi pair"):
        protocol["bell_outcomes"] = "phi_pair" if variant == "Phi pair" else "psi_pair"
    draft["protocol"] = protocol
    _, expanded, operations = expand_config(draft)
    st.plotly_chart(tree_figure(tree, s, protocol["distillation_rounds"]), use_container_width=True, key="draft-tree")
    st.dataframe(pd.DataFrame(operations), hide_index=True, use_container_width=True)
    st.caption("Cutoffs accept |t1−t2| ≤ cutoff, rather than bounding total duration. No cutoff compiles to the horizon.")
    with st.expander("Derived physical node rates and submitted backend parameters"):
        _, physical, _ = expand_config(converted(draft, model="full_real_x"))
        rate_rows = [{"node": i, "GammaX_per_slot": g[0], "GammaY_per_slot": g[1],
                      "GammaZ_per_slot": g[2], "kappa_per_slot": k}
                     for i, (g, k) in enumerate(zip(physical["pauli_mode_decay_rates"], physical["amplitude_damping_rate"]))]
        if draft["slot_duration_seconds"]:
            for row in rate_rows:
                row.update({label.replace("per_slot", "per_second"): value/draft["slot_duration_seconds"]
                            for label, value in list(row.items()) if "per_slot" in label})
        st.dataframe(pd.DataFrame(rate_rows), hide_index=True)
        st.json(expanded)
    st.download_button("Export draft config JSON", config_json(draft), "config.json", "application/json")
    if preview or submitted:
        st.session_state.draft = deepcopy(draft)
    if submitted:
        evaluate_and_save(draft)
except Exception as error:
    st.error(f"Configuration/evaluation failed: {error}. The last successful result is retained.")

paths = sorted(DEFAULT_DIRECTORY.glob("run-*/run.json"), key=lambda p: p.stat().st_mtime, reverse=True)
labels = {}
for path in paths:
    try:
        data = json.loads(path.read_text())
        labels[str(path.parent)] = (f"{data['config']['name']} · {data['config']['prediction_model']} · "
                                   f"T={data['horizon']} · {path.parent.name}")
    except (ValueError, KeyError, OSError):
        continue
with st.expander("Reload a saved result"):
    reload_path = st.selectbox("Local run bundle", list(labels), format_func=labels.get) if labels else None
    if st.button("Reload result", disabled=reload_path is None):
        try:
            st.session_state.last = load_run(reload_path)
        except Exception as error:
            st.error(f"Reload failed: {error}")

run = st.session_state.last
if run is not None:
    st.subheader(run_label(run))
    st.caption("Results below use this run's saved submitted configuration.")
    if run.metadata["source_fingerprint"] != source_fingerprint():
        st.info("Historical result: source fingerprint differs from current code. Explicit re-evaluation uses current code.")
    if run.summary["status"] == "no_completion":
        st.warning("No completion inside this horizon. Rate and coverage are zero; conditional state quantities are undefined.")
    elif run.metadata["diagnostics"]["provisional"]:
        st.warning("Provisional rate and mean: probability coverage is below target. Missing mass is unresolved late completion.")
    summary = run.summary
    rate_field = "rate_bits_per_second" if run.config["slot_duration_seconds"] else "rate_bits_per_slot"
    columns = st.columns(4)
    columns[0].metric("Pooled key rate (bits/second)" if run.config["slot_duration_seconds"] else "Pooled key rate (bits/slot)",
                       f"{summary[rate_field]:.8g}")
    columns[1].metric("Pooled secret fraction", "undefined" if summary["pooled_secret_fraction"] is None
                       else f"{summary['pooled_secret_fraction']:.8g}")
    columns[2].metric("Covered probability", f"{summary['coverage']:.12g}")
    columns[3].metric("Average Phi+ fidelity", "undefined" if summary["average_fidelity"] is None
                       else f"{summary['average_fidelity']:.8g}")
    st.write({"Repository finite-horizon mean (slots)": summary["mean_wait_repository_slots"],
              "Observed mean, conditional on completion in horizon (slots)": summary["mean_wait_observed_slots"],
              "Repository mean (seconds)": summary["mean_wait_repository_seconds"],
              "Horizon": run.metadata["horizon"], "Wall time (s)": run.metadata["wall_seconds"]})
    st.caption("Neither mean certifies the infinite-horizon mean. High coverage alone does not certify a close ranking. Headline BB84 objective is pooled; no extra sifting factor.")
    if st.button("Re-evaluate at 2× horizon"):
        old = run
        try:
            run = evaluate_and_save(converted(run.config, double_horizon=True))
            st.write({"coverage_change": run.summary["coverage"]-old.summary["coverage"],
                      "rate_change_bits_per_slot": run.summary["rate_bits_per_slot"]-old.summary["rate_bits_per_slot"]})
            st.session_state.horizon_change = (old.metadata["run_id"], run.metadata["run_id"],
                run.summary["coverage"]-old.summary["coverage"],
                run.summary["rate_bits_per_slot"]-old.summary["rate_bits_per_slot"])
            st.rerun()
        except Exception as error:
            st.error(f"Re-evaluation failed: {error}")
    if "horizon_change" in st.session_state:
        st.write({"Last horizon comparison (old ID, new ID, coverage change, rate change bits/slot)":
                  st.session_state.horizon_change})
    if st.button("Re-evaluate in full model"):
        try:
            evaluate_and_save(converted(run.config, model="full_real_x"))
            st.rerun()
        except Exception as error:
            st.error(f"Re-evaluation failed: {error}")
    st.download_button("Export submitted config JSON", config_json(run.config), "submitted-config.json", "application/json")
    st.download_button("Download per-time CSV", per_time_csv(run), "per-time.csv", "text/csv")
    with st.expander("Run diagnostics and provenance"):
        st.json(run.metadata)

time_tab, state_tab, compare_tab = st.tabs(["Time", "State", "Compare"])
with compare_tab:
    retained_paths = [p for p in st.session_state.get("saved_selection", []) if p in labels]
    selected_paths = st.multiselect("Saved runs to overlay / compare (up to 12)", list(labels),
                                    format_func=labels.get, max_selections=12,
                                    default=retained_paths, key="compare-paths")
    st.session_state.saved_selection = selected_paths
    selected = []
    for path in selected_paths:
        try:
            selected.append(load_run(path))
        except Exception as error:
            st.error(f"Cannot load {path}: {error}")
    if selected:
        st.dataframe(pd.DataFrame(comparison_rows(selected)), hide_index=True, use_container_width=True)
        st.dataframe(pd.DataFrame(parameter_differences(selected)), hide_index=True, use_container_width=True)
        st.download_button("Download summary CSV", summary_csv(selected), "summary.csv", "text/csv")
        try:
            plan = comparison_plan([r.config for r in selected])
            plan_signature = canonical({"scenario": scenario_key(plan[0]),
                                        "horizon": plan[0]["numerics"]["t_trunc"],
                                        "candidates": sorted({candidate_key(c) for c in plan}),
                                        "source": source_fingerprint()})
            pending = sum(not st.session_state.service.is_cached(c) for c in plan)
            st.write(f"{len(plan)//2} distinct manual candidates; {pending} pending backend calls, {len(plan)} result rows.")
            if st.button("Evaluate candidates in both models"):
                completed = []
                progress = st.progress(0.0)
                for i, config in enumerate(plan):
                    try:
                        completed.append(evaluate_and_save(config))
                    except Exception as error:
                        st.error(f"Candidate {config['name']} / {config['prediction_model']} failed: {error}")
                    progress.progress((i+1)/len(plan))
                st.session_state.recent_comparison = completed
                st.session_state.expected_candidates = {candidate_key(c) for c in plan}
                st.session_state.comparison_signature = plan_signature
            completed = st.session_state.recent_comparison
            same_selection = st.session_state.get("comparison_signature") == plan_signature
            if completed and same_selection:
                st.dataframe(pd.DataFrame(comparison_rows(completed)), hide_index=True)
                try:
                    loss = selection_loss(completed, st.session_state.expected_candidates)
                    st.write(f"Selection loss within these {loss['candidates']} candidates")
                    st.json(loss)
                    st.caption("Near-tied winners use the stated ranking tolerance; ranges retain all winners. Relative loss is undefined when the best full-model rate is zero.")
                except ValueError as error:
                    st.warning(str(error))
        except ValueError as error:
            st.info(f"Exploratory overlay only: {error}")
    else:
        st.caption("Evaluate or reload runs, then select saved protocols here. Selection loss requires one physical scenario and horizon.")

with time_tab:
    if run is not None:
        log_pmf = st.checkbox("Logarithmic PMF axis")
        overlay = [run] + [r for r in selected if r.metadata["run_id"] != run.metadata["run_id"]]
        st.caption(f"Quality curves mask pmf ≤ {DISPLAY_THRESHOLD:g} for display only. All positive bins enter metrics and exports. Mixed unit overlays use slots.")
        figures = time_figures(overlay, log_pmf)
        for i, figure in enumerate(figures):
            st.plotly_chart(figure, use_container_width=True, key=f"time-{i}")
        st.download_button("Download self-contained HTML figures", figures_html(figures+[state_figure(run),
                           tree_figure(run.config["protocol"]["tree"], run.config["links"],
                                       run.config["protocol"]["distillation_rounds"])]),
                           "figures.html", "text/html")
    else:
        st.info("Submit Evaluate to create the first result.")

with state_tab:
    if run is not None:
        st.plotly_chart(state_figure(run), use_container_width=True, key="state-curves")
        st.write({"Pooled secret fraction": run.summary["pooled_secret_fraction"],
                  "Ideal asymptotic processing separated by completion-time bins": run.summary["time_resolved_secret_fraction"]})
        st.caption("Completion-time separation is a diagnostic of this result; it supplies no syndrome resolution, postselection protocol, or finite-key claim.")
        if run.summary["pooled_coordinates"] is not None:
            choice = st.selectbox("Inspect state", ["Pooled"]+np.flatnonzero(run.pmf > 0).tolist(),
                                  format_func=lambda v: v if v == "Pooled" else f"Completion at slot {v} (pmf={run.pmf[v]:.3g})")
            x = np.asarray(run.summary["pooled_coordinates"]) if choice == "Pooled" else run.x[choice]
            st.write(dict(zip(("II", "IZ=zR", "ZI=zL", "ZZ", "XX", "YY"), x.tolist())))
            matrix = coordinates_to_matrix(x)
            st.write({"Trace": float(np.trace(matrix).real), "Minimum eigenvalue": float(np.linalg.eigvalsh(matrix).min())})
            st.plotly_chart(matrix_figure(x), use_container_width=True, key="matrix")
            st.write({"Bell populations (phi+, psi+, psi-, phi-)": bell_populations(x).tolist()})
            st.caption("Bell populations describe overlaps, not the full state when local biases are present. The pooled matrix is rho(x̄), weighted by completion mass.")
