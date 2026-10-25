"""Local repeater explorer. Run with: streamlit run src/visualization/app.py."""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.visualization.config import MODELS, OUTCOMES, converted, expand_config, load_config
from src.visualization.evaluate import EvaluationService
from src.visualization.experiments import Experiment, PARAMETERS
from src.visualization.metrics import DISPLAY_THRESHOLD, bell_populations
from src.visualization.plots import (main_figures, matrix_figure, search_figure, state_figure,
                                     sweep_figures, time_figures)
from src.visualization.protocols import (balanced_tree,
                                         tree_slider_choices, tree_slider_position)
from src.visualization.tree_editor import protocol_tree
from src.visualization import storage

PRESETS = Path(__file__).with_name("presets")
STATE_MODES = {"Werner fidelity": "werner", "Bell weights": "bell_weights", "Real-X coordinates": "real_x"}
BELL_LABELS = {"all_corrected": "All corrected", "phi_pair": "Φ pair", "psi_pair": "Ψ pair", "phi_plus": "Φ+ only"}
SEARCH_LABELS = {"tree": "Tree", "cutoff": "Shared cutoff", "distillation": "Distillation placement",
                 "bell_outcomes": "Bell acceptance"}

st.set_page_config(page_title="Repeater explorer", layout="wide")
st.title("Repeater explorer")
ss = st.session_state
if "draft" not in ss:
    ss.draft = load_config((PRESETS/"default.json").read_text())
    ss.epoch = 0
if "service" not in ss:
    ss.service = EvaluationService()
if "last" not in ss:
    ss.last = None
if "job" not in ss:
    ss.job = None
if "experiment" not in ss:
    ss.experiment = None


def load_draft(config):
    if ss.job is not None:
        ss.job.cancelled = True
        storage.save_experiment(ss.job)
        ss.job = None
    ss.experiment = None
    ss.draft = deepcopy(config)
    ss.epoch += 1
    st.rerun()


def first(value, default):
    value = value[0] if isinstance(value, list) and value else value
    return default if value is None else value


def run_paths():
    return sorted(storage.DEFAULT_DIRECTORY.glob("run-*"), key=lambda p: p.stat().st_mtime, reverse=True)


def saved_label(path):
    try:
        return json.loads((Path(path)/"config.json").read_text())["name"] + " · " + Path(path).name[4:12]
    except (OSError, ValueError, KeyError):
        return Path(path).name


with st.expander("Load a scenario or configuration"):
    left, right = st.columns(2)
    with left:
        preset = st.selectbox("Scenario", sorted(PRESETS.glob("*.json")), format_func=lambda p: p.stem.replace("_", " "))
        if st.button("Load scenario"):
            load_draft(load_config(preset.read_text()))
        uploaded = st.file_uploader("Configuration JSON", type="json")
        if st.button("Load uploaded configuration", disabled=uploaded is None):
            try:
                load_draft(load_config(uploaded.getvalue().decode("utf-8")))
            except (ValueError, KeyError, TypeError) as error:
                st.error(str(error))
    with right:
        pasted = st.text_area("Paste configuration JSON")
        if st.button("Load pasted configuration"):
            try:
                load_draft(load_config(pasted))
            except (ValueError, KeyError, TypeError) as error:
                st.error(str(error))
        saved = st.selectbox("Saved run", run_paths(), format_func=saved_label, index=None, key="saved-load")
        if st.button("Load saved run", disabled=saved is None):
            try:
                run = storage.load_run(saved)
                ss.last = run
                ss.experiment = None
                load_draft(run.config)
            except (ValueError, OSError, KeyError) as error:
                st.error(str(error))

mode = st.radio("Mode", ("Single run", "Parameter sweep", "Bayesian optimization"), horizontal=True)
c = deepcopy(ss.draft)
key = lambda name: f"draft-{ss.epoch}-{name}"
state_label = st.selectbox("State mode", list(STATE_MODES),
                           index=list(STATE_MODES.values()).index(c["initial_state"]["kind"]), key=key("state-mode"))
state_mode = STATE_MODES[state_label]
links = int(st.number_input("Elementary links", min_value=1, max_value=8, value=c["links"], key=key("links")))
if links != c["links"]:
    c["links"] = links
    ss[key(f"tree-shape-{links}")] = 0.0
    c["protocol"].update(tree=balanced_tree(links), distillation_rounds={}, cutoff_overrides={})
    for field in ("p_gen",):
        if isinstance(c["hardware"][field], list):
            c["hardware"][field] = [first(c["hardware"][field], .02)]*links
    if isinstance(c["initial_state"].get("fidelity"), list):
        c["initial_state"]["fidelity"] = [first(c["initial_state"]["fidelity"], .98)]*links
    values = c["initial_state"].get("values")
    if values is not None and np.asarray(values).ndim == 2:
        c["initial_state"]["values"] = [values[0]]*links
    for field, value in c["memory"].items():
        if field != "kind" and isinstance(value, list) and (field != "pauli_mode_decay_rates" or np.asarray(value).ndim == 2):
            c["memory"][field] = [value[0]]*(links+1)

trees = tree_slider_choices(links)
middle = len(trees)//2
slider_key = key(f"tree-shape-{links}")
if slider_key not in ss:
    ss[slider_key] = tree_slider_position(c["protocol"]["tree"], links)
position = st.slider("Tree shape", -1.0, 1.0,
                     step=1.0/middle if middle else 1.0, format="%.3f" if middle>100 else "%.2f", disabled=not middle,
                     help="−1: fully left-unbalanced; 0: balanced; +1: fully right-unbalanced.",
                     key=slider_key)
choice = middle + round(position*middle)
tree = deepcopy(trees[choice])
if tree != c["protocol"]["tree"]:
    c["protocol"]["cutoff_overrides"] = {}
c["protocol"]["tree"] = tree
labels = st.columns(3)
labels[0].caption("Left-unbalanced")
labels[1].caption("Balanced")
labels[2].caption("Right-unbalanced")
st.caption("Edit a node’s number to set its distillation rounds.")
c["protocol"]["distillation_rounds"] = protocol_tree(
    tree, links, c["protocol"]["distillation_rounds"], key=key("protocol-tree"))

basic = st.container()
advanced = st.expander("Advanced settings")
errors = []
with advanced:
    c["name"] = st.text_input("Run name", c["name"], key=key("name"))
    c["prediction_model"] = st.selectbox("Prediction model", MODELS,
        index=MODELS.index(c["prediction_model"]), key=key("model"),
        format_func=lambda value: "Full real-X" if value == "full_real_x" else "Local Pauli channel twirl")
    memory_mode = st.selectbox("Memory controls", ("t1_t2", "pauli_adc"),
                              index=("t1_t2", "pauli_adc").index(c["memory"]["kind"]), key=key("memory-mode"))
    heterogeneous = st.toggle("Per-link / per-node settings",
        value=isinstance(c["hardware"]["p_gen"], list) or isinstance(c["memory"].get("t1_slots"), list)
        or np.asarray(c["initial_state"].get("values", [])).ndim == 2
        or isinstance(c["initial_state"].get("fidelity"), list)
        or np.asarray(c["memory"].get("pauli_mode_decay_rates", [])).ndim == 2,
        key=key("heterogeneous"))
    advanced_phi = st.checkbox("Allow Φ+ only acceptance", value=c["protocol"]["bell_outcomes"] == "phi_plus", key=key("phi-plus"))
    c["numerics"]["coverage_target"] = st.number_input("Coverage target", 0.0, 1.0,
                                float(c["numerics"]["coverage_target"]), format="%.6f", key=key("coverage"))
    seconds = st.checkbox("Set seconds per slot", value=c["slot_duration_seconds"] is not None, key=key("seconds-toggle"))
    c["slot_duration_seconds"] = st.number_input("Seconds per slot", min_value=1e-12,
                      value=float(c["slot_duration_seconds"] or 1), key=key("seconds")) if seconds else None
    override_text = st.text_area("Per-operation cutoff overrides", json.dumps(c["protocol"]["cutoff_overrides"]), key=key("overrides"))
    try:
        c["protocol"]["cutoff_overrides"] = json.loads(override_text)
    except ValueError as error:
        errors.append(f"Invalid cutoff overrides: {error}")
    link_table = st.container()
    node_table = st.container()
    raw = st.container()
    if mode == "Parameter sweep":
        max_evaluations = int(st.number_input("Maximum sweep evaluations", 1, 10000, 500))
    if mode == "Bayesian optimization":
        method = st.selectbox("Search method", ("bayesian", "enumeration"), format_func=str.title)
        budget = int(st.number_input("Trial budget", 1, 500, 20))
        initial_trials = int(st.number_input("Initial trials", 1, 500, 5))
        seed = int(st.number_input("Random seed", 0, 2**32-1, 0))
        cutoff_min = int(st.number_input("Minimum searched cutoff", 0, 10000, 0))
        cutoff_max = int(st.number_input("Maximum searched cutoff", 0, 10000, 20))
        include_no_cutoff = st.checkbox("Search No cutoff", value=True)
        distillation_max = int(st.number_input("Maximum searched rounds per node", 0, 8, 1))
        outcome_choices = st.multiselect("Searched Bell policies", OUTCOMES if advanced_phi else OUTCOMES[:3],
                                        default=list(OUTCOMES[:3]), format_func=lambda v: BELL_LABELS[v])

with basic:
    a, b, d = st.columns(3)
    with a:
        fidelity = st.number_input("Initial fidelity F0", 0.0, 1.0,
                   float(first(c["initial_state"].get("fidelity", .98), .98)),
                   disabled=state_mode != "werner" or heterogeneous, key=key("fidelity"))
    with b:
        c["hardware"]["p_swap"] = st.number_input("Swap hardware success", 0.0, 1.0, float(c["hardware"]["p_swap"]), key=key("p-swap"))
    with d:
        c["hardware"]["p_distillation"] = st.number_input("Distillation hardware success", 0.0, 1.0, float(c["hardware"]["p_distillation"]), key=key("p-dist"))
    a, b, d = st.columns(3)
    with a:
        pgen = st.number_input("Generation success", 0.0, 1.0, float(first(c["hardware"]["p_gen"], .02)),
                              disabled=heterogeneous, key=key("p-gen"))
    lifetimes = []
    for column, label, field, fallback in ((b, "T1", "t1_slots", 2000.), (d, "T2", "t2_slots", 1000.)):
        previous = c["memory"].get(field, fallback)
        with column:
            infinite = st.checkbox(f"Infinite {label}", value=previous is None, key=key(f"infinite-{label}"))
            lifetime = st.number_input(f"{label} (slots)", min_value=1e-9, value=float(first(previous, fallback)),
                                       disabled=infinite or heterogeneous or memory_mode != "t1_t2", key=key(label))
        lifetimes.append(None if infinite else lifetime)
    a, b, d = st.columns(3)
    with a:
        no_cutoff = st.checkbox("No cutoff", value=c["protocol"]["memory_cutoff_slots"] is None, key=key("no-cutoff"))
        cutoff = st.number_input("Shared memory cutoff (slots)", min_value=0,
                                  value=c["protocol"]["memory_cutoff_slots"] or 0, disabled=no_cutoff, key=key("cutoff"))
        c["protocol"]["memory_cutoff_slots"] = None if no_cutoff else int(cutoff)
    with b:
        c["numerics"]["t_trunc"] = int(st.number_input("Horizon", min_value=2, value=c["numerics"]["t_trunc"], key=key("horizon")))
    with d:
        outcomes = OUTCOMES if advanced_phi else OUTCOMES[:3]
        current = c["protocol"]["bell_outcomes"]
        c["protocol"]["bell_outcomes"] = st.selectbox("Global Bell-outcome acceptance", outcomes,
            index=outcomes.index(current) if current in outcomes else 0,
            format_func=lambda v: BELL_LABELS[v], key=key("outcomes"))

# Advanced tables override the corresponding shared controls explicitly.
if heterogeneous:
    with link_table:
        old_gen = c["hardware"]["p_gen"]
        data = {"Link": list(range(links)), "Generation success": old_gen if isinstance(old_gen, list) else [old_gen]*links}
        if state_mode == "werner":
            old_f = c["initial_state"].get("fidelity", .98)
            data["Fidelity"] = old_f if isinstance(old_f, list) else [old_f]*links
        table = st.data_editor(pd.DataFrame(data), disabled=["Link"], hide_index=True, key=key(f"links-{links}-{state_mode}"))
        c["hardware"]["p_gen"] = table["Generation success"].tolist()
        if state_mode == "werner":
            fidelity = table["Fidelity"].tolist()
else:
    c["hardware"]["p_gen"] = pgen
if state_mode == "werner":
    c["initial_state"] = {"kind": "werner", "fidelity": fidelity}
else:
    names = ["Φ+", "Ψ+", "Ψ−", "Φ−"] if state_mode == "bell_weights" else ["II", "IZ", "ZI", "ZZ", "XX", "YY"]
    defaults = [.98, .02/3, .02/3, .02/3] if state_mode == "bell_weights" else [1, 0, 0, 1, 1, -1]
    values = c["initial_state"].get("values", defaults)
    if np.asarray(values).shape[-1] != len(names):
        values = defaults
    if np.asarray(values).ndim == 1:
        values = [values]*(links if heterogeneous else 1)
    elif not heterogeneous:
        values = [values[0]]
    with (link_table if heterogeneous else basic):
        table = st.data_editor(pd.DataFrame(values, columns=names), hide_index=True, key=key(f"state-{links}-{state_mode}-{heterogeneous}"))
    c["initial_state"] = {"kind": state_mode, "values": table.to_numpy().tolist() if heterogeneous else table.iloc[0].tolist()}

if memory_mode == "t1_t2":
    if heterogeneous:
        with node_table:
            data = {"Node": list(range(links+1))}
            for field, value in zip(("t1_slots", "t2_slots"), lifetimes):
                old = c["memory"].get(field, value)
                data[field] = old if isinstance(old, list) else [old]*(links+1)
            table = st.data_editor(pd.DataFrame(data, dtype=object), disabled=["Node"], hide_index=True,
                column_config={name: st.column_config.NumberColumn() for name in ("t1_slots", "t2_slots")}, key=key(f"memories-{links}"))
            lifetimes = [[None if pd.isna(v) else float(v) for v in table[field]] for field in ("t1_slots", "t2_slots")]
            st.caption("Blank lifetimes mean infinity. Link i connects nodes i and i+1.")
    c["memory"] = {"kind": "t1_t2", "t1_slots": lifetimes[0], "t2_slots": lifetimes[1]}
else:
    with node_table:
        try:
            _, p, _ = expand_config(converted(c, model="full_real_x"))
            gamma, kappa = p["pauli_mode_decay_rates"], p["amplitude_damping_rate"]
        except (ValueError, TypeError, KeyError):
            gamma, kappa = [[.00075, .00075, 0]]*(links+1), [.0005]*(links+1)
        data = {"Node": list(range(links+1)) if heterogeneous else ["Shared"],
                "ΓX": [g[0] for g in gamma], "ΓY": [g[1] for g in gamma], "ΓZ": [g[2] for g in gamma], "κ": kappa}
        if not heterogeneous:
            data = {k: v[:1] for k, v in data.items()}
        table = st.data_editor(pd.DataFrame(data), disabled=["Node"], hide_index=True, key=key(f"rates-{links}-{heterogeneous}"))
        rates, damping = table[["ΓX", "ΓY", "ΓZ"]].to_numpy().tolist(), table["κ"].tolist()
        c["memory"] = {"kind": "pauli_adc", "pauli_mode_decay_rates": rates if heterogeneous else rates[0],
                       "amplitude_damping_rate": damping if heterogeneous else damping[0]}

request = None
if mode == "Parameter sweep":
    available = [name for name in PARAMETERS if (name != "Initial fidelity" or state_mode == "werner")
                 and (name not in ("T1", "T2") or memory_mode == "t1_t2")]
    selected = st.multiselect("Sweep parameters", available, default=["T1"] if "T1" in available else [available[0]], max_selections=3)
    bounds = {"Initial fidelity": (.9, 1.), "Swap hardware success": (.2, 1.), "Distillation hardware success": (.5, 1.),
              "Generation success": (.01, .2), "T1": (500., 4000.), "T2": (500., 2000.),
              "Shared memory cutoff": (0., 20.), "Horizon": (128., 2048.)}
    axes = []
    for name in selected:
        st.caption(name)
        columns = st.columns(4)
        with columns[0]:
            low = st.number_input("Minimum", value=bounds[name][0], key=f"sweep-min-{name}")
        with columns[1]:
            high = st.number_input("Maximum", value=bounds[name][1], key=f"sweep-max-{name}")
        with columns[2]:
            points = int(st.number_input("Points", 1, 1000, 15, key=f"sweep-points-{name}"))
        with columns[3]:
            spacing = st.selectbox("Spacing", ("linear", "log"), key=f"sweep-spacing-{name}")
        axes.append(dict(parameter=name, min=low, max=high, points=points, spacing=spacing))
    coupled = st.checkbox("Pure relaxation: T2=2T1", disabled="T1" not in selected or "T2" in selected)
    request = dict(mode="sweep", axes=axes, max_evaluations=max_evaluations,
                   pure_relaxation=coupled and "T1" in selected and "T2" not in selected)
elif mode == "Bayesian optimization":
    dims = st.multiselect("Search protocol controls", list(SEARCH_LABELS), default=["tree", "cutoff"],
                         format_func=lambda v: SEARCH_LABELS[v])
    request = dict(mode="optimize", method=method, dimensions=dims, budget=budget, initial=initial_trials,
                   seed=seed, cutoff_min=cutoff_min, cutoff_max=cutoff_max, include_no_cutoff=include_no_cutoff,
                   distillation_max=distillation_max, outcomes=outcome_choices)

ss.draft = deepcopy(c)
plan = None
try:
    _, expanded, operations = expand_config(c)
    if request:
        plan = Experiment(c, request)
        st.caption(f"Evaluation budget: {plan.planned} trials · no automatic horizon growth")
    with raw:
        st.dataframe(operations, hide_index=True, use_container_width=True)
        st.json({"memory_rates": {k: expanded[k] for k in ("amplitude_damping_rate", "pauli_mode_decay_rates")},
                 "submitted_configuration": c})
except (ValueError, KeyError, TypeError, OverflowError) as error:
    errors.append(str(error))
for error in errors:
    st.error(error)

label = {"Single run": "Run", "Parameter sweep": "Run sweep", "Bayesian optimization": "Optimize"}[mode]
if st.button(label, type="primary", disabled=bool(errors) or ss.job is not None):
    if plan is not None:
        ss.experiment = plan
        ss.job = plan
    else:
        try:
            with st.spinner("Evaluating…"):
                run = ss.service.evaluate(c)
                storage.save_run(run)
            ss.last = run
            ss.experiment = None
        except (ValueError, TypeError, KeyError, RuntimeError, OSError, ArithmeticError) as error:
            st.error(f"Evaluation failed: {error}")

job_area = st.container()
with job_area:
    if ss.job is not None:
        st.progress(len(ss.job.entries)/ss.job.planned, text=f"{len(ss.job.entries)}/{ss.job.planned} trials")
        if st.button("Cancel remaining trials"):
            ss.job.cancelled = True
            try:
                storage.save_experiment(ss.job)
            except OSError as error:
                st.error(f"Could not save history: {error}")
            ss.job = None

study_area = st.container()
with study_area:
    experiment = ss.experiment
    if experiment and experiment.entries:
        preferred = ss.pop("preferred_trial", None)
        if preferred and preferred[0] == experiment.id:
            ss[f"inspect-{experiment.id}"] = preferred[1]
        st.subheader("Sweep" if experiment.mode == "sweep" else "Search")
        if experiment.mode == "sweep":
            charts = sweep_figures(experiment)
            selector_key = f"inspect-{experiment.id}"
            for i, fig in enumerate(charts):
                # A new row every two metrics, preserving a compact 2x2 layout.
                if i % 2 == 0:
                    chart_columns = st.columns(2)
                with chart_columns[i % 2]:
                    if len(experiment.axes) < 3:
                        event = st.plotly_chart(fig, use_container_width=True, key=f"grid-{experiment.id}-{i}",
                                                on_select="rerun", selection_mode="points")
                        points = event.selection.points
                        signature = json.dumps(points, sort_keys=True)
                        event_key = f"selection-{experiment.id}-{i}"
                        if points and ss.get(event_key) != signature:
                            selected_index = points[0].get("customdata")
                            if isinstance(selected_index, list):
                                selected_index = selected_index[0]
                            if selected_index:
                                ss[selector_key] = int(selected_index)
                            ss[event_key] = signature
                    else:
                        st.plotly_chart(fig, use_container_width=True, key=f"grid-{experiment.id}-{i}")
        else:
            st.plotly_chart(search_figure(experiment), use_container_width=True)
            with st.expander("Ranked trials"):
                table = pd.DataFrame([e.row() for e in experiment.entries])
                st.dataframe(table.sort_values("SKR", ascending=False, na_position="last").head(10), hide_index=True)
        if experiment.best:
            st.caption(f"Best found: trial {experiment.best.index} · {experiment.best.run.summary['rate_bits_per_slot']:.6g} bits/slot")
        else:
            st.warning("No eligible best candidate: completed trials are failed, invalid, or under-covered.")
        choices = [e.index for e in experiment.entries if e.run is not None]
        if choices:
            selector_key = f"inspect-{experiment.id}"
            if ss.get(selector_key) not in choices:
                ss[selector_key] = experiment.best.index if experiment.best else choices[0]
            selected_index = st.selectbox("Inspect trial", choices, key=selector_key)
            ss.last = next(e.run for e in experiment.entries if e.index == selected_index)
        if experiment.cancelled:
            st.info("Cancelled; completed trials were retained.")

# Reserve positions so comparisons below can supply overlays to the main plots above.
main_result = st.container()
more = st.expander("More visualizations")
saved_area = st.expander("Saved runs / Compare")
run = ss.last
if run is not None:
    overlays = []
    with saved_area:
        paths = st.multiselect("Saved runs to overlay", run_paths(), format_func=saved_label,
                              max_selections=12, key="saved-overlays")
        for path in paths:
            try:
                overlays.append(storage.load_run(path))
            except (ValueError, OSError, KeyError) as error:
                st.warning(f"Cannot load {path.name}: {error}")
        if overlays:
            st.dataframe([{"Name": r.config["name"], "SKR": r.summary["rate_bits_per_slot"],
                           "SKF": r.summary["pooled_secret_fraction"], "Coverage": r.summary["coverage"],
                           "Horizon": r.metadata["horizon"]} for r in overlays], hide_index=True)
        st.download_button("Configuration JSON", storage.config_json(run.config), "config.json", "application/json")
        st.download_button("Per-time CSV", storage.per_time_csv(run), "per-time.csv", "text/csv")
        st.download_button("Arrays NPZ", storage.arrays_npz(run), "arrays.npz", "application/octet-stream")
        st.download_button("Summary CSV", storage.summary_csv([run, *overlays]), "summary.csv", "text/csv")
        st.caption(f"Saved under {storage.DEFAULT_DIRECTORY}")
        if experiment:
            st.download_button("Study history JSON", storage.config_json(experiment.history()), "history.json", "application/json")
            st.download_button("Study trials CSV", storage.csv_text([e.row() for e in experiment.entries]), "trials.csv", "text/csv")
    with main_result:
        st.subheader("Results")
        st.caption(f"{run.config['name']} · submitted horizon {run.metadata['horizon']} · {run.metadata['wall_seconds']:.3f}s")
        summary = run.summary
        cards = st.columns(5)
        for column, label, field in zip(cards,
            ("SKR (bits/slot)", "SKF", "Average waiting time", "Average Φ+ fidelity", "Covered probability"),
            ("rate_bits_per_slot", "pooled_secret_fraction", "mean_wait_repository_slots", "average_fidelity", "coverage")):
            value = summary[field]
            column.metric(label, "—" if value is None else (f"{value:.3%}" if field == "coverage" else f"{value:.6g}"))
        if summary["status"] == "no_completion":
            st.info("No completion inside this horizon; conditional state metrics are undefined.")
        elif run.metadata["diagnostics"]["provisional"]:
            st.warning("Provisional rate and waiting-time estimate: insufficient covered probability.")
        log_pmf = st.checkbox("Logarithmic PMF axis")
        histogram = st.selectbox("Fidelity view", ("Completion time", "Probability distribution")) == "Probability distribution"
        figures = main_figures([run, *[r for r in overlays if r.metadata["run_id"] != run.metadata["run_id"]]], log_pmf, histogram)
        for column, fig in zip(st.columns(2), figures):
            column.plotly_chart(fig, use_container_width=True)
        if experiment and experiment.mode == "optimize":
            with st.expander("Selected protocol", expanded=True):
                protocol_tree(run.config["protocol"]["tree"], run.config["links"],
                              run.config["protocol"]["distillation_rounds"], editable=False,
                              key=f"result-tree-{run.metadata['run_id']}")
        if st.button("Run at 2× horizon", disabled=ss.job is not None):
            try:
                with st.spinner("Checking longer horizon…"):
                    doubled = ss.service.evaluate(converted(run.config, double_horizon=True))
                    storage.save_run(doubled)
                ss.horizon_change = {"old_rate": summary["rate_bits_per_slot"], "new_rate": doubled.summary["rate_bits_per_slot"],
                                     "old_coverage": summary["coverage"], "new_coverage": doubled.summary["coverage"]}
                ss.last = doubled
                # Preserve study history while allowing the doubled run to remain selected.
                ss.experiment = None
                st.rerun()
            except (ValueError, RuntimeError, OSError) as error:
                st.error(str(error))
        if "horizon_change" in ss:
            st.caption(str(ss.horizon_change))
    with more:
        details = time_figures([run, *overlays]) + [state_figure(run)]
        for fig in details:
            st.plotly_chart(fig, use_container_width=True)
        populated = np.flatnonzero(run.pmf > DISPLAY_THRESHOLD).tolist()
        inspect = st.selectbox("Inspect state", ["Average", *populated])
        coordinates = summary["pooled_coordinates"] if inspect == "Average" else run.x[inspect]
        if coordinates is not None:
            st.plotly_chart(matrix_figure(coordinates), use_container_width=True)
            st.plotly_chart(go.Figure(go.Bar(x=["Φ+", "Ψ+", "Ψ−", "Φ−"], y=bell_populations(coordinates))), use_container_width=True)
            st.write(dict(zip(("II", "IZ", "ZI", "ZZ", "XX", "YY"), map(float, coordinates))))
        st.caption(f"Quality display threshold: PMF > {DISPLAY_THRESHOLD:g}. Metrics retain all positive mass.")
        st.json({"diagnostics": run.metadata["diagnostics"], "configuration": run.config,
                 "SKF from average state": summary["pooled_secret_fraction"],
                 "SKF with completion-time bins": summary["time_resolved_secret_fraction"]})
        if experiment:
            st.dataframe([e.row() for e in experiment.entries], hide_index=True)
    with saved_area:
        st.download_button("HTML figures", storage.figures_html(figures+details), "figures.html", "text/html")
elif ss.job is None:
    with main_result:
        st.info("Choose a protocol and run an experiment.")

# Yield between completed evaluations so Cancel is processed on the next rerun.
if ss.job is not None:
    try:
        with st.spinner("Evaluating next trial…"):
            entry = ss.job.step(ss.service)
            storage.save_experiment(ss.job)
        best = ss.job.best
        if best or entry.run is not None:
            ss.preferred_trial = (ss.job.id, best.index if best else entry.index)
        if entry.run is not None and ss.last is None:
            ss.last = entry.run
        if ss.job.finished:
            ss.job = None
        st.rerun()
    except (ValueError, OSError, RuntimeError) as error:
        ss.job.cancelled = True
        ss.job = None
        st.error(f"Experiment stopped; completed trials remain in this session: {error}")
