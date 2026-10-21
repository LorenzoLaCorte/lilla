"""Offline Plotly figures. Quality display masking never changes evaluation arrays."""
from hashlib import sha256

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.core.pauli_adc.state import coordinates_to_matrix
from src.visualization.metrics import DISPLAY_THRESHOLD, time_quantities
from src.visualization.protocols import locations, parse_tree


def run_label(run):
    return (f"{run.config['name']} · {run.config['prediction_model']} · "
            f"T={run.metadata['horizon']} · {run.metadata['run_id'][:8]}")


def color(run):
    hue = int(sha256(run.metadata["run_id"].encode()).hexdigest()[:8], 16) % 36000 / 100
    return f"hsl({hue},65%,40%)"


def times(run):
    scale = run.config["slot_duration_seconds"] or 1
    return np.arange(len(run.pmf))*scale


def time_figures(runs, log_pmf=False):
    distribution = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                 subplot_titles=("Completion-time PMF", "Covered CDF"))
    quality = make_subplots(rows=4, cols=1, shared_xaxes=True,
                           subplot_titles=("Phi+ fidelity", "eX", "eZ", "r(t): processing separated by completion time"))
    for run in runs:
        t, label, stroke = times(run), run_label(run), color(run)
        distribution.add_trace(go.Scatter(x=t[1:], y=run.pmf[1:], name=label,
                                         line_color=stroke), row=1, col=1)
        distribution.add_trace(go.Scatter(x=t[1:], y=run.pmf.cumsum()[1:], name=label,
                                         showlegend=False, line_color=stroke), row=2, col=1)
        q = time_quantities(run.x)
        for i, key in enumerate(("fidelity", "eX", "eZ", "secret_fraction"), 1):
            y = np.where(run.pmf > DISPLAY_THRESHOLD, q[key], np.nan)
            quality.add_trace(go.Scatter(x=t, y=y, name=label, line_color=stroke,
                                        showlegend=i == 1, connectgaps=False), row=i, col=1)
    distribution.update_yaxes(type="log" if log_pmf else "linear", row=1, col=1)
    unit = "seconds" if all(r.config["slot_duration_seconds"] for r in runs) else "slots"
    # Mixed units must be displayed in slots so an exploratory overlay is meaningful.
    if len({bool(r.config['slot_duration_seconds']) for r in runs}) > 1:
        return time_figures_in_slots(runs, log_pmf)
    distribution.update_xaxes(title_text=f"Completion time ({unit})", row=2, col=1)
    quality.update_xaxes(title_text=f"Completion time ({unit})", row=4, col=1)
    distribution.update_layout(height=550, hovermode="x unified", template="plotly_white")
    quality.update_layout(height=800, hovermode="x unified", template="plotly_white")
    return [distribution, quality]


def time_figures_in_slots(runs, log_pmf=False):
    from copy import deepcopy
    copies = deepcopy(runs)
    for r in copies:
        r.config["slot_duration_seconds"] = None
    return time_figures(copies, log_pmf)


def state_figure(run):
    labels = ("zL (ZI)", "zR (IZ)", "cZZ", "cXX", "cYY")
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, subplot_titles=labels)
    for i, (label, column) in enumerate(zip(labels, (2, 1, 3, 4, 5)), 1):
        y = np.where(run.pmf > DISPLAY_THRESHOLD, run.x[:, column], np.nan)
        fig.add_trace(go.Scatter(x=times(run), y=y, name=label, connectgaps=False), row=i, col=1)
    fig.update_layout(height=850, template="plotly_white", title=run_label(run))
    fig.update_xaxes(title_text="Completion time (seconds)" if run.config["slot_duration_seconds"]
                     else "Completion time (slots)", row=5, col=1)
    return fig


def matrix_figure(coordinates):
    matrix = coordinates_to_matrix(coordinates).real
    fig = go.Figure(go.Heatmap(z=matrix, x=["00", "01", "10", "11"],
                               y=["00", "01", "10", "11"], colorscale="RdBu",
                               zmid=0, text=matrix, texttemplate="%{text:.5f}"))
    fig.update_layout(title="Real density matrix in the computational basis", height=400)
    return fig


def tree_figure(tree, links, rounds=None):
    root = parse_tree(tree, links)
    fig = go.Figure()
    positions = {}

    def visit(n, depth=0):
        x = sum(n.segments)/2
        positions[f"{n.segments[0]}:{n.segments[1]}"] = (x, -depth)
        if n.left is not None:
            for child in (n.left, n.right):
                cx, cy = visit(child, depth+1)
                fig.add_trace(go.Scatter(x=[x, cx], y=[-depth, cy], mode="lines",
                                         line_color="#777", hoverinfo="skip", showlegend=False))
        return x, -depth

    visit(root)
    nodes = locations(root)
    labels = [f"{'Link ' if n.left is None else 'Span '}{span}" +
              (" · distill" if (rounds or {}).get(span) else "") for span, n in nodes.items()]
    fig.add_trace(go.Scatter(x=[positions[s][0] for s in nodes],
                             y=[positions[s][1] for s in nodes], text=labels,
                             mode="markers+text", textposition="top center", showlegend=False))
    fig.update_layout(height=300, template="plotly_white", margin=dict(t=30, b=20))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig
