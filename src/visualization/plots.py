"""Offline Plotly figures. Quality display masking never changes evaluation arrays."""
from hashlib import sha256

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.core.pauli_adc.state import coordinates_to_matrix
from src.visualization.metrics import DISPLAY_THRESHOLD, time_quantities
from src.visualization.protocols import locations, parse_tree


def main_figures(runs, log_pmf=False, histogram=False):
    """The two always-visible result plots; time axes are consistently in slots."""
    pmf = go.Figure()
    fidelity = go.Figure()
    for run in runs:
        t = np.arange(len(run.pmf))
        label, stroke = run_label(run), color(run)
        pmf.add_trace(go.Scatter(x=t[1:], y=run.pmf[1:], name=label,
                                line_color=stroke))
        f = time_quantities(run.x)["fidelity"]
        if histogram:
            # Conditional fidelity distribution, weighted by actual completion mass.
            mask = run.pmf > 0
            mass, edges = np.histogram(f[mask], bins=np.linspace(0, 1, 41),
                                      weights=run.pmf[mask])
            if run.pmf.sum():
                mass /= run.pmf.sum()
            fidelity.add_trace(go.Bar(x=(edges[1:]+edges[:-1])/2, y=mass,
                                      width=np.diff(edges), name=label, marker_color=stroke))
        else:
            fidelity.add_trace(go.Scatter(x=t, y=np.where(run.pmf > DISPLAY_THRESHOLD, f, np.nan),
                                         name=label, line_color=stroke, connectgaps=False))
    for fig in (pmf, fidelity):
        fig.update_layout(height=340, template="plotly_white", margin=dict(t=40, b=40),
                          showlegend=len(runs)>1, hovermode="closest")
    pmf.update_layout(title="Completion-time PMF")
    pmf.update_xaxes(title="Completion time (slots)")
    pmf.update_yaxes(title="Probability", type="log" if log_pmf else "linear")
    fidelity.update_layout(title="Φ+ fidelity distribution" if histogram else "Φ+ fidelity")
    fidelity.update_xaxes(title="Φ+ fidelity" if histogram else "Completion time (slots)")
    fidelity.update_yaxes(title="Conditional probability" if histogram else "Fidelity")
    return [pmf, fidelity]


def sweep_figures(experiment):
    """Four metrics on a fixed-tree 1D slice, 2D heatmap, or 3D scatter grid."""
    axes = [a["parameter"] for a in experiment.axes]
    figures = []
    from src.visualization.experiments import METRICS
    for label in ("SKR", "SKF", "Average waiting time", "Average Phi+ fidelity"):
        field = METRICS[label]
        entries = [e for e in experiment.entries if e.run is not None]
        values = [e.run.summary[field] for e in entries]
        custom = [e.index for e in entries]
        fig = go.Figure()
        if len(axes) == 1:
            fig.add_trace(go.Scatter(x=[e.point[axes[0]] for e in entries], y=values,
                                    customdata=custom, mode="lines+markers", name=label))
            fig.update_xaxes(title=axes[0], type="log" if experiment.axes[0].get("spacing") == "log" else "linear")
            fig.update_yaxes(title=label)
        elif len(axes) == 2:
            xs = sorted({e.point[axes[0]] for e in experiment.entries})
            ys = sorted({e.point[axes[1]] for e in experiment.entries})
            lookup = {(e.point[axes[0]], e.point[axes[1]]): v for e, v in zip(entries, values)}
            fig.add_trace(go.Heatmap(x=xs, y=ys, z=[[lookup.get((x, y)) for x in xs] for y in ys],
                                    colorbar_title=label))
            # Selectable overlay: Heatmap itself does not emit Plotly selection events.
            fig.add_trace(go.Scatter(x=[e.point[axes[0]] for e in entries],
                                    y=[e.point[axes[1]] for e in entries], customdata=custom,
                                    mode="markers", marker=dict(size=10, opacity=0.15, color="black"),
                                    text=[str(v) for v in values], hovertemplate="%{text}<extra></extra>"))
            fig.update_xaxes(title=axes[0])
            fig.update_yaxes(title=axes[1])
        else:
            valid = [(e, v) for e, v in zip(entries, values) if v is not None]
            fig.add_trace(go.Scatter3d(x=[e.point[axes[0]] for e, _ in valid],
                                      y=[e.point[axes[1]] for e, _ in valid],
                                      z=[e.point[axes[2]] for e, _ in valid],
                                      customdata=[e.index for e, _ in valid], mode="markers",
                                      marker=dict(size=5, color=[v for _, v in valid],
                                                  colorscale="Viridis", showscale=True, colorbar_title=label)))
            fig.update_layout(scene=dict(xaxis_title=axes[0], yaxis_title=axes[1], zaxis_title=axes[2]))
        fig.update_layout(title=label, height=340, template="plotly_white", showlegend=False)
        figures.append(fig)
    return figures


def search_figure(experiment):
    trials, scores, best = [], [], []
    incumbent = None
    for entry in experiment.entries:
        rate = entry.run.summary["rate_bits_per_slot"] if entry.eligible else None
        if rate is not None:
            incumbent = max(incumbent, rate) if incumbent is not None else rate
        trials.append(entry.index)
        scores.append(rate)
        best.append(incumbent)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trials, y=scores, mode="markers", name="Covered trial"))
    fig.add_trace(go.Scatter(x=trials, y=best, name="Best found"))
    fig.update_layout(height=300, template="plotly_white", xaxis_title="Trial", yaxis_title="SKR (bits/slot)")
    return fig


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
        x = n.segments[0]
        if n.left is not None:
            children = [visit(child, depth+1) for child in (n.left, n.right)]
            x = sum(cx for cx, _ in children)/2
            for cx, cy in children:
                fig.add_trace(go.Scatter(x=[x, cx], y=[-depth, cy], mode="lines",
                                         line=dict(color="#777", width=1.5), hoverinfo="skip", showlegend=False))
        positions[f"{n.segments[0]}:{n.segments[1]}"] = (x, -depth)
        return x, -depth

    visit(root)
    nodes = locations(root)
    counts = [(rounds or {}).get(span, 0) for span in nodes]
    labels = [f"{'Elementary link' if n.left is None else 'Link span'} {span}"
              f"<br>Distillation rounds: {count}" for (span, n), count in zip(nodes.items(), counts)]
    fig.add_trace(go.Scatter(x=[positions[s][0] for s in nodes],
                             y=[positions[s][1] for s in nodes], text=list(map(str, counts)),
                             customdata=list(nodes), hovertext=labels, hovertemplate="%{hovertext}<extra></extra>",
                             marker=dict(size=34, color="white", line=dict(color="#334155", width=1.5)),
                             textfont=dict(size=16, color="#0f172a"),
                             mode="markers+text", textposition="middle center", showlegend=False))
    depth = max(-y for _, y in positions.values())
    fig.update_layout(height=max(230, 80*(depth+1)), template="plotly_white", margin=dict(t=25, b=25, l=30, r=30))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, range=[-depth-.5, .5])
    return fig
