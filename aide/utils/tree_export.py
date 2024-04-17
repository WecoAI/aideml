"""Export journal to HTML visualization of tree + code."""

import json
import textwrap
from pathlib import Path

import numpy as np
from igraph import Graph
from ..journal import Journal


def get_edges(journal: Journal):
    for node in journal:
        for c in node.children:
            yield (node.step, c.step)


def generate_layout(n_nodes, edges, layout_type="rt"):
    """Generate visual layout of graph"""
    layout = Graph(
        n_nodes,
        edges=edges,
        directed=True,
    ).layout(layout_type)
    y_max = max(layout[k][1] for k in range(n_nodes))
    layout_coords = []
    for n in range(n_nodes):
        layout_coords.append((layout[n][0], 2 * y_max - layout[n][1]))
    return np.array(layout_coords)


def normalize_layout(layout: np.ndarray):
    """Normalize layout to [0, 1]"""
    layout = (layout - layout.min(axis=0)) / (layout.max(axis=0) - layout.min(axis=0))
    layout[:, 1] = 1 - layout[:, 1]
    layout[:, 1] = np.nan_to_num(layout[:, 1], nan=0)
    layout[:, 0] = np.nan_to_num(layout[:, 0], nan=0.5)
    return layout


def cfg_to_tree_struct(cfg, jou: Journal):
    edges = list(get_edges(jou))
    layout = normalize_layout(generate_layout(len(jou), edges))

    # metrics = np.array([n.metric.value_npsafe for n in jou])
    # metrics = (metrics - np.nanmin(metrics)) / (np.nanmax(metrics) - np.nanmin(metrics))
    # metrics = np.nan_to_num(metrics, nan=1)
    # metrics[:] = 0
    metrics = np.array([0 for n in jou])

    return dict(
        edges=edges,
        layout=layout.tolist(),
        plan=[textwrap.fill(n.plan, width=80) for n in jou.nodes],
        code=[n.code for n in jou],
        term_out=[n.term_out for n in jou],
        analysis=[n.analysis for n in jou],
        exp_name=cfg.exp_name,
        metrics=metrics.tolist(),
    )


def generate_html(tree_graph_str: str):
    template_dir = Path(__file__).parent / "viz_templates"
    
    with open(template_dir / "template.js") as f:
        js = f.read()
        js = js.replace("<placeholder>", tree_graph_str)

    with open(template_dir / "template.html") as f:
        html = f.read()
        html = html.replace("<!-- placeholder -->", js)

        return html


def generate(cfg, jou: Journal, out_path: Path):
    tree_graph_str = json.dumps(cfg_to_tree_struct(cfg, jou))
    html = generate_html(tree_graph_str)
    with open(out_path, "w") as f:
        f.write(html)
