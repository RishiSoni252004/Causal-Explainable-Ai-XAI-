"""
causal_analysis.py — Causal graph modeling, NetworkX & Plotly visualization,
and educational causal reasoning for the Causal XAI project.

Provides:
    • Directed causal graph (NetworkX DiGraph) modeling.
    • Interactive visualization using both NetworkX + Plotly and Matplotlib.
    • Dynamic graph synthesis for custom uploaded datasets (hypothesized DAGs).
    • Interactive graph mutation: adding/removing nodes, directed edges, and editing relationships.
    • Session persistence and serialization to/from JSON.
    • Relationship inspection and explanation with observed correlation context.
    • Rigorous scientific distinction between observational associations and causal assumptions.
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go

__all__ = [
    "build_causal_graph",
    "plot_causal_graph",
    "get_edge_descriptions",
    "generate_initial_causal_graph",
    "graph_to_dict",
    "dict_to_graph",
    "add_node_to_graph",
    "remove_node_from_graph",
    "add_edge_to_graph",
    "remove_edge_from_graph",
    "update_edge_in_graph",
    "get_graph_edge_descriptions",
    "compute_dag_layout",
    "build_causal_plotly_figure",
    "CORRELATION_VS_CAUSATION",
    "TRUSTWORTHY_AI_CONTENT",
    "SCIENTIFIC_DISCLAIMER_CONTENT",
]


# ---------------------------------------------------------------------------
# Default Demo Causal Graph (Student Performance)
# ---------------------------------------------------------------------------

CAUSAL_EDGES = [
    # Direct causes of the outcome
    ("study_hours",           "result"),
    ("attendance",            "result"),
    ("previous_scores",       "result"),
    ("assignments_completed", "result"),
    ("class_participation",   "result"),
    ("sleep_hours",           "result"),

    # Inter-feature causal pathways
    ("sleep_hours",           "study_hours"),        # better sleep → more effective study
    ("sleep_hours",           "attendance"),          # better sleep → more likely to attend
    ("study_hours",           "assignments_completed"),  # more study → more assignments done
    ("attendance",            "class_participation"),    # attending class → participating
    ("previous_scores",       "study_hours"),         # past success motivates study
]

EDGE_DESCRIPTIONS = {
    ("study_hours",           "result"):
        "More study hours improve conceptual understanding and increase the probability of passing.",
    ("attendance",            "result"):
        "Higher attendance ensures exposure to structured course material, boosting performance.",
    ("previous_scores",       "result"):
        "Strong prior academic performance provides a cognitive foundation for continued success.",
    ("assignments_completed", "result"):
        "Completing assignments reinforces practical problem solving and directly contributes to grades.",
    ("class_participation",   "result"):
        "Active participation deepens comprehension and active engagement with the subject.",
    ("sleep_hours",           "result"):
        "Adequate sleep improves cognitive function, memory consolidation, and exam performance.",
    ("sleep_hours",           "study_hours"):
        "Well-rested students can concentrate for longer periods and study more effectively.",
    ("sleep_hours",           "attendance"):
        "Sufficient sleep makes it easier to wake up on time and attend morning lectures regularly.",
    ("study_hours",           "assignments_completed"):
        "Students who dedicate more time to academic work are more likely to submit assignments.",
    ("attendance",            "class_participation"):
        "Attending class creates the necessary environment and opportunity to participate in discussions.",
    ("previous_scores",       "study_hours"):
        "Students with higher past scores often possess greater academic self-efficacy, encouraging study habits.",
}

NODE_ROLES = {
    "study_hours":           "feature",
    "attendance":            "feature",
    "previous_scores":       "feature",
    "sleep_hours":           "feature",
    "assignments_completed": "feature",
    "class_participation":   "feature",
    "result":                "target",
}


def build_causal_graph() -> nx.DiGraph:
    """Construct and return the default demo student performance causal DAG."""
    G = nx.DiGraph()
    for node, role in NODE_ROLES.items():
        G.add_node(node, role=role, label=_pretty(node))
    for src, tgt in CAUSAL_EDGES:
        desc = EDGE_DESCRIPTIONS.get((src, tgt), f"Hypothesized influence of {src} on {tgt}.")
        G.add_edge(
            src, tgt,
            relationship_type="Domain Knowledge Assumption",
            description=desc,
            correlation=None
        )
    return G


def get_edge_descriptions() -> list[dict]:
    """Return a list of dicts describing each demo causal relationship."""
    rows = []
    for (src, tgt), desc in EDGE_DESCRIPTIONS.items():
        rows.append({
            "Cause": _pretty(src),
            "Effect": _pretty(tgt),
            "Type": "Domain Knowledge Assumption",
            "Relationship Description": desc,
        })
    return rows


# ---------------------------------------------------------------------------
# Graph Synthesis & Serialization for Custom Uploaded Datasets
# ---------------------------------------------------------------------------

def generate_initial_causal_graph(
    feature_cols: list[str],
    target_col: str,
    df: pd.DataFrame | None = None,
) -> dict:
    """Generate an initial candidate causal graph dictionary from selected dataset schema.

    Important: All generated connections are explicitly flagged as 'Hypothesized Influence'
    rather than proven causal facts.
    """
    G = nx.DiGraph()

    # Add features as cause candidates
    for feat in feature_cols:
        G.add_node(feat, role="feature", label=_pretty(feat))

    # Add target as outcome node
    G.add_node(target_col, role="target", label=_pretty(target_col))

    # Generate initial direct edges from each feature to target
    for feat in feature_cols:
        corr_val = None
        if df is not None and feat in df.columns and target_col in df.columns:
            if pd.api.types.is_numeric_dtype(df[feat]) and pd.api.types.is_numeric_dtype(df[target_col]):
                try:
                    c = df[[feat, target_col]].dropna().corr().iloc[0, 1]
                    corr_val = float(c) if not np.isnan(c) else None
                except Exception:
                    corr_val = None

        corr_str = f" (Observed sample correlation: r = {corr_val:+.2f})" if corr_val is not None else ""
        desc = (
            f"Hypothesized direct influence of '{_pretty(feat)}' on '{_pretty(target_col)}'.{corr_str} "
            f"Assumed prior to causal identification."
        )

        G.add_edge(
            feat,
            target_col,
            relationship_type="Hypothesized Direct Cause",
            description=desc,
            correlation=corr_val,
        )

    # Detect high inter-feature correlations for hypothesis generation
    if df is not None and len(feature_cols) > 1:
        num_feats = [f for f in feature_cols if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
        if len(num_feats) >= 2:
            try:
                corr_matrix = df[num_feats].dropna().corr()
                for i in range(len(num_feats)):
                    for j in range(i + 1, len(num_feats)):
                        f1, f2 = num_feats[i], num_feats[j]
                        r = corr_matrix.loc[f1, f2]
                        if abs(r) >= 0.55:
                            # Add candidate hypothesized inter-feature edge
                            src, tgt = f1, f2
                            desc = (
                                f"Strong statistical co-variation (r = {r:+.2f}). "
                                f"Hypothesized potential dependency or shared mechanism between '{_pretty(src)}' and '{_pretty(tgt)}'."
                            )
                            G.add_edge(
                                src, tgt,
                                relationship_type="Hypothesized Association",
                                description=desc,
                                correlation=float(r)
                            )
            except Exception:
                pass

    return graph_to_dict(G)


def graph_to_dict(G: nx.DiGraph) -> dict:
    """Serialize a NetworkX DiGraph into a JSON-compatible dictionary."""
    nodes = []
    for n in G.nodes():
        nodes.append({
            "id": str(n),
            "role": G.nodes[n].get("role", "feature"),
            "label": G.nodes[n].get("label", _pretty(str(n))),
        })

    edges = []
    for u, v in G.edges():
        edges.append({
            "source": str(u),
            "target": str(v),
            "relationship_type": G.edges[u, v].get("relationship_type", "Hypothesized Direct Cause"),
            "description": G.edges[u, v].get("description", ""),
            "correlation": G.edges[u, v].get("correlation", None),
        })

    return {
        "nodes": nodes,
        "edges": edges,
        "is_dag": nx.is_directed_acyclic_graph(G),
    }


def dict_to_graph(data: dict) -> nx.DiGraph:
    """Deserialize a dictionary structure back into a NetworkX DiGraph."""
    G = nx.DiGraph()
    for n in data.get("nodes", []):
        node_id = str(n["id"])
        G.add_node(
            node_id,
            role=n.get("role", "feature"),
            label=n.get("label", _pretty(node_id)),
        )

    for e in data.get("edges", []):
        src = str(e["source"])
        tgt = str(e["target"])
        G.add_edge(
            src, tgt,
            relationship_type=e.get("relationship_type", "Hypothesized Direct Cause"),
            description=e.get("description", ""),
            correlation=e.get("correlation", None),
        )

    return G


# ---------------------------------------------------------------------------
# Graph Editing Operations (Add, Remove, Edit)
# ---------------------------------------------------------------------------

class GraphEditResult(tuple):
    """A 2-tuple (graph, error) that also supports dict/DiGraph attribute and item access."""
    def __new__(cls, graph, error=None):
        return super().__new__(cls, (graph, error))

    @property
    def graph(self):
        return self[0]

    @property
    def error(self):
        return self[1]

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            return super().__getitem__(item)
        if isinstance(self[0], dict):
            return self[0][item]
        raise TypeError(f"'{type(self[0]).__name__}' object is not subscriptable")

    def __contains__(self, item):
        if isinstance(self[0], dict):
            return item in self[0]
        elif hasattr(self[0], "nodes"):
            return item in self[0].nodes()
        return False

    def __getattr__(self, name):
        return getattr(self[0], name)


def _coerce_input_graph(G) -> tuple[nx.DiGraph, bool]:
    """Helper to convert dict or GraphEditResult to nx.DiGraph, remembering if input was dict-based."""
    if isinstance(G, tuple) and hasattr(G, "graph"):
        G = G.graph
    elif isinstance(G, tuple) and len(G) == 2 and isinstance(G[0], (dict, nx.DiGraph)):
        G = G[0]

    if isinstance(G, dict):
        return dict_to_graph(G), True
    return G, False


def _wrap_graph_result(G_nx: nx.DiGraph, err: str | None, was_dict: bool) -> GraphEditResult:
    """Helper to return GraphEditResult wrapping dict if input was dict, or nx.DiGraph."""
    out_graph = graph_to_dict(G_nx) if was_dict else G_nx
    return GraphEditResult(out_graph, err)


def add_node_to_graph(
    G: nx.DiGraph | dict,
    node_name: str,
    role: str = "feature",
) -> tuple[nx.DiGraph | dict, str | None]:
    """Add a node to the graph. Accepts either nx.DiGraph or serialized dict."""
    G_nx, was_dict = _coerce_input_graph(G)
    clean_name = node_name.strip().replace(" ", "_")
    if not clean_name:
        return _wrap_graph_result(G_nx, "Node name cannot be blank.", was_dict)
    if clean_name in G_nx.nodes():
        return _wrap_graph_result(G_nx, f"Node '{clean_name}' already exists in the graph.", was_dict)

    G_copy = G_nx.copy()
    G_copy.add_node(clean_name, role=role, label=_pretty(clean_name))
    return _wrap_graph_result(G_copy, None, was_dict)


def remove_node_from_graph(
    G: nx.DiGraph | dict,
    node_name: str,
) -> tuple[nx.DiGraph | dict, str | None]:
    """Remove a node and all its connected edges from the graph."""
    G_nx, was_dict = _coerce_input_graph(G)
    if node_name not in G_nx.nodes():
        return _wrap_graph_result(G_nx, f"Node '{node_name}' not found in graph.", was_dict)

    G_copy = G_nx.copy()
    G_copy.remove_node(node_name)
    return _wrap_graph_result(G_copy, None, was_dict)


def add_edge_to_graph(
    G: nx.DiGraph | dict,
    source: str,
    target: str,
    relationship_type: str = "Hypothesized Direct Cause",
    description: str = "",
    correlation: float | None = None,
) -> tuple[nx.DiGraph | dict, str | None]:
    """Add a directed edge from source to target. Validates acyclicity and self-loops."""
    G_nx, was_dict = _coerce_input_graph(G)
    if source == target:
        return _wrap_graph_result(G_nx, "Self-loops are not allowed in causal graphs (a variable cannot cause itself directly).", was_dict)
    if source not in G_nx.nodes() or target not in G_nx.nodes():
        return _wrap_graph_result(G_nx, f"Both source '{source}' and target '{target}' must exist in the graph.", was_dict)
    if G_nx.has_edge(source, target):
        return _wrap_graph_result(G_nx, f"Directed edge '{source} ➔ {target}' already exists.", was_dict)

    G_copy = G_nx.copy()
    G_copy.add_edge(
        source, target,
        relationship_type=relationship_type,
        description=description.strip() or f"Hypothesized directed influence of {source} on {target}.",
        correlation=correlation,
    )

    warning = None
    if not nx.is_directed_acyclic_graph(G_copy):
        warning = f"⚠️ Warning: Adding '{source} ➔ {target}' introduced a cycle into the graph. Standard causal DAGs require acyclicity."

    return _wrap_graph_result(G_copy, warning, was_dict)


def remove_edge_from_graph(
    G: nx.DiGraph | dict,
    source: str,
    target: str,
) -> tuple[nx.DiGraph | dict, str | None]:
    """Remove a directed edge between source and target."""
    G_nx, was_dict = _coerce_input_graph(G)
    if not G_nx.has_edge(source, target):
        return _wrap_graph_result(G_nx, f"Edge '{source} ➔ {target}' does not exist in graph.", was_dict)

    G_copy = G_nx.copy()
    G_copy.remove_edge(source, target)
    return _wrap_graph_result(G_copy, None, was_dict)


def update_edge_in_graph(
    G: nx.DiGraph | dict,
    old_source: str,
    old_target: str,
    new_source: str,
    new_target: str,
    relationship_type: str,
    description: str,
    correlation: float | None = None,
) -> tuple[nx.DiGraph | dict, str | None]:
    """Update an existing edge's attributes or direction."""
    G_nx, was_dict = _coerce_input_graph(G)
    if not G_nx.has_edge(old_source, old_target):
        return _wrap_graph_result(G_nx, f"Original edge '{old_source} ➔ {old_target}' not found.", was_dict)

    G_copy = G_nx.copy()
    G_copy.remove_edge(old_source, old_target)

    if new_source == new_target:
        return _wrap_graph_result(G_nx, "A variable cannot cause itself directly (self-loop).", was_dict)

    G_copy.add_edge(
        new_source, new_target,
        relationship_type=relationship_type,
        description=description.strip() or f"Hypothesized influence of {new_source} on {new_target}.",
        correlation=correlation,
    )

    warning = None
    if not nx.is_directed_acyclic_graph(G_copy):
        warning = f"⚠️ Warning: The updated edge '{new_source} ➔ {new_target}' created a cycle in the causal graph."

    return _wrap_graph_result(G_copy, warning, was_dict)


def get_graph_edge_descriptions(G: nx.DiGraph | dict, df: pd.DataFrame | None = None) -> list[dict]:
    """Extract tabular descriptions and observed correlations for all edges in G."""
    G_nx, _ = _coerce_input_graph(G)
    rows = []
    for u, v in G_nx.edges():
        data = G_nx.edges[u, v]
        rel_type = data.get("relationship_type", "Hypothesized Direct Cause")
        desc = data.get("description", "")
        corr = data.get("correlation", None)

        if corr is None and df is not None and u in df.columns and v in df.columns:
            if pd.api.types.is_numeric_dtype(df[u]) and pd.api.types.is_numeric_dtype(df[v]):
                try:
                    c = df[[u, v]].dropna().corr().iloc[0, 1]
                    corr = float(c) if not np.isnan(c) else None
                except Exception:
                    corr = None

        corr_display = f"{corr:+.2f}" if corr is not None else "N/A (Categorical / Latent)"

        rows.append({
            "Cause (Source)": _pretty(u),
            "Effect (Target)": _pretty(v),
            "Relationship Status": rel_type,
            "Observed Correlation (r)": corr_display,
            "Causal Rationale / Hypothesis": desc,
        })
    return rows


# ---------------------------------------------------------------------------
# Layout Algorithms & Visualizations (Plotly & Matplotlib)
# ---------------------------------------------------------------------------

def compute_dag_layout(G: nx.DiGraph | dict) -> dict:
    """Compute an intuitive visual positioning dictionary for DAG nodes.

    Places Confounders/Latent factors at the top, Features on the left,
    and Target outcomes on the right.
    """
    G_nx, _ = _coerce_input_graph(G)
    pos = {}
    target_nodes = [n for n, d in G_nx.nodes(data=True) if d.get("role") == "target"]
    confounder_nodes = [n for n, d in G_nx.nodes(data=True) if d.get("role") == "confounder"]
    feature_nodes = [n for n in G_nx.nodes() if n not in target_nodes and n not in confounder_nodes]

    # Top: Confounders
    n_conf = len(confounder_nodes)
    for idx, n in enumerate(confounder_nodes):
        x = -0.5 + (idx / max(n_conf - 1, 1)) if n_conf > 1 else 0.0
        pos[n] = (x, 1.6)

    # Left: Features
    n_feats = len(feature_nodes)
    for idx, n in enumerate(feature_nodes):
        if n_feats == 1:
            pos[n] = (-1.5, 0.0)
        else:
            y = 1.3 - 2.6 * (idx / (n_feats - 1))
            x = -1.5 + 0.35 * np.sin(np.pi * (idx / (n_feats - 1)))
            pos[n] = (x, y)

    # Right: Target(s)
    n_targets = len(target_nodes)
    for idx, n in enumerate(target_nodes):
        if n_targets == 1:
            pos[n] = (1.5, 0.0)
        else:
            y = 0.8 - 1.6 * (idx / (n_targets - 1))
            pos[n] = (1.5, y)

    # Fallback for disconnected / unassigned nodes
    for n in G_nx.nodes():
        if n not in pos:
            pos[n] = (0.0, 0.0)

    return pos


def build_causal_plotly_figure(
    G: nx.DiGraph | dict,
    title: str = "Interactive Causal Directed Acyclic Graph (DAG)",
) -> go.Figure:
    """Render an interactive, dark-themed Causal DAG using NetworkX and Plotly."""
    G_nx, _ = _coerce_input_graph(G)
    if len(G_nx.nodes()) == 0:
        fig = go.Figure()
        fig.update_layout(
            title="Empty Causal Graph",
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font=dict(color="#94a3b8"),
        )
        return fig

    pos = compute_dag_layout(G_nx)

    edge_x = []
    edge_y = []
    annotations = []
    mid_x, mid_y, mid_hover = [], [], []

    for u, v, data in G_nx.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        # Shorten arrow endpoints so they don't clip inside node circle
        dx, dy = x1 - x0, y1 - y0
        dist = np.sqrt(dx**2 + dy**2)
        if dist > 0:
            scale_src = 0.16 / dist
            scale_tgt = 0.22 / dist
            ax_pos = x0 + dx * scale_src
            ay_pos = y0 + dy * scale_src
            x_pos = x1 - dx * scale_tgt
            y_pos = y1 - dy * scale_tgt
        else:
            ax_pos, ay_pos, x_pos, y_pos = x0, y0, x1, y1

        annotations.append(
            dict(
                ax=ax_pos, ay=ay_pos,
                x=x_pos, y=y_pos,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.4,
                arrowwidth=2,
                arrowcolor="#818cf8",
                opacity=0.85,
            )
        )

        # Midpoint for hover tooltip
        mx = (x0 + x1) / 2
        my = (y0 + y1) / 2
        mid_x.append(mx)
        mid_y.append(my)
        rel_type = data.get("relationship_type", "Hypothesized Direct Cause")
        desc = data.get("description", "")
        corr = data.get("correlation")
        corr_txt = f"<br><b>Observed Correlation (r):</b> {corr:+.2f}" if corr is not None else ""
        mid_hover.append(
            f"<b>Directed Link:</b> {_pretty(u)} ➔ {_pretty(v)}<br>"
            f"<b>Status:</b> {rel_type}{corr_txt}<br>"
            f"<b>Hypothesized Mechanism:</b> {desc}"
        )

    edge_line_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1.5, color="#475569"),
        hoverinfo="none",
        showlegend=False,
    )

    edge_mid_trace = go.Scatter(
        x=mid_x, y=mid_y,
        mode="markers",
        marker=dict(size=12, color="#818cf8", opacity=0.45, symbol="diamond"),
        hoverinfo="text",
        hovertext=mid_hover,
        showlegend=False,
    )

    node_x, node_y, node_colors, node_text, node_hover = [], [], [], [], []
    for n, data in G_nx.nodes(data=True):
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        role = data.get("role", "feature")

        if role == "target":
            node_colors.append("#f43f5e")
        elif role == "confounder":
            node_colors.append("#f59e0b")
        else:
            node_colors.append("#6366f1")

        clean_name = _pretty(n)
        node_text.append(clean_name)
        node_hover.append(
            f"<b>Node:</b> {clean_name} (<code>{n}</code>)<br>"
            f"<b>Role:</b> {role.title()}<br>"
            f"<b>Incoming Causes (In-degree):</b> {G_nx.in_degree(n)}<br>"
            f"<b>Outgoing Influences (Out-degree):</b> {G_nx.out_degree(n)}"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        textfont=dict(color="#f8fafc", size=11, family="sans-serif"),
        hoverinfo="text",
        hovertext=node_hover,
        marker=dict(
            size=36,
            color=node_colors,
            line=dict(width=2, color="#1e293b"),
            opacity=0.95,
        ),
        showlegend=False,
    )

    # Legend proxy traces
    leg_feat = go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=12, color="#6366f1"),
        name="Feature (Hypothesized Cause)"
    )
    leg_tgt = go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=12, color="#f43f5e"),
        name="Target (Outcome)"
    )
    leg_conf = go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=12, color="#f59e0b"),
        name="Confounder (Latent / Context)"
    )

    fig = go.Figure(
        data=[edge_line_trace, edge_mid_trace, node_trace, leg_feat, leg_tgt, leg_conf],
        layout=go.Layout(
            title=dict(text=title, font=dict(color="#f8fafc", size=15)),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.12,
                xanchor="center",
                x=0.5,
                font=dict(color="#cbd5e1", size=11),
                bgcolor="rgba(15, 23, 42, 0.7)",
                bordercolor="#334155",
                borderwidth=1,
            ),
            hovermode="closest",
            margin=dict(b=40, l=25, r=25, t=55),
            annotations=annotations,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            height=540,
        ),
    )
    return fig


def plot_causal_graph(G: nx.DiGraph | dict):
    """Render the causal DAG with a polished dark-theme matplotlib style (Legacy Demo Support)."""
    G_nx, _ = _coerce_input_graph(G)
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    pos = compute_dag_layout(G_nx)

    node_colors = []
    for n in G_nx.nodes():
        role = G_nx.nodes[n].get("role", "feature")
        if role == "target":
            node_colors.append("#f43f5e")
        elif role == "confounder":
            node_colors.append("#f59e0b")
        else:
            node_colors.append("#6366f1")

    labels = {n: _pretty(n) for n in G_nx.nodes()}

    nx.draw_networkx_edges(
        G_nx, pos, ax=ax,
        edge_color="#8888aa",
        width=1.8,
        alpha=0.75,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18,
        connectionstyle="arc3,rad=0.08",
        min_source_margin=22,
        min_target_margin=22,
    )

    nx.draw_networkx_nodes(
        G_nx, pos, ax=ax,
        node_color=node_colors,
        node_size=2400,
        edgecolors="#222",
        linewidths=2,
        alpha=0.92,
    )

    nx.draw_networkx_labels(
        G_nx, pos, labels, ax=ax,
        font_size=9,
        font_weight="bold",
        font_color="white",
    )

    feature_patch = mpatches.Patch(color="#6366f1", label="Feature (Hypothesized Cause)")
    target_patch = mpatches.Patch(color="#f43f5e", label="Target (Effect)")
    conf_patch = mpatches.Patch(color="#f59e0b", label="Confounder (Context / Latent)")

    ax.legend(
        handles=[feature_patch, target_patch, conf_patch],
        loc="upper left",
        fontsize=9,
        facecolor="#1a1a2e",
        edgecolor="#333",
        labelcolor="#ccc",
    )

    ax.set_title(
        "Causal Directed Acyclic Graph (DAG) — Hypothesized Structure",
        color="#ccc", fontsize=13, fontweight="bold", pad=15,
    )
    ax.axis("off")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Educational Content & Scientific Foundations
# ---------------------------------------------------------------------------

SCIENTIFIC_DISCLAIMER_CONTENT = {
    "title": "Scientific Guidelines: Qualitative DAGs vs. Formal Causal Inference",
    "content": """
> [!IMPORTANT]
> **Qualitative Assumption Modeling vs. Statistical Evidence**
> 1. **Assumptions, Not Discoveries:** Every arrow in this directed graph represents a **human hypothesis or domain assumption**. Generating or drawing an arrow between variables does **not** prove true causation.
> 2. **Correlation $\\neq$ Causation:** Observational datasets record statistical co-variation ($P(Y|X)$). True causal effects require interventional distributions ($P(Y|\\text{do}(X))$), randomized controlled trials (RCTs), instrumental variables, or verified structural equation models.
> 3. **Confounding & Latent Bias:** A strong statistical correlation between $A$ and $B$ can be driven entirely by an unobserved common cause $U$ ($A \\leftarrow U \\rightarrow B$).
> 4. **No Formal Inference Claimed:** This interactive canvas provides a structural sandbox for explicit reasoning under Pearl's Structural Causal Model (SCM) framework, but does not claim that automated causal discovery has taken place.
""",
}

CORRELATION_VS_CAUSATION = {
    "title": "Correlation vs Causation",
    "content": """
**Correlation** means two variables tend to move together — when one changes,
the other often changes too. But this does *not* mean one **causes** the other.

**Causation** means a change in one variable directly **produces** a change
in another. Establishing causation requires more than observing patterns;
it demands understanding the **mechanism** behind the relationship.

---

**Example from data:**

| Approach | Statement | Interpretation |
|----------|-----------|----------------|
| 🔗 **Correlation** | *"Variables $X$ and $Y$ move in tandem ($r > 0$)."* | Associational pattern in observed records |
| ➡️ **Causation** | *"Intervening to set $\\text{do}(X = x)$ changes $Y$."* | Interventional reality under change |

The correlation is easy to observe in data. The causal claim is much stronger
and requires domain knowledge, controlled experiments, or formal causal identification to support.

---

**Why does this matter?**

- **Ice cream sales** are correlated with **drowning rates** — but ice cream
  doesn't cause drowning. Both are caused by a **confounding variable**: hot summer weather.
- Similarly, a predictive machine-learning model might latch onto a **spurious correlation** that doesn't
  reflect a true causal mechanism, resulting in catastrophic failure when external policies change.
""",
}

TRUSTWORTHY_AI_CONTENT = {
    "title": "Why Causal Analysis Makes AI More Trustworthy",
    "content": """
Traditional machine-learning models learn **associations** from data.
They can tell you *what* factors are associated with an outcome, but they
cannot tell you *why* — or what would happen if you **intervened**.

**Causal AI** goes further by asking:

> *"If we **change** a specific factor, what effect will that have on the outcome?"*

---

| Aspect | Traditional ML | Causal AI |
|--------|---------------|-----------|
| **Question** | What patterns exist? | What causes what? |
| **Statement** | *"Feature $X$ is associated with outcome $Y$."* | *"Would increasing $X$ actually cause an improvement in $Y$?"* |
| **Reliability** | May rely on spurious correlations | Seeks invariant mechanisms |
| **Actionability** | Describes what happened | Prescribes what to intervene on |
| **Fairness** | May perpetuate proxy biases | Can identify and isolate unfair causal pathways |

---

**Benefits for trustworthy systems:**

1. **Transparency** — stakeholders can inspect, debate, and falsify the assumed causal
   structure, transforming a black-box model into an auditable process.
2. **Robustness** — models grounded in causal mechanisms are resilient when
   underlying data distributions shift (out-of-distribution generalization).
3. **Actionability** — decision-makers can reason about **interventions**
   (e.g., *"If we invest in a specific intervention, will outcomes improve?"*).
4. **Fairness & Counterfactual Equity** — causal graphs help identify and block
   **unfair causal pathways** (e.g., sensitive attributes influencing outcomes through proxies).
""",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pretty(name: str) -> str:
    """Convert snake_case to Title Case for human-friendly display."""
    return str(name).replace("_", " ").title()
