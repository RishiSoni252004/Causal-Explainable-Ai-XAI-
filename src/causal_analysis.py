"""
causal_analysis.py — Basic causal graph and educational causal reasoning.

Provides:
    • A directed causal graph (NetworkX DiGraph) encoding assumed
      domain-knowledge relationships between student features and outcomes.
    • A styled matplotlib visualisation of the graph.
    • Structured relationship descriptions and educational content on
      correlation vs causation and trustworthy AI.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Causal graph definition
# ---------------------------------------------------------------------------

# Each edge (A, B) reads: "A is assumed to causally influence B."
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

# Human-readable explanation for every edge
EDGE_DESCRIPTIONS = {
    ("study_hours",           "result"):
        "More study hours improve understanding and increase the probability of passing.",
    ("attendance",            "result"):
        "Higher attendance ensures exposure to course material, boosting performance.",
    ("previous_scores",       "result"):
        "Strong prior academic performance provides a foundation for continued success.",
    ("assignments_completed", "result"):
        "Completing assignments reinforces learning and directly contributes to grades.",
    ("class_participation",   "result"):
        "Active participation deepens comprehension and engagement with the subject.",
    ("sleep_hours",           "result"):
        "Adequate sleep improves cognitive function, memory consolidation, and exam performance.",
    ("sleep_hours",           "study_hours"):
        "Well-rested students can study more effectively and for longer periods.",
    ("sleep_hours",           "attendance"):
        "Sufficient sleep makes it easier to wake up and attend classes regularly.",
    ("study_hours",           "assignments_completed"):
        "Students who dedicate more time to studying are more likely to complete assignments.",
    ("attendance",            "class_participation"):
        "Students who attend class have the opportunity (and tendency) to participate.",
    ("previous_scores",       "study_hours"):
        "Students with higher past scores are often more motivated to maintain study habits.",
}

# Node metadata for styling
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
    """Construct and return the causal directed graph."""
    G = nx.DiGraph()
    for node, role in NODE_ROLES.items():
        G.add_node(node, role=role)
    G.add_edges_from(CAUSAL_EDGES)
    return G


def get_edge_descriptions() -> list[dict]:
    """Return a list of dicts describing each causal relationship."""
    rows = []
    for (src, tgt), desc in EDGE_DESCRIPTIONS.items():
        rows.append({
            "Cause": _pretty(src),
            "Effect": _pretty(tgt),
            "Relationship": desc,
        })
    return rows


# ---------------------------------------------------------------------------
# Graph visualisation
# ---------------------------------------------------------------------------

def plot_causal_graph(G: nx.DiGraph):
    """Render the causal DAG with a polished dark-theme style.

    Returns a matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    # Layout — use a shell layout with 'result' in the centre
    pos = _causal_layout(G)

    # Node colours
    node_colors = []
    for n in G.nodes():
        role = G.nodes[n].get("role", "feature")
        if role == "target":
            node_colors.append("#f5576c")
        else:
            node_colors.append("#667eea")

    # Labels
    labels = {n: _pretty(n) for n in G.nodes()}

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#8888aa",
        width=2.0,
        alpha=0.7,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=20,
        connectionstyle="arc3,rad=0.1",
        min_source_margin=25,
        min_target_margin=25,
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=2800,
        edgecolors="#222",
        linewidths=2,
        alpha=0.92,
    )

    # Draw labels
    nx.draw_networkx_labels(
        G, pos, labels, ax=ax,
        font_size=9,
        font_weight="bold",
        font_color="white",
    )

    # Legend
    feature_patch = mpatches.Patch(color="#667eea", label="Feature (Cause)")
    target_patch = mpatches.Patch(color="#f5576c", label="Target (Effect)")
    ax.legend(
        handles=[feature_patch, target_patch],
        loc="upper left",
        fontsize=10,
        facecolor="#1a1a2e",
        edgecolor="#333",
        labelcolor="#ccc",
    )

    ax.set_title(
        "Assumed Causal Graph — Student Performance",
        color="#ccc", fontsize=14, fontweight="bold", pad=15,
    )
    ax.axis("off")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Educational content
# ---------------------------------------------------------------------------

CORRELATION_VS_CAUSATION = {
    "title": "Correlation vs Causation",
    "content": """
**Correlation** means two variables tend to move together — when one changes,
the other often changes too. But this does *not* mean one **causes** the other.

**Causation** means a change in one variable directly **produces** a change
in another. Establishing causation requires more than observing patterns;
it demands understanding the **mechanism** behind the relationship.

---

**Example from our data:**

| Approach | Statement |
|----------|-----------|
| 🔗 **Correlation** | *"Students who study more hours tend to pass more often."* |
| ➡️ **Causation** | *"Increasing study hours **causes** a higher probability of passing."* |

The correlation is easy to observe in data. The causal claim is much stronger
and requires domain knowledge, experiments, or formal causal reasoning to support.

---

**Why does this matter?**

- **Ice cream sales** are correlated with **drowning rates** — but ice cream
  doesn't cause drowning. Both are caused by a **confounding variable**: hot
  weather.
- Similarly, a model might learn a **spurious correlation** that doesn't
  reflect a true causal mechanism, leading to unreliable predictions.
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
| **Statement** | *"Study hours are associated with passing."* | *"Would increasing study hours actually cause a change in the probability of passing?"* |
| **Reliability** | May rely on spurious correlations | Seeks genuine mechanisms |
| **Actionability** | Describes what happened | Prescribes what to do |
| **Fairness** | May perpetuate biases | Can identify and remove unfair pathways |

---

**Benefits for trustworthy systems:**

1. **Transparency** — stakeholders can inspect and debate the assumed causal
   structure, not just a black-box model.
2. **Robustness** — models grounded in causal mechanisms are less likely to
   break when the data distribution shifts.
3. **Actionability** — decision-makers can reason about **interventions**
   (e.g., *"If we add a tutoring programme, will pass rates improve?"*).
4. **Fairness** — causal graphs help identify and block **unfair causal
   pathways** (e.g., a demographic variable influencing the outcome through
   a proxy).
""",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pretty(name: str) -> str:
    """Convert snake_case to Title Case for display."""
    return name.replace("_", " ").title()


def _causal_layout(G: nx.DiGraph) -> dict:
    """Custom deterministic layout with 'result' on the right and
    features arranged on the left/centre."""
    pos = {
        "sleep_hours":           (-1.6,  1.2),
        "previous_scores":       (-1.6, -1.2),
        "study_hours":           (-0.3,  1.5),
        "attendance":            (-0.3,  0.0),
        "assignments_completed": ( 0.6,  1.0),
        "class_participation":   ( 0.6, -1.0),
        "result":                ( 1.8,  0.0),
    }
    return pos
