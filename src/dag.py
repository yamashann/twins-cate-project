"""Build and render the DAG for the Twins causal model.

Nodes
-----
U        : unobserved pair-level / maternal confounders (placental position, etc.)
X        : observed maternal and pregnancy covariates (gestational age, parity, ...)
T        : indicator for the heavier twin (1 = heavier)
Y        : one-year mortality

Edges
-----
U -> X, U -> T, U -> Y     : unobserved confounding
X -> T, X -> Y             : observed confounding (the backdoor adjustment set)
T -> Y                     : causal effect of interest

Backdoor set
------------
{X} blocks all backdoor paths from T to Y that go through observed variables; the
adjustment is valid under the conditional-independence assumption Y(t) ⊥ T | X. The
unobserved U is what the sensitivity analysis stresses.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from .utils import ensure_dir


def build_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(["U", "X", "T", "Y"])
    g.add_edges_from(
        [
            ("U", "X"),
            ("U", "T"),
            ("U", "Y"),
            ("X", "T"),
            ("X", "Y"),
            ("T", "Y"),
        ]
    )
    return g


def to_dowhy_gml(g: nx.DiGraph) -> str:
    """Return a GML string compatible with DoWhy's CausalModel."""
    return "\n".join(nx.generate_gml(g))


def render_dag(out_path: Path) -> Path:
    """Render the DAG to a PNG using graphviz if available, otherwise matplotlib."""
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    g = build_graph()
    try:
        from graphviz import Digraph

        dot = Digraph(format=out_path.suffix.lstrip(".") or "png")
        dot.attr(rankdir="LR")
        observed = {"X", "T", "Y"}
        for n in g.nodes:
            if n in observed:
                dot.node(n, shape="ellipse", style="filled", fillcolor="#dde6f5")
            else:
                dot.node(n, shape="ellipse", style="dashed")
        for u, v in g.edges:
            style = "dashed" if u == "U" else "solid"
            dot.edge(u, v, style=style)
        dot.render(out_path.with_suffix(""), cleanup=True)
        return out_path
    except Exception:
        import matplotlib.pyplot as plt

        pos = {"U": (0, 1), "X": (1, 1), "T": (2, 0.5), "Y": (3, 1)}
        fig, ax = plt.subplots(figsize=(6, 3.5))
        nx.draw(
            g,
            pos=pos,
            with_labels=True,
            ax=ax,
            node_color="#dde6f5",
            edgecolors="black",
            node_size=1600,
            arrows=True,
        )
        ax.set_title("Twins DAG")
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        return out_path


if __name__ == "__main__":
    p = render_dag(Path("figures/dag.png"))
    print(f"DAG rendered to {p}")
