import networkx as nx
from graph_def import NODES, EDGES

def build_graph() -> nx.DiGraph:
    G = nx.DiGraph()
    for nid, attrs in NODES.items():
        G.add_node(nid, **attrs)
    for s, t, attrs in EDGES:
        G.add_edge(s, t, **attrs)
    return G

def enumerate_candidates(G: nx.DiGraph, anchors: list[str]):
    """
    Enumerate 1-hop and 2-hop *paths* per anchor.

    Returns dict entries shaped for the downstream runtime:
      {
        'anchor': 'DIABETES',
        'path': ['DIABETES','CORTICO','FILL_4'],   # 1-hop or 2-hop
        'apply_nodes': ['CORTICO','FILL_4'],       # what gets ANDed with baseline
        'unrealized_gain': 1.9 * 2.5
      }
    """
    out = []
    for a in anchors:
        for n1 in G.successors(a):
            g1 = G[a][n1]["lift"]
            out.append({
                "anchor": a,
                "path": [a, n1],
                "apply_nodes": [n1],
                "unrealized_gain": g1,
            })
            for n2 in G.successors(n1):
                g2 = g1 * G[n1][n2]["lift"]
                out.append({
                    "anchor": a,
                    "path": [a, n1, n2],
                    "apply_nodes": [n1, n2],
                    "unrealized_gain": g2,
                })
    return out

def union_dedupe(candidates: list[dict]) -> list[dict]:
    """
    Union + dedupe across anchors.
    Keyed by the full path, because hop-2 meaning is path-specific.
    """
    best = {}
    for c in candidates:
        key = tuple(c["path"])
        if key not in best or c["unrealized_gain"] > best[key]["unrealized_gain"]:
            best[key] = c
    return list(best.values())