"""Per-trip road graph (contract 1). Node/arc ids are local to each trip, so a graph is built per trip."""
import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def arc_lengths(trip):
    return np.hypot(*(trip["nxy"][trip["arcs"][:, 1]] - trip["nxy"][trip["arcs"][:, 0]]).T)


def build_graph(nodes, arcs, nxy=None):
    """networkx DiGraph; edge key data: arc_id (row index in .arcs) and length (m, if nxy given)."""
    g = nx.DiGraph()
    g.add_nodes_from(range(len(nodes)))
    if nxy is not None:
        L = np.hypot(*(nxy[arcs[:, 1]] - nxy[arcs[:, 0]]).T)
        g.add_edges_from((int(a), int(b), {"arc_id": i, "length": float(L[i])}) for i, (a, b) in enumerate(arcs))
    else:
        g.add_edges_from((int(a), int(b), {"arc_id": i}) for i, (a, b) in enumerate(arcs))
    return g


def connectivity_stats(trip):
    """Nodes, arcs, weakly-connected components, size of the largest one, isolated nodes."""
    N, arcs = len(trip["nodes"]), trip["arcs"]
    g = coo_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(N, N))
    k, lab = connected_components(g, directed=False)
    sizes = np.bincount(lab)
    deg = np.bincount(arcs[:, 0], minlength=N) + np.bincount(arcs[:, 1], minlength=N)
    key = arcs[:, 0].astype(np.int64) * N + arcs[:, 1]
    rev = arcs[:, 1].astype(np.int64) * N + arcs[:, 0]
    return dict(n_nodes=N, n_arcs=len(arcs), n_components=int(k), largest_comp_frac=float(sizes.max() / N),
                isolated_nodes=int((deg == 0).sum()), two_way_frac=float(np.isin(key, rev).mean()))
