import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import torch

from graphs import test_set

def save_graph(data, path, max_edges=500):
    ei, y = data.edge_index, data.y
    if ei.size(1) > max_edges:
        idx = torch.randperm(ei.size(1))[:max_edges]
        ei, y = ei[:, idx], y[idx]

    G = nx.MultiDiGraph()
    G.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))

    # This will color the edges
    colors = ["lightgray" if lbl == 0 else "red" for lbl in y.tolist()]

    pos = nx.spring_layout(G, seed=0)
    plt.figure(figsize=(12, 12))
    nx.draw_networkx_nodes(G, pos, node_size=20, node_color="steelblue")
    nx.draw_networkx_edges(G, pos, edge_color=colors, arrows=False, alpha=0.5, width=0.7)
    plt.axis("off")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

def main():
    g = max(test_set, key=lambda d: (d.y != 0).sum().item())
    save_graph(g, "window.png")

if __name__ == "__main__":
    main()