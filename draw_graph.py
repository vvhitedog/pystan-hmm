import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout


def draw_tree_graph(G, labels):
    pos = graphviz_layout(G, prog="twopi")
    plt.figure()
    nx.draw(G, pos, edge_color='black', width=1, linewidths=1, node_shape = 's',
            node_size=4000, node_color='pink', alpha=0.9,
            labels={node: labels[node] for node in G.nodes()})
    plt.axis('off')
    plt.show()


def draw_tree(edges, labels):
    G = nx.Graph(directed=True)
    G.add_edges_from(edges)
    draw_tree_graph(G, labels)
