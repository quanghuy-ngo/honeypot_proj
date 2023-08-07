import networkx as nx
import matplotlib
from networkx.readwrite.gpickle import write_gpickle
from utility import display, report
from treedecomposition import build_tree_decomposition
from networkx.drawing.layout import multipartite_layout


DA = 0
DG = nx.DiGraph(DA=DA)

DG.add_edge(1, 0)
DG.add_edge(2, 0)
DG.add_edge(3, 1)
DG.add_edge(3, 2)
DG.add_edge(4, 1)
DG.add_edge(4, 2)
DG.add_edge(5, 3)
DG.add_edge(6, 4)
DG.add_edge(10, 0)
DG.add_edge(11, 0)
DG.add_edge(7, 10)
DG.add_edge(7, 11)
DG.add_edge(7, 1)


DG.nodes[0]["node_type"] = "DA"
DG.nodes[1]["node_type"] = ""
DG.nodes[2]["node_type"] = ""
DG.nodes[3]["node_type"] = ""
DG.nodes[4]["node_type"] = ""
DG.nodes[5]["node_type"] = ""
DG.nodes[6]["node_type"] = ""
DG.nodes[7]["node_type"] = ""
DG.nodes[10]["node_type"] = ""
DG.nodes[11]["node_type"] = ""


DG.nodes[0]["layer"] = 4
DG.nodes[1]["layer"] = 3
DG.nodes[2]["layer"] = 3
DG.nodes[10]["layer"] = 3
DG.nodes[11]["layer"] = 3

DG.nodes[3]["layer"] = 2
DG.nodes[4]["layer"] = 2
DG.nodes[7]["layer"] = 2

DG.nodes[5]["layer"] = 1
DG.nodes[6]["layer"] = 1

DG.nodes[1]["blockable"] = True
DG.nodes[3]["blockable"] = True
DG.nodes[4]["blockable"] = True


DG.nodes[10]["blockable"] = False
DG.nodes[11]["blockable"] = False
DG.nodes[2]["blockable"] = False

DG.nodes[5]["node_type"] = "S"
DG.nodes[6]["node_type"] = "S"
DG.nodes[7]["node_type"] = "S"


# for v in range(10, 15):
#     DG.add_edge(v, 3)
#     DG.nodes[v]["node_type"] = "S"
#     DG.nodes[v]["layer"] = 0
#     DG[v][3]["blockable"] = True

report(DG)
DG.graph["budget"] = 2
DG.graph["start_nodes"] = tuple(list(range(10, 13)))
DG.graph["split_nodes"] = (3,)

# write_gpickle(DG, "/home/m/Dropbox/examplegraph.gpickle")
write_gpickle(DG, "/home/andrewngo/Desktop/CFR/Code/AAAI/examplegraph_2.gpickle")
# C:\Users\DELL\Documents\Research_Express\routes\AAAI\examplegraph.py

pos = multipartite_layout(DG, subset_key="layer")
pos[11][1] = -0.3
pos[10][1] = -0.4
pos[2][1] = 0.6
pos[1][1] = 0
matplotlib.use("TkAgg")
nx.draw(
    DG,
    pos=pos,
    with_labels=True,
    font_size=16,
    node_size=800,
    node_color="#cccccc",
    font_weight="bold",
    # width=[1 if DG[u][v]["blockable"] else 3 for u, v in DG.edges],
)
matplotlib.pyplot.show()


def beautify_td_nodes(v):
    numbers = [v[0]] + list(v[1])
    return "  " + ",".join([str(x) for x in numbers]) + "  "


TD = build_tree_decomposition(DG)
display(TD)
pos = multipartite_layout(TD, subset_key="layer")
pos[11][1] = 0.6
nx.draw(
    TD,
    pos=pos,
    with_labels=True,
    font_size=16,
    node_size=1600,
    node_color="#cccccc",
    font_weight="bold",
    labels={v: beautify_td_nodes(v) for v in TD.nodes()},
)
matplotlib.pyplot.show()