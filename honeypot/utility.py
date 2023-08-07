import networkx as nx
import matplotlib
from networkx.drawing.layout import multipartite_layout
from patch import topological_generations
from networkx.algorithms.shortest_paths.generic import shortest_path
from networkx.exception import NetworkXNoPath
from math import prod
from pulp import *
import copy
from networkx.algorithms import tournament
import time
def report(CG, ignore_assert=False):
    print("##############REPORT################")
    n = CG.number_of_nodes()
    m = CG.number_of_edges()
    print("n,m", n, m)
    for v in CG.nodes():
        if CG.out_degree(v) == 0 and not ignore_assert:
            assert v == CG.graph["DA"]
    d = [CG.out_degree(v) for v in CG.nodes() if CG.out_degree(v) > 1] + [1]
    print("max out degree", max(d))
    print("number of splitting nodes", len(d) - 1)
    print("number of cf", prod(d))
    print("number of feedback edges", m - n + 1)
    print("####################################")
    
    
    string = "n,m: " + str(n) + ", "+ str(m) + "\n" + "max out degree: " + str(max(d)) + "\n" + "number of splitting nodes: " + str(len(d) - 1) + "\n" + "number of cf: " + str(prod(d)) + "\n" + "number of feedback edges: " + str(m - n + 1) + "\n"
    return string

def correct_dist(CG, v):
    DA = CG.graph["DA"]
    try:
        path = shortest_path(CG, v, DA)
        return len(path) - 1
    except NetworkXNoPath:
        return None


def is_start(CG, v):
    if v in CG.nodes():
        return CG.nodes[v]["node_type"] == "S"
    else:
        return False


def remove_start(CG, v):
    assert is_start(CG, v)
    CG.nodes[v]["node_type"] = ""


def is_blockable(CG, u, v):
    if "blockable" in CG[u][v]:
        return CG[u][v]["blockable"]
    else:
        return False

def is_node_blockable(CG, v):
    if "blockable" in CG.nodes[v]:
        if v == CG.graph["DA"] or v in CG.graph["starting_nodes"]:
            return False
        else:
            return CG.nodes[v]["blockable"]
    else:
        return False

def set_blockable(CG, u, v, val):
    CG[u][v]["blockable"] = val


def remove_dead_nodes(CG):
    keepGoing = True
    while keepGoing:
        keepGoing = False
        for v in set(CG.nodes):
            if correct_dist(CG, v) is None:
                CG.remove_node(v)
                keepGoing = True


def evaluate(CG):
    # todo: not efficient
    # but ok-ish if the number of starting nodes is small
    res = 0
    DA = CG.graph["DA"]
    for v in CG.nodes():
        if is_start(CG, v):
            try:
                path = shortest_path(CG, v, DA)
                res += f(len(path) - 1, CG)
            except NetworkXNoPath:
                pass
    return res


def f(CG, num_block_path, node):
    score = (1 - ((num_block_path)/CG.graph["path_number_dict"][node]))*(1/(len(CG.graph["starting_nodes"]))) 
    return score


# attacker clean path proportion score
# def evaluate_flow(G, block_node_list):
#     path_update = G.graph["path_all"]
#     path_number_dict = G.graph["path_number_dict"]
#     starting_nodes = G.graph["starting_nodes"]
#     score = 1
    
#     for block_node in block_node_list:
#         path_temp = []
#         score = 0
#         for path in path_update:
#             # if block node in a path, delete that path.
#             if block_node not in path:
#                 path_temp.append(path)
#                 score += (1/(len(starting_nodes)))*(1/path_number_dict[path[0]])
#         path_update = path_temp
#     return score


def evaluate_flow(G, block_node_list):
    path_update = G.graph["path_all"]
    path_number_dict = G.graph["path_number_dict"]
    starting_nodes = G.graph["starting_nodes"]
    score = 0
    for block_node in block_node_list:
        path_temp = []
        #score = 0
        for path in path_update:
            # if block node in a path, delete that path.
            if block_node in path:
                #path_temp.append(path)
                score += (1/(len(starting_nodes)))*(1/path_number_dict[path[0]])
            else:
                path_temp.append(path)
        path_update = path_temp
    score = 1 - score
    return score



def evaluate_flow_2(G, block_node_list):
    path_update = G.graph["path_all"]
    path_number_dict = G.graph["path_number_dict"]
    starting_nodes = G.graph["starting_nodes"]
    score = 1
    
    path_number_dict = get_total_path(G)
    
    for block_node in block_node_list:
        path_temp = []
        score = 0
        for path in path_update:
            # if block node in a path, delete that path.
            if block_node not in path:
                path_temp.append(path)
                score += (1/(len(starting_nodes)))*(1/path_number_dict[path[0]])
        path_update = path_temp
    return score

def evaluate_flow_competent(G, block_node_list):
    starting_nodes = G.graph["starting_nodes"]
    DA = G.graph["DA"]
    score = 0
    G_blocked = copy.deepcopy(G)
    G_blocked.remove_nodes_from(block_node_list)

    for i in range(len(starting_nodes)):
        try: 
            path = nx.shortest_path(G_blocked, source=starting_nodes[i], target=DA)
            score += 1/len(starting_nodes)
        except NetworkXNoPath:
            continue
            

    return score


def get_total_path(graph_condensed):
    
    starting_nodes = graph_condensed.graph["starting_nodes"]
    budget = graph_condensed.graph["budget"]
    DA = graph_condensed.graph["DA"]
    blockable_node = []
    for i in graph_condensed.nodes():
        if graph_condensed.nodes[i]["blockable"]:
            blockable_node.append(i)
    
    print(len(blockable_node))
    
    
    
    # Linear Programming to find the clean path
    
    prob1 = LpProblem("Finding Total of Clean Path SubProblem", LpMaximize)
    
    theta = LpVariable.dicts("# of total path to DA", graph_condensed.nodes(), lowBound = 0, cat="Integer")
    
    _variable = theta[DA]
    _variable.setInitialValue(1)
    _variable.fixValue()
    
    prob1 += (
        lpSum([theta[n] for n in starting_nodes]),
        "Chance attacker get to DA node from given starting nodes",)

    for i in graph_condensed.nodes():
        
        if i == DA: continue
        inbound_node = []
        for u, v in graph_condensed.out_edges(i):
            inbound_node.append(v)
        
        # print(i, inbound_node)
        prob1 += (
                theta[i] - lpSum(theta[n] for n in inbound_node) == 0,
            ""
        )      
    
    
    
    prob1.writeLP("netflowsimple.lp")

    # The problem is solved using PuLP's choice of Solver
    prob1.solve()
    # The status of the solution is printed to the screen
    
    theta = dict()
    for v in prob1.variables():
        node = int(v.name.split("_")[-1])
        theta[node] = v.varValue
    return theta



def display(DG, always_delete=None):
    matplotlib.use("TkAgg")

    CG = DG.copy()
    if always_delete is not None:
        CG.remove_nodes_from(always_delete)
        remove_dead_nodes(CG)
    layers = list(topological_generations(CG))
    for i, vs in enumerate(layers):
        for v in vs:
            DG.nodes[v]["layer"] = i
    for u in DG.nodes():
        if "layer" not in DG.nodes[u]:
            DG.nodes[u]["layer"] = len(layers) - 1 - correct_dist(DG, u)
    


def upper_lower_bounds(CG):
    print("Upper bound", evaluate(CG))
    CG_copy = CG.copy()
    print(CG_copy.edges())
    for u, v in CG.edges():
        if is_blockable(CG, u, v):
            CG_copy.remove_edge(u, v)
    lb = evaluate(CG_copy)
    print(CG_copy.edges())
    print("Lower bound", lb)
    return lb

def draw_networkx(G, name):
    color_map = []
    for node in G:
        if node in G.graph["starting_nodes"]:
            color_map.append('red')
        elif node  == G.graph["DA"]:
            color_map.append("green")
        else: 
            color_map.append('blue') 
    nx.draw(G,node_color=color_map, with_labels = True)
    import matplotlib.pyplot as plt
    plt.savefig(name + ".png")
    plt.clf()

def draw_hist(x, name):
    import matplotlib.pyplot as plt
    plt.hist(x, bins = len(x))
    plt.savefig(name + ".png")
    plt.clf()