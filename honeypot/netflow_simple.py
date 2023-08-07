from argparse import ArgumentParser
from itertools import permutations
import json
from multiprocessing.dummy import current_process
import tarfile
from threading import currentThread
from tracemalloc import start
from turtle import update
from setupgraph import random_setup, flow_graph_setup
from trivialFPT import trivialFPT, trivialFPT_helper
from greedy import greedy, greedy_v2
from utility import upper_lower_bounds, report
from treedecomposition import dp
from classification import all_classifications
import timeit
from networkx.readwrite.gpickle import write_gpickle, read_gpickle
from gnn import run_gnn
import networkx as nx
# "sample.gpickle",
# "sample-dag.gpickle",
# "r100.gpickle",
# "r100-dag.gpickle",
# "r200.gpickle",
# "r200-dag.gpickle",
# "r500.gpickle",
# "r500-dag.gpickle",
# "r1000.gpickle",
# "r1000-dag.gpickle",
# "r2000.gpickle",
# "r2000-dag.gpickle",

def parse_args():
    parser = ArgumentParser(prog="ShotHound", prefix_chars="-/", add_help=False, description=f'Finding practical paths in BloodHound')
    parser.add_argument('budget', type = int)
    parser.add_argument('start', type = int)
    parser.add_argument('seed', type = int)
    args = parser.parse_args()
    return args

args_input = parse_args()
args = {
    # "fn": "examplegraph_1",
    "fn": "r2000_alledge",
    "budget": args_input.budget,
    "start_node_number": args_input.start,
    "seed": args_input.seed,
    "blockable_p": 1,   # blockable_p = 1 means that all are blockable
    "double_edge_p": 0, # if double_edge = 0, no double edge appear
    "multi_block_p": 1,
}


def recursive_update(path):
    current_node = path[0]
    # Lets find out how many shortest path deviated path starting from this node
    # deviated_nodes = []
    deviated_paths = []
    CG_copy = CG.copy()
    deviated_paths.append(path)
    CG_copy.remove_edge(path[0], path[1])
    # print(path)
    while True:
        try: 
            new_path = nx.shortest_path(CG_copy, source=current_node, target=DA)
        except nx.exception.NetworkXNoPath:
            break
        if len(new_path) > len(path):
            break
        if new_path[1] != path[1]: # only consider path that start to deviatated at this node
            deviated_paths.append(new_path)
            CG_copy.remove_edge(new_path[0], new_path[1])
    # print(deviated_paths)

    # if current_node in combining_nodes:
    # receive propagated information from predecessor edges to current nodes
    # if CG.nodes[current_node]["node_type"] != "S":
        # sum over all predecessor edge
    if current_node not in starting_nodes:
        CG.nodes[current_node]["chance"] = 0 # normal node
    else:
        CG.nodes[current_node]["chance"] = 1/(len(starting_nodes)) # entry node 
    for v in list(CG.predecessors(current_node)):
        CG.nodes[current_node]["chance"] += CG[v][current_node]["chance"]
    # print(CG.nodes[current_node]["chance"])
# propagate chance information from node to edges
    for v in deviated_paths:
        CG[current_node][v[1]]["chance"] = CG.nodes[current_node]["chance"]/len(deviated_paths)
    # elif current_node in spliting_nodes:
    # else:
        # a straight path 
    
    if path[1] == DA:
        return
    for depath in deviated_paths:
        recursive_update(depath[1:])
    return


def update_chance(G, starting_nodes):
    for entry in starting_nodes:
        shortest_path = nx.shortest_path(G, source=entry, target=DA)
        recursive_update(shortest_path)
    return
def draw_networkx(G, name):
    nx.draw(G, with_labels = True)
    import matplotlib.pyplot as plt
    plt.savefig(name + ".png")
    plt.clf()

def draw_hist(x, name):
    import matplotlib.pyplot as plt
    plt.hist(x, bins = len(x))
    plt.savefig(name + ".png")
    plt.clf()
if __name__ == "__main__":
    global spliting_nodes, combining_nodes, CG, DA
    # args["seed"] = 2
    CG = flow_graph_setup(**args)
    print("done SETUP-------------")
    report(CG)
    spliting_nodes = {v:CG.out_degree(v) for v in CG.nodes() if (CG.out_degree(v) > 1)}
    combining_nodes = {v:CG.out_degree(v) for v in CG.nodes() if (CG.in_degree(v) > 1)}
    print(spliting_nodes)
    starting_nodes = []
    DA = CG.graph["DA"]
    draw_networkx(CG, "CG_example_netflow")
    # initialise the chance of getting to edge and nodes to 0

    # predecessor(u) return nodes that being pointed by an edge from node u
    # succcessor(u) return nodes that have edge point to node u
    for v in CG.nodes():
        CG.nodes[v]["chance"] = 0
        if CG.nodes[v]["node_type"] == "S":
            starting_nodes.append(v)
            CG.nodes[v]["chance"] = 1/(len(starting_nodes))
            temp = nx.shortest_path(CG, source=v, target=DA)
    for u, v in CG.edges():
        CG[u][v]["chance"] = 0 

    update_chance(CG, starting_nodes=starting_nodes)
    for v in list(CG.predecessors(DA)):
        CG.nodes[DA]["chance"] += CG[v][DA]["chance"]
        # print(CG[v][DA]["chance"])
    new_graph = nx.DiGraph()
    for u, v in CG.edges():
        # print("edges:", u,v, "chance: ", CG[u][v]["chance"])
        if CG[u][v]["chance"] != 0:
            new_graph.add_edge(u,v)
        if v == DA:
            print("edges:", u,v, "chance: ", CG[u][v]["chance"])
    print(CG.nodes[DA]["chance"])

    draw_networkx(new_graph, "reduced_example_netflow")
    
    sp_hist = []
    for i in range(len(starting_nodes)):
        shortest_path = nx.shortest_path(CG, source=starting_nodes[i], target=DA)
        sp_hist.append(len(shortest_path))
    draw_hist(sp_hist, "shortest_path_hist")
    
    assert CG.nodes[DA]["chance"] == 1



    # for entry_node in CG.nodes():
    #     if CG.nodes[entry_node]["node_type"] == "S":
    #         starting_nodes.append(entry_node)
    #         CG.nodes[entry_node]["chance"] = 1/(args["start_node_number"])
    #         print(CG.nodes[entry_node]["chance"])
    #         shortest_path = nx.shortest_path(CG, source=entry_node, target=DA)
    #         print(shortest_path)
    #         for v in range(shortest_path):
    #             # propagate the chance to others nodes
    #             CG_copy = CG.copy()
    #             predecessor = []
    #             count = 0
    #             while True:
    #                 try:
    #                     list_path = nx.shortest_path(CG_copy, source=v, target=DA)
    #                     # print(list_path)
    #                 except nx.exception.NetworkXNoPath:
    #                     break
    #                 # if there is path with same length to the original path, included it to predecessor
    #                 if count == 0:
    #                     max_len = len(list_path)
    #                     count += 1
    #                 ###################
    #                 if len(list_path) != 0 and len(list_path) == max_len:
    #                     predecessor.append(list_path[1])
    #                     print(predecessor)
    #                     print(list_path)
    #                     if list_path[1] == DA:
    #                         print("heeae")
    #                         break
    #                     CG_copy.remove_edge(list_path[0], list_path[1])
    #                 else:
    #                     break
    #                 # check for feasible predecessor ()
    #             if v != entry_node:
    #                 # propagate to node
    #                 for x in list(CG.successors(v)):
    #                     CG.nodes[v]["chance"] += CG[x][v]["chance"]
    #                 # propagate out nodes
    #             for x in list(CG.predecessors(v)):
    #                 if x in predecessor:
    #                     CG[v][x]["chance"] = CG.nodes[v]["chance"]/(len(predecessor))




