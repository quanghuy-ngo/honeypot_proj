from platform import node
from tracemalloc import start
from networkx.readwrite.gpickle import read_gpickle
import random
from utility import is_blockable, is_start, report
import networkx as nx
from patch import topological_generations
import copy
import itertools as it
import copy


def random_setup(
    fn, seed, start_node_number, blockable_p, double_edge_p, multi_block_p, budget
):
    # graph_file_names = [
    #     "r500-dag.gpickle"
    # ]
    graph_file_names = fn + ".gpickle"
    graphs = {}
    graphs[graph_file_names] = read_gpickle("/home/quanghuyngo/Desktop/CFR/Code/honeypot/" + graph_file_names)

    CG = graphs[fn + ".gpickle"].copy()
    random.seed(seed)
    CG.graph["budget"] = budget
    for v in CG.nodes():
        if v == CG.graph["DA"]:
            CG.nodes[v]["node_type"] = "DA"
        else:
            CG.nodes[v]["node_type"] = ""
    # modification in the way of taking entry node
    start_nodes = random.sample(list(CG.nodes()), k=min(start_node_number+1, len(list(CG.nodes()))))
    print(list(CG.nodes()))
    print(start_nodes)
    if CG.graph["DA"] in start_nodes:
        start_nodes.remove(CG.graph["DA"])
    start_nodes = start_nodes[:start_node_number]
    assert len(start_nodes) == start_node_number
    for v in start_nodes:
        CG.nodes[v]["node_type"] = "S"
    double_edge_list = []
    for u, v in CG.edges():
        ran = random.random()
        if ran <= blockable_p:
            if random.random() <= double_edge_p and CG.nodes[v]["node_type"] == "":
                double_edge_list.append((u, v))
            else:
                CG[u][v]["blockable"] = True
        else:
            CG[u][v]["blockable"] = False

    #####################
    for u in CG.nodes():
        if CG.out_degree(u) > 1 and random.random() <= multi_block_p:
            for v in list(CG.successors(u)):
                CG[u][v]["blockable"] = True

    for u, v in double_edge_list:
        a = hash((u, v))
        CG.add_edge(u, a)
        CG.nodes[a]["node_type"] = ""
        CG.nodes[a]["layer"] = CG.nodes[v]["layer"]
        CG[u][v]["blockable"] = True
        CG[u][a]["blockable"] = True
        for x in list(CG.successors(v)):
            CG[v][x]["blockable"] = False
            CG.add_edge(a, x)
            CG[a][x]["blockable"] = False
    preprocess(CG)
    ################################
    # CG[2][0]["blockable"] = False
    # CG[1][0]["blockable"] = False
    ################################
    return CG




def flow_graph_setup(
    fn, seed, start_node_number, blockable_p, double_edge_p, multi_block_p, budget, cross_path_p, no_one_hop
):
    # graph_file_names = [
    #     "r500-dag.gpickle"
    # ]
    graph_file_names = fn + ".gpickle"
    graphs = {}
    graphs[graph_file_names] = read_gpickle("/Users/huyngo/Desktop/Research/honeypot/" + graph_file_names)

    CG = graphs[fn + ".gpickle"].copy()
    random.seed(seed)
    CG.graph["budget"] = budget
    
    for v in CG.nodes():
        if v == CG.graph["DA"]:
            CG.nodes[v]["node_type"] = "DA"
        else:
            CG.nodes[v]["node_type"] = ""
    # modification in the way of taking entry node


    # AAAI-23 paper blockable node experiemtn

    if blockable_p == -1:
        max_hop = 1 
        print(max_hop)
        #find max hop
        for n in CG.nodes():
            if CG.nodes[n]["label"] != "User":
                hop_to_DA = nx.shortest_path_length(CG, source=n, target=CG.graph["DA"])
                if hop_to_DA > max_hop:
                    max_hop = hop_to_DA
        
        for n in CG.nodes():
            if CG.nodes[n]["label"] != "User":
                hop_to_DA = nx.shortest_path_length(CG, source=n, target=CG.graph["DA"])
                ran = random.random()
                p = hop_to_DA/max_hop
                if ran <= p:
                    CG.nodes[n]["blockable"] = True
                else:
                    CG.nodes[n]["blockable"] = False
            else:
                CG.nodes[n]["blockable"] = False
                
    # Can only block computer
    elif blockable_p == -2:
        for n in CG.nodes():
            if CG.nodes[n]["label"] == "Computer":
                CG.nodes[n]["blockable"] = True
            else:
                CG.nodes[n]["blockable"] = False
    else:
        for v in CG.nodes():
            ran = random.random()
            if CG.nodes[v]["label"] != "":
                if ran <= blockable_p:
                    CG.nodes[v]["blockable"] = True
                else:
                    CG.nodes[v]["blockable"] = False


    count = 0
    count_b = 0
    for n in CG.nodes():
        if CG.nodes[n]["blockable"] == False:
            count_b += 1
        count += 1
    # print(count)
    # print(count_b)
    
    # starting node


    start_nodes = []
    if start_node_number == -1:
        for n in CG.nodes():
            if CG.nodes[n]["label"] == "User":
                if no_one_hop == True:
                    hop_to_DA = nx.shortest_path_length(CG, source=n, target=CG.graph["DA"])
                    
                    if hop_to_DA >= 2:
                        start_nodes.append(n)
                else:
                    start_nodes.append(n)
    else:
        user_nodes = []
        for n in CG.nodes():
            if CG.nodes[n]["label"] == "User":
                if no_one_hop == True:
                    hop_to_DA = nx.shortest_path_length(CG, source=n, target=CG.graph["DA"])
                    if hop_to_DA >= 2:
                        user_nodes.append(n)
                else:
                    user_nodes.append(n)
        start_nodes = random.sample(list(user_nodes), k=min(start_node_number+1, len(list(CG.nodes()))))
        # print(list(CG.nodes()))
        if CG.graph["DA"] in start_nodes:
            start_nodes.remove(CG.graph["DA"])
        start_nodes = start_nodes[:start_node_number]
        assert len(start_nodes) == start_node_number
        
    # print("Starting nodes:", start_nodes)
    # print(len(start_nodes))
    # print("DA nodes: ", CG.graph["DA"])
    
    # remove starting nodes that 1 hop away to DA (this is for honeypot allocation on nodes, we basically can not do anything is does not have intermediate nodes)

        
    for v in start_nodes:
        CG.nodes[v]["node_type"] = "S"
    # double_edge_list = []
    # for u, v in CG.edges():
    #     ran = random.random()
    #     if ran <= blockable_p:
    #         if random.random() <= double_edge_p and CG.nodes[v]["node_type"] == "":
    #             double_edge_list.append((u, v))
    #         else:
    #             CG[u][v]["blockable"] = True
    #     else:
    #         CG[u][v]["blockable"] = False


    #assign honey allocatable nodes



    #####################
    # for u in CG.nodes():
    #     if CG.out_degree(u) > 1 and random.random() <= multi_block_p:
    #         for v in list(CG.successors(u)):
    #             CG[u][v]["blockable"] = True


    # create second edge if
    # for u, v in double_edge_list:
    #     # print("herhehreerre")
    #     a = hash((u, v))
    #     print(a)
    #     CG.add_edge(u, a)
    #     CG.nodes[a]["node_type"] = ""
    #     CG.nodes[a]["layer"] = CG.nodes[v]["layer"]
    #     CG[u][v]["blockable"] = True
    #     CG[u][a]["blockable"] = True
    #     for x in list(CG.successors(v)):
    #         CG[v][x]["blockable"] = False
    #         CG.add_edge(a, x)
    #         CG[a][x]["blockable"] = False
    preprocess(CG)
    ################################
    # CG[2][0]["blockable"] = False
    # CG[1][0]["blockable"] = False
    ################################
    return CG

# def get_process_graph(CG):
#     # an alternative for the get_spath_graph with out finding all shortest path graph
#     graph_original = copy.deepcopy(CG)
#     starting_nodes = []
#     for v in CG.nodes():
#         CG.nodes[v]["chance"] = 0
#         if CG.nodes[v]["node_type"] == "S":
#             starting_nodes.append(v)
#     DA = CG.graph["DA"]

#     for v in graph_original.nodes():
#         graph_original.nodes[v]["node_type"] = CG.nodes[v]["node_type"]
#         graph_original.nodes[v]["blockable"] = CG.nodes[v]["blockable"]
#     # for u, v in graph_condensed.edges():
#     #     graph_condensed[u][v]["blockable"] = CG[u][v]["blockable"]
        
#     graph_original.graph["DA"] = CG.graph["DA"]
#     graph_original.graph["budget"] = CG.graph["budget"]
#     graph_original.graph["start_nodes"] = CG.graph["start_nodes"]
#     graph_original.graph["starting_nodes"] = starting_nodes
#     return graph_original




def get_shortest_graph_in_place(CG):
    # an alternative for the get_spath_graph with out finding all shortest path graph
    graph_original = copy.deepcopy(CG)
    starting_nodes = []
    for v in CG.nodes():
        if CG.nodes[v]["node_type"] == "S":
            starting_nodes.append(v)
    DA = CG.graph["DA"]


    path_all = []
    path_number_dict = dict()
    path_dict = dict()
    layers = dict()
    for entry in starting_nodes:
        # print([p for p in nx.all_shortest_paths(CG,source=entry,target=DA)])
        temp = list(nx.all_shortest_paths(CG,source=entry,target=DA))
        path_number_dict[entry] = len(temp)
        path_all = path_all + temp

    edge_list = extract_edge(path_all)
    graph_condensed = nx.DiGraph()
    graph_condensed.add_edges_from(edge_list)


    shortest_nodes = []
    for n in graph_condensed.nodes():
        shortest_nodes.append(n)
    for u, v in graph_original.edges():
        if (u, v) in graph_condensed.edges():
            graph_original[u][v]["shortest_edges"] = True
        else: 
            graph_original[u][v]["shortest_edges"] = False
    
    for v in graph_original.nodes():
        graph_original.nodes[v]["node_type"] = CG.nodes[v]["node_type"]
        graph_original.nodes[v]["blockable"] = CG.nodes[v]["blockable"]
    # for u, v in graph_condensed.edges():
    #     graph_condensed[u][v]["blockable"] = CG[u][v]["blockable"]
        
    graph_original.graph["DA"] = CG.graph["DA"]
    graph_original.graph["budget"] = CG.graph["budget"]
    graph_original.graph["start_nodes"] = CG.graph["start_nodes"]
    graph_original.graph["starting_nodes"] = starting_nodes
    graph_original.graph["path_all"] = path_all
    graph_original.graph["path_number_dict"] = path_number_dict
    graph_original.graph["shortest_nodes"] = shortest_nodes
    return graph_original



def get_spath_graph(CG, cross_path_p):
    # global path_all, DA, path_number_dict, starting_nodes
    # report(CG)
    spliting_nodes = {v:CG.out_degree(v) for v in CG.nodes() if (CG.out_degree(v) > 1)}
    combining_nodes = {v:CG.out_degree(v) for v in CG.nodes() if (CG.in_degree(v) > 1)}

    # print(CG.nodes())
    DA = CG.graph["DA"]
    # draw_networkx(CG, "CG_example_netflow")
    starting_nodes = []
    for v in CG.nodes():
        CG.nodes[v]["chance"] = 0
        if CG.nodes[v]["node_type"] == "S":
            starting_nodes.append(v)

    path_all = []
    path_number_dict = dict()
    path_dict = dict()
    layers = dict()
    for entry in starting_nodes:
        # print([p for p in nx.all_shortest_paths(CG,source=entry,target=DA)])
        temp = list(nx.all_shortest_paths(CG,source=entry,target=DA))
        path_number_dict[entry] = len(temp)
        path_all = path_all + temp
        for path in temp:
            layer_id = 1
            # print(path)
            for i in range(len(path)-1, -1, -1):
                # print(i)
                # print(layer_id)
                if layer_id not in layers:
                    layers[layer_id] = set()
                layers[layer_id].add(path[i])
                layer_id += 1
    
    # print(layers)s
    # print(path_all)
    edge_list = extract_edge(path_all)
    graph_condensed = nx.DiGraph()
    graph_condensed.add_edges_from(edge_list)


    # generate cross path 
    temp = []
    # cross_path_edges = []
    for i in range(len(layers)):
        temp.append(list(layers[len(layers) - i]))
    layers = temp
    # print(temp)
    new_edge = []
    for i in range(len(layers)-2):
        this_layer = list(set(layers[i]) - set(starting_nodes) - set([DA]))
        next_layer = list(set(layers[i+1]) - set(starting_nodes) - set([DA]))
        
        # print(this_layer)
        if len(this_layer) < 2 or len(next_layer) == 0:
            continue
    
        temp_this_layer = copy.deepcopy(this_layer)
        temp_next_layer = copy.deepcopy(next_layer)
        
        for node1, node2 in it.combinations(this_layer, 2):

            node1_child = list(graph_condensed.successors(node1))
            node2_child = [x for x in graph_condensed.successors(node2) if x not in node1_child]
            
            node1_child = list(set(node1_child) - set(starting_nodes) - set([DA]))
            node2_child = list(set(node2_child) - set(starting_nodes) - set([DA]))
            if len(node1_child) == 0 or len(node2_child) == 0 :
                continue
            node3 = random.choice(node1_child)
            node4 = random.choice(node2_child)
            if random.random() <= cross_path_p:

                if graph_condensed.has_edge(node1, node4) == False:
                    graph_condensed.add_edge(node1, node4)
                    new_edge.append((node1, node4))
                    print("Add edge: ", node1, node4)
                if graph_condensed.has_edge(node2, node3) == False:
                    graph_condensed.add_edge(node2, node3)
                    new_edge.append((node2, node3))
                    print("Add edge: ", node2, node3)

    

    # Welp, to be honest, i dont know why finding all shortest path again work, 
    # although it produce the same graph (same node set and edges set),
    # if have time please check it 
    
    path_all = []
    path_number_dict = dict()
    path_dict = dict()
    for entry in starting_nodes:
        # print([p for p in nx.all_shortest_paths(CG,source=entry,target=DA)])
        temp = list(nx.all_shortest_paths(graph_condensed,source=entry,target=DA))
        path_number_dict[entry] = len(temp)
        path_all = path_all + temp
    # print(path_all)
    edge_list = extract_edge(path_all)
    graph_condensed = nx.DiGraph()
    graph_condensed.add_edges_from(edge_list)


    for v in graph_condensed.nodes():
        graph_condensed.nodes[v]["node_type"] = CG.nodes[v]["node_type"]
        graph_condensed.nodes[v]["blockable"] = CG.nodes[v]["blockable"]
    # for u, v in graph_condensed.edges():
    #     graph_condensed[u][v]["blockable"] = CG[u][v]["blockable"]
        
    graph_condensed.graph["DA"] = CG.graph["DA"]
    graph_condensed.graph["budget"] = CG.graph["budget"]
    graph_condensed.graph["start_nodes"] = CG.graph["start_nodes"]
    graph_condensed.graph["path_all"] = path_all
    graph_condensed.graph["path_number_dict"] = path_number_dict
    graph_condensed.graph["starting_nodes"] = starting_nodes
    print("done SETUP-------------")
    # print(report(graph_condensed))
    return graph_condensed


def extract_edge(path_list):
    edge_list = []
    for path in path_list:
        for i in range(len(path)):
            if i == len(path)-1:
                break
            edge_list.append((path[i], path[i+1]))
    return edge_list



def preprocess(CG):
    for u in CG.nodes():
        if CG.out_degree(u) <= 1:
            continue
        unblockable_next = {}
        unblockable_len = {}
        for v in list(CG.successors(u)):
            path = uncached_path_to_split_node_or_DA(CG, u, v)
            path_len = len(path) - 1
            if not is_path_blockable(CG, path):
                if path[-1] not in unblockable_next:
                    unblockable_next[path[-1]] = v
                    unblockable_len[path[-1]] = path_len
                elif path_len < unblockable_len[path[-1]]:
                    unblockable_next[path[-1]] = v
                    unblockable_len[path[-1]] = path_len
        for v in list(CG.successors(u)):
            path = uncached_path_to_split_node_or_DA(CG, u, v)
            path_len = len(path) - 1
            if path[-1] not in unblockable_len:
                continue
            if (
                path_len >= unblockable_len[path[-1]]
                and v != unblockable_next[path[-1]]
            ):
                CG.remove_edge(u, v)

    # has_blockable = False
    # for u, v in CG.edges():
    #     if is_blockable(CG, u, v):
    #         has_blockable = True
    # assert has_blockable

    remove_in_degree_0(CG)

    start_nodes = []
    split_nodes = []
    for u in CG.nodes():
        if CG.out_degree(u) > 1:
            split_nodes.append(u)
        if is_start(CG, u):
            start_nodes.append(u)
    CG.graph["start_nodes"] = tuple(sorted(start_nodes))
    CG.graph["split_nodes"] = tuple(sorted(split_nodes))


def remove_in_degree_0(CG):
    keep_deleting = True
    while keep_deleting:
        keep_deleting = False
        for u in list(CG.nodes()):
            if not is_start(CG, u) and CG.in_degree(u) == 0:
                CG.remove_node(u)
                keep_deleting = True


def uncached_path_to_split_node_or_DA(CG, u, v):
    if CG.out_degree(v) > 1 or v == CG.graph["DA"]:
        return [u, v]
    else:
        future = list(CG.successors(v))
        assert len(future) == 1
        return [u] + uncached_path_to_split_node_or_DA(CG, v, future[0])


def is_path_blockable(CG, path):
    for i in range(len(path) - 1):
        if is_blockable(CG, path[i], path[i + 1]):
            return True
    return False
