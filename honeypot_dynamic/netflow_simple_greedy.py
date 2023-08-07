from argparse import ArgumentParser
# import grp
from importlib.resources import path
# from itertools import permutations
# import json
from multiprocessing.dummy import current_process
# import tarfile
# from threading import currentThread
# from tracemalloc import start
# from turtle import update
# from typing import final
from setupgraph import random_setup, flow_graph_setup, get_spath_graph
from utility import upper_lower_bounds, report, draw_networkx, draw_hist, evaluate_flow
import timeit
from networkx.readwrite.gpickle import write_gpickle, read_gpickle
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

# def parse_args():
#     parser = ArgumentParser(prog="ShotHound", prefix_chars="-/", add_help=False, description=f'Finding practical paths in BloodHound')
#     parser.add_argument('budget', type = int)
#     parser.add_argument('start', type = int)
#     parser.add_argument('seed', type = int)
#     args = parser.parse_args()
#     return args

# args_input = parse_args()
# args = {
#     # "fn": "examplegraph_2",
#     "fn": "r2000",
#     "budget": args_input.budget,
#     "start_node_number": args_input.start,
#     "seed": args_input.seed,
#     "blockable_p": 1,   # blockable_p = 1 means that all are blockable
#     "double_edge_p": 0, # if double_edge = 0, no double edge appear
#     # "cross_path_p": 0,
#     "multi_block_p": 1,
# }


# def recursive_update(path):
#     current_node = path[0]
#     # Lets find out how many shortest path deviated path starting from this node
#     # deviated_nodes = []
#     deviated_paths = []
#     CG_copy = CG.copy()
#     deviated_paths.append(path)
#     CG_copy.remove_edge(path[0], path[1])
#     # print(path)
#     while True:
#         try: 
#             new_path = nx.shortest_path(CG_copy, source=current_node, target=DA)
#         except nx.exception.NetworkXNoPath:
#             break
#         if len(new_path) > len(path):
#             break
#         if new_path[1] != path[1]: # only consider path that start to deviatated at this node
#             deviated_paths.append(new_path)
#             CG_copy.remove_edge(new_path[0], new_path[1])
#     if current_node not in starting_nodes:
#         CG.nodes[current_node]["chance"] = 0 # normal node
#     else:
#         CG.nodes[current_node]["chance"] = 1/(len(starting_nodes)) # entry node 
#     for v in list(CG.predecessors(current_node)):
#         CG.nodes[current_node]["chance"] += CG[v][current_node]["chance"]
#     # print(CG.nodes[current_node]["chance"])
# # propagate chance information from node to edges
#     for v in deviated_paths:
#         CG[current_node][v[1]]["chance"] = CG.nodes[current_node]["chance"]/len(deviated_paths)
#     # elif current_node in spliting_nodes:
#     # else:
#         # a straight path 
    
#     if path[1] == DA:
#         return
#     for depath in deviated_paths:
#         recursive_update(depath[1:])
#     return


def update_chance(G, starting_nodes):
    for entry in starting_nodes:
        shortest_path = nx.shortest_path(G, source=entry, target=DA)
        recursive_update(shortest_path)
    return



def extract_edge(path_list):
    edge_list = []
    for path in path_list:
        for i in range(len(path)):
            if i == len(path)-1:
                break
            edge_list.append((path[i], path[i+1]))
    return edge_list



def greedy_node_v1(graph_condensed):
    path_all = graph_condensed.graph["path_all"]
    path_number_dict = graph_condensed.graph["path_number_dict"]
    starting_nodes = graph_condensed.graph["starting_nodes"]
    budget = graph_condensed.graph["budget"]
    DA = graph_condensed.graph["DA"]
    # report(graph_condensed)
    for v in graph_condensed.nodes():
        graph_condensed.nodes[v]["chance"] = 0

    for node in graph_condensed.nodes():
        for path in path_all:
            if node in path:
                graph_condensed.nodes[node]["chance"] += 1/len(starting_nodes)*(1/path_number_dict[path[0]])
    # start to eliminate nodes

    list_node_chance = []
    for node in graph_condensed.nodes():
        list_node_chance.append((node, graph_condensed.nodes[node]["chance"]))
    list_node_chance.sort(key=lambda i:i[-1],reverse=True)


    score = 1
    block_node_list = []
    for i in list_node_chance:
        node = i[0]
        block_node_list_temp = block_node_list.copy()
        if budget == 0:
            break
        if node != DA and node not in starting_nodes and graph_condensed.nodes[node]["blockable"] == True:
            block_node_list_temp.append(node)
            temp_score = evaluate_flow(graph_condensed, block_node_list_temp)
            # print(temp_score)
            if temp_score < score:
                block_node_list.append(node)
                score = temp_score
                # path_remain = temp_path
                budget = budget - 1

    final_score = evaluate_flow(graph_condensed, block_node_list)
    print("score: ", final_score)
    
    print("budget remain:", budget)
    print(block_node_list)
    # draw_networkx(graph_condensed, "reduced_example_netflow") 
    return final_score, block_node_list

def greedy_node_v2(graph_condensed):
    path_all = graph_condensed.graph["path_all"]
    path_number_dict = graph_condensed.graph["path_number_dict"]
    starting_nodes = graph_condensed.graph["starting_nodes"]
    budget = graph_condensed.graph["budget"]
    DA = graph_condensed.graph["DA"]
    # report(graph_condensed)
    graph_condensed_copy = graph_condensed.copy()

    old_budget = budget+1
    block_node_list = []
    path_remain = []

    # print(list(graph_condensed.nodes()))
    while old_budget != budget and budget != 0:
        old_budget = budget
        for v in graph_condensed_copy.nodes():
            graph_condensed_copy.nodes[v]["chance"] = 0
        # path_number_dict = dict()
        path_all = []
        for entry in starting_nodes:
            # print([p for p in nx.all_shortest_paths(CG,source=entry,target=DA)])
            try:
                temp = list(nx.all_shortest_paths(graph_condensed_copy,source=entry,target=DA))
                # path_number_dict[entry] = len(temp)
                path_all = path_all + temp
            except:
                continue

        for node in graph_condensed_copy.nodes():
            for path in path_all:
                if node in path:
                    graph_condensed_copy.nodes[node]["chance"] += 1/len(starting_nodes)*(1/path_number_dict[path[0]])
        # start to eliminate nodes

        list_node_chance = []
        for node in graph_condensed_copy.nodes():
            list_node_chance.append((node, graph_condensed_copy.nodes[node]["chance"]))
        list_node_chance.sort(key=lambda i:i[-1],reverse=True)
        # score = evaluate(graph_condensed, block_node_list_temp[])
        for i in list_node_chance:
            node = i[0]
            if budget == 0:
                break
            if node != DA and node not in starting_nodes and graph_condensed_copy.nodes[node]["blockable"] == True:

                block_node_list.append(node)
                # path_remain = temp_path

                budget = budget - 1
                graph_condensed_copy.remove_node(block_node_list[-1])
                break
        
            

    # final_score = evaluate_flow(graph_condensed, block_node_list)
    # print("score: ", final_score)
    
    # print("budget remain:", budget)
    # print(block_node_list)
    # draw_networkx(graph_condensed, "reduced_example_netflow") 
    return block_node_list


def greedy_node_v3(graph_condensed):
    path_all = graph_condensed.graph["path_all"]
    path_number_dict = graph_condensed.graph["path_number_dict"]
    starting_nodes = graph_condensed.graph["starting_nodes"]
    budget = graph_condensed.graph["budget"]
    DA = graph_condensed.graph["DA"]
    # report(graph_condensed)
    graph_condensed_copy = graph_condensed.copy()

    old_budget = budget+1
    block_node_list = []
    path_remain = []
    # print(list(graph_condensed.nodes()))
    while old_budget != budget and budget != 0:
        for v in graph_condensed_copy.nodes():
            graph_condensed_copy.nodes[v]["chance"] = 0

        path_number_dict = dict()
        path_all = []
        for entry in starting_nodes:
            # print([p for p in nx.all_shortest_paths(CG,source=entry,target=DA)])
            try:
                temp = list(nx.all_shortest_paths(graph_condensed_copy,source=entry,target=DA))
                # path_number_dict[entry] = len(temp)
                path_all = path_all + temp
            except:
                continue

        for node in graph_condensed_copy.nodes():
            for path in path_all:
                if node in path:
                    graph_condensed_copy.nodes[node]["chance"] += 1/len(starting_nodes)*(1/path_number_dict[path[0]])
        # start to eliminate nodes

        list_node_chance = []
        for node in graph_condensed_copy.nodes():
            list_node_chance.append((node, graph_condensed_copy.nodes[node]["chance"]))
        list_node_chance.sort(key=lambda i:i[-1],reverse=True)


        # score = evaluate(graph_condensed, block_node_list_temp[])
        for i in list_node_chance:
            node = i[0]
            if budget == 0:
                break
            if node != DA and node not in starting_nodes and graph_condensed_copy.nodes[node]["blockable"] == True:

                block_node_list.append(node)
                # path_remain = temp_path
                old_budget = budget
                budget = budget - 1
                graph_condensed_copy.remove_node(block_node_list[-1])
                break
        
            

    final_score = evaluate_flow(graph_condensed, block_node_list)
    print("score: ", final_score)
    
    print("budget remain:", budget)
    print(block_node_list)
    # draw_networkx(graph_condensed, "reduced_example_netflow") 
    return final_score, block_node_list


# if __name__ == "__main__":
#     global spliting_nodes, combining_nodes, CG, DA, path_number_dict, starting_nodes
#     # args["seed"] = 2
#     CG = flow_graph_setup(**args)

#     # CG = read_gpickle("/home/andrewngo/Desktop/CFR/Code/AAAI/examplegraph_1.gpickle")

#     print("done SETUP-------------")
#     report(CG)
#     graph_condensed = get_spath_graph(CG)


#     print("------start greedy------")
#     greedy_node_v1(graph_condensed)


    
    




