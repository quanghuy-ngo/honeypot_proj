
#t so simple now :((
from argparse import ArgumentParser
from ast import arg
import json
from linecache import lazycache
from setupgraph import get_shortest_graph_in_place, get_spath_graph_2, dynamic_graph_setup, flip_edge, normal_hassession_prob, generate_hassession, binomial_hassession_prob, dynamic_graph_setup_inplace
from netflow_simple_lp import mip_flow, mip_flow_2, mip_dygraph, mip_flow_phiattack
from utility import report, evaluate_flow, draw_networkx, remove_dead_nodes
import timeit
from networkx.readwrite.gpickle import write_gpickle, read_gpickle
from timeout import timeout
import copy
import pickle
import numpy as np
from pulp import *
import random
import networkx as nx
from networkx.exception import NetworkXNoPath

#python3 mip_feature.py --fn dyadsimx05_all --budget 20 --start -1 --blockable -2 --sampling_type normal --seed 0 --thread 0 --graph_number 100 --algo mixed
def parse_args():
    parser = ArgumentParser(prog="ShotHound", prefix_chars="-/", add_help=False, description=f'Finding practical paths in BloodHound')
    parser.add_argument('--budget', type = int)
    parser.add_argument('--start', type = int)
    parser.add_argument('--fn', type = str)
    parser.add_argument('--blockable_p', type = float)
    parser.add_argument('--sampling_type', type = str)
    parser.add_argument('--seed', type = int)
    parser.add_argument('--thread', type = int)
    parser.add_argument('--graph_number', type = int)
    parser.add_argument('--graph_number_total', type = int)
    parser.add_argument('--algo', type = str)
    args = parser.parse_args()
    return args
args_input = parse_args()


number_of_runs = 1
args = {
    "fn" : args_input.fn,
    "budget": args_input.budget,
    "start": args_input.start,
    "thread": args_input.thread,
    "graph_number": args_input.graph_number,
    "graph_number_total": args_input.graph_number_total,
    "sampling_type": args_input.sampling_type,
    "blockable_p": args_input.blockable_p,
    "seed": args_input.seed,
    "algo": args_input.algo
}

def evaluate_flow_feature_competent(G, block_node_list):
    node_to_feature_id = G.graph["node_to_feature_id"]
    starting_nodes = G.graph["starting_nodes"]
    # print(len(starting_nodes))
    feature_list = [0 for _ in range(len(node_to_feature_id))]
    DA = G.graph["DA"]
    score = 0
    G_blocked = copy.deepcopy(G)
    G_blocked.remove_nodes_from(block_node_list)

    for i in range(len(starting_nodes)):
        try: 
            path = nx.shortest_path(G_blocked, source=starting_nodes[i], target=DA)
            feature_list[node_to_feature_id[starting_nodes[i]]] += 1/len(starting_nodes)
            score += 1/len(starting_nodes)
        except NetworkXNoPath:
            continue
            
    return score, feature_list
def evaluate_flow_feature_flat(G, block_node_list):
    '''heuristic 2 based on the observation that some starting nodes have couple of thousand of 
    path to DA while the majority of nodes only have < 10 path to DA. Heuristic 2 
    get maximum 20 path for starting nodes have more than 100 
    '''

    path_update = G.graph["path_all"]
    path_number_dict = G.graph["path_number_dict"]
    starting_nodes = G.graph["starting_nodes"]
    # nodes_in_use = G.graph["nodes_in_use"]
    node_to_feature_id = G.graph["node_to_feature_id"]
    score = 1
    # print("sdadassadsa")
    # print(len(path_update))
    count_path = {u : 0 for u in starting_nodes}
    score_dict = {u : 0 for u in starting_nodes} # dictionary of score of every starting nodes
    feature_list = [0 for _ in range(len(node_to_feature_id))]
    
    count = 0
    for n in path_number_dict:
        count += path_number_dict[n]
    assert count == len(path_update)

    for block_node in block_node_list:
        path_temp = []
        
        for path in path_update:
            # if block node in a path, delete that path.
            if path[0] not in starting_nodes:
                continue
            # if count_path[path[0]] >= settings.max_feature_spath:
            #     continue

            if block_node in path:
                score += (1/(len(starting_nodes)))*(1/path_number_dict[path[0]])
                count_path[path[0]] += 1
                feature_list[node_to_feature_id[path[0]]] += (1/(len(starting_nodes)))*(1/path_number_dict[path[0]])
            else:                 
                path_temp.append(path)
        path_update = path_temp
    # pprint(count_path)
    # pprint(path_number_dict)

    return score, feature_list

def vote_blocking(budget, block_strategy):
    # ranking the nodes in block strategy
    vote = dict()
    for strat in block_strategy:
        for i in strat:
            if i not in vote:
                vote[i] = 0
            else:
                vote[i] = vote[i] + 1

    return sorted(vote, key=vote.get, reverse=True)[:budget]



def dyMIP_feature(graph, hasmask, type):
    '''This function extract graph feature of MIP(1)'''
    blocking_list = []
    average_score_list = []
    feature_list = []
    #start = (graph_number)*thread
    #end = start + (graph_number)

    for i in range(len(hasmask)):
        print("Batch: ", i)
        CG_sample = flip_edge(graph, hasmask[i])
        graph_original = get_shortest_graph_in_place(CG_sample)
        if type == "flat":
            blocking_strat = mip_flow_phiattack(graph_original, 0)
            _, feature = evaluate_flow_feature_flat(graph_original, blocking_strat)
        elif type == "competent":   
            blocking_strat = mip_flow_phiattack(graph_original, 1)
            _, feature = evaluate_flow_feature_competent(graph_original, blocking_strat)
        elif type == "mixed":
            blocking_strat = mip_flow_phiattack(graph_original, 0.5)
            _, feature_flat = evaluate_flow_feature_flat(graph_original, blocking_strat)
            _, feature_competent = evaluate_flow_feature_competent(graph_original, blocking_strat)
            feature = feature_flat + feature_competent

        feature_list.append(feature)
    return average_score_list, blocking_list, feature_list

# def dyMIP_feature_2(graph, hasmask):
#     '''This function extract graph feature of MIP(1) with applying voting algorithm'''
#     blocking_list = []
#     average_score_list = []
#     feature_list = []
#     #start = (graph_number)*thread
#     #end = start + (graph_number)
#     blocking_list = []
#     for i in range(len(hasmask)):
#         print("Batch: ", i)
#         CG_sample = flip_edge(graph, hasmask[i])
#         graph_condensed = get_spath_graph_2(CG_sample)
#         average_score, blocking_strat = mip_dygraph([graph_condensed])
#         blocking_list.append(blocking_strat)
#         average_score_list.append(average_score)

#     vote_block = vote_blocking(graph.graph["budget"], blocking_list)
#     for i in range(start, end):
#         _, feature = evaluate_flow_feature_flat(graph_condensed, vote_block)
#         feature_list.append(feature)

#     return average_score_list, blocking_list, feature_list


if __name__ == "__main__":
    global output
    # if args["start_node_number"] == -1:
    #     number_of_runs = 1
    print(args)


    # graph_dir = "/hpcfs/users/a1798528/honeypot_dynamic_HPC/processed_dygraph/" + args["fn"] + ".gpickle" # dyadsim025_all_10000_0.gpickle
    # CG = read_gpickle(graph_dir)

    fn, budget, start_node_number, thread, graph_number, graph_number_total, sampling_type, blockable_p, seed, algo = args.values()
    CG = dynamic_graph_setup_inplace(fn, seed, start_node_number, blockable_p, budget, False)
    CG.graph["budget"] = args["budget"]




    block_strategy_mip = []
    block_strategy_greedy = []
    score_list = []
    start_time = timeit.default_timer()

    hasmask = None
    start = (args["graph_number"])*args["thread"]
    end = start + (args["graph_number"])
    hasmask = None
    if args["sampling_type"] == "normal":
        p = normal_hassession_prob(CG)
        data = generate_hassession(graph_number_total, p, CG.graph["hassession_to_idx"], seed )
        hasmask = data[start: end]
    elif args["sampling_type"] == "binomial":

        prob = binomial_hassession_prob(CG)
        p = prob
        data = generate_hassession(graph_number_total, p, CG.graph["hassession_to_idx"], seed )
        hasmask = data[start: end]
    elif args["sampling_type"] == "lb":
        hasmask_dir = "/Users/huyngo/Desktop/Research/honeypot_dynamic/processed_dygraph/" + args["fn"] + "_hasmask_mts" + ".pkl"
        with open(hasmask_dir, 'rb') as f:
            data = pickle.load(f)
        hasmask = data["sample_mts"]
        args["graph_sample_number"] = len(hasmask[0])
        args["seed"] = 0
    data = []
    possible_starting_nodes = []


    starting_nodes = CG.graph["starting_nodes"]
    print(len(starting_nodes))
    CG.graph["node_to_feature_id"] = {starting_nodes[i] : i for i in range(len(starting_nodes))}
    print(len(CG.graph["node_to_feature_id"]))
    data = []
    print(len(hasmask))
    average_score_list, blocking_list, feature_list = dyMIP_feature(CG, hasmask, args["algo"])
    time = timeit.default_timer() - start_time


    print("Average Score batches: ")

    print(time)
    save_dict = dict()
    save_dict["graph"] = CG
    save_dict["features"] = feature_list
    pkl_dir = "/gpfs/users/a1798528/honeypot_dynamic_HPC/dymip_feature/feature_" + str(args['fn']) + "_"  + str(args['budget']) + "_" + str(args["start"]) + "_" + str(args["blockable_p"])+ "_" + str(args["sampling_type"]) + "_" + str(args["graph_number"]) + "_" +  str(args["algo"]) + "_" + str(args["seed"]) + "_" + str(args["thread"]) + ".pkl"
    

    
    
    f = open(pkl_dir,"wb")
    pickle.dump(save_dict,f)
    f.close()



