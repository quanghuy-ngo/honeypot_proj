from argparse import ArgumentParser
from itertools import permutations
# import json
from setupgraph import random_setup, flow_graph_setup, get_spath_graph,  flip_edge
# from netflow_simple_greedy import greedy_node_v1, greedy_node_v2
from netflow_simple_lp import mip_flow, mip_flow_2, mip_dygraph
from netflow_simple_TD_v2 import tree_width
from utility import upper_lower_bounds, report, evaluate_flow, draw_networkx
import timeit
from networkx.readwrite.gpickle import write_gpickle, read_gpickle
from timeout import timeout
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle


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
    parser.add_argument('fn', type = str)
    args = parser.parse_args()

    return args
args_input = parse_args()

number_of_runs = 1
args = {
    # "fn": "examplegraph",
    # "fn": "adsim05",
    "fn" : args_input.fn,
    "budget": args_input.budget,
    "start_node_number": args_input.start,
    "blockable_p": 0.8,
    "double_edge_p": 0,
    "multi_block_p": 1,
    "cross_path_p": 0,
    "no_one_hop": False,
}

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


def dyMIP(graph, start_snapshot, mip_num, train_num):
    blocking_list = []
    average_score_list = []
    for batch in range(int(train_num/mip_num)):
        print("Batch: ", batch)
        graph_batch = []
        i = 0
        
        while(i < mip_num):
        # for i in range(batch, batch+mip_num):
            graph_idx = start_snapshot + i + batch
            print(sum(graph.graph["hassession_mask"][graph_idx]))
            CG_sample = flip_edge(graph, graph.graph["hassession_mask"][graph_idx])
            graph_condensed = get_spath_graph(CG_sample, args["cross_path_p"])
            print(len(graph_condensed.nodes()))
            if len(graph_condensed.nodes()) > 10:
                print("------------done SETUP graph_condensed-------------")
                graph_batch.append(graph_condensed)
                i += 1
            else:
                print("Bad Graph Sample")

        average_score, blocking_strat = mip_dygraph(graph_batch)
        average_score_list.append(average_score)
        blocking_list.append(blocking_strat)
    return average_score_list, blocking_list


# def clean_snapshot(CG):
#     snapshot = CG.graph["hassesssion_mask"]
#     for i in range(len(snapshot)):

def train(CG, start_snapshot, mip_num, train_num):
    # graph_list = CG.graph["hassession_mask"][start, end]
    score_list, blocking_list = dyMIP(CG, start_snapshot, mip_num,  train_num)
    blocking_strat = vote_blocking(args["budget"], blocking_list)
    return score_list, blocking_strat

def test(CG, start, end, blocking_strat):
    test_mask = CG.graph["hassession_mask"][start:end]
    num_iter = end - start
    score_list = []
    
    for i in tqdm(range(0,num_iter)):
        CG_sample = flip_edge(CG, test_mask[i])
        graph_condensed = get_spath_graph(CG_sample, 0)
        if len(graph_condensed.nodes()) > 10:
            score = evaluate_flow(graph_condensed, blocking_strat)
            score_list.append(score)
            if i % (10) == 0:
                print("###########################")
                print(f"monte carlo iteration {i}")
                print("Current Score: ", score)
    return score_list

if __name__ == "__main__":
    graph_dir = "/Users/huyngo/Desktop/Research/honeypot_dynamic/processed_dygraph/" + args["fn"] + ".gpickle"
    CG = read_gpickle(graph_dir)
    # print(len(CG.graph["hassession_mask"]))
    #skip 100 first snapshot to stablise the the log in log off number
    #check if there is condensed graph exist in graph

    print("Training")
    print(len(CG.graph["hassession_mask"]))
    score_list, blocking_strat = train(CG, 100, 24, 24*7)
    
    print("Testing")

    score_list = test(CG, 24*7, len(CG.graph["hassession_mask"]), blocking_strat)
    score_dir = "/Users/huyngo/Desktop/Research/honeypot_dynamic/result/score_" + args["fn"] + str(args["budget"]) + ".pkl"
    with open(score_dir, 'wb') as f:
        pickle.dump(score_list, f)
    
    
    



