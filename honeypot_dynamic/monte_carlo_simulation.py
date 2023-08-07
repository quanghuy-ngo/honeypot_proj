from argparse import ArgumentParser
from ast import arg
import json
from linecache import lazycache
from setupgraph import flow_graph_setup, get_spath_graph, dynamic_graph_setup, flip_edge
from netflow_simple_greedy import greedy_node_v1, greedy_node_v2
from netflow_simple_lp import mip_flow, mip_flow_2, mip_dygraph
from netflow_simple_TD_v2 import dp_flow, tree_width
from utility import report, evaluate_flow, draw_networkx, remove_dead_nodes
import timeit
from networkx.readwrite.gpickle import write_gpickle, read_gpickle
from timeout import timeout
import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm 

def parse_args():
    parser = ArgumentParser(prog="ShotHound", prefix_chars="-/", add_help=False, description=f'Finding practical paths in BloodHound')
    parser.add_argument('blocking_dir', type = str)
    args = parser.parse_args()
    return args
args_input = parse_args()

number_of_runs = 1
args = {
    # "fn": "examplegraph",
    # "fn": "adsim05",
    "blocking_dir" : args_input.blocking_dir,
}


def monte_carlo_simulation(graph, blocking_strategy, monte_carlo_iteration):
    monte_carlo_mask = graph.graph["hassession_mask_monte_carlo"]
    score_list = []

    for i in tqdm(range(monte_carlo_iteration)):
        CG_sample = flip_edge(graph, monte_carlo_mask[i])
        graph_condensed = get_spath_graph(CG_sample, 0)
        score = evaluate_flow(graph_condensed, blocking_strategy)
        score_list.append(score)
        if i % (10) == 0:
            print("###########################")
            print(f"monte carlo iteration {i}")
            print("Average: ", sum(score_list) / len(score_list))
            print("Variance: ",np.var(score_list))
            print("STD: ", np.std(score_list))


if __name__ == "__main__":
    # dir = "/home/quanghuyngo/Desktop/CFR/Code/honeypot_dynamic/dygraph_blocking/dyMIP_dyadsim025_all_10000_10000_1_10_20_0_.pkl"
    dir = "/home/quanghuyngo/Desktop/CFR/Code/honeypot_dynamic/dygraph_blocking/" + args["blocking_dir"] + ".pkl"
    
    with open(dir, 'rb') as f:
        data = pickle.load(f)
    # average_score_list = data["average_score_list"]
    blocking_list = data["blocking_list"]
    # time = data["time"]
    graph = read_gpickle(data["graph_dir"])
    # graph = read_gpickle("/home/quanghuyngo/Desktop/CFR/Code/honeypot_dynamic/processed_dygraph/dyadsim025_all_10000_10000_1.gpickle")
    blocking_strategy = blocking_list[0]
    monte_carlo_simulation(graph, blocking_strategy, monte_carlo_iteration=10000)




# dyMIP_dyadsim025_all_10000_10000_1_10_20_0.pkl

# /hpcfs/users/a1798528/honeypot_dynamic_HPC