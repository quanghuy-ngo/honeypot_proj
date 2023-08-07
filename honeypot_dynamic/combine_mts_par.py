
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
import os

def parse_args():
    parser = ArgumentParser(prog="ShotHound", prefix_chars="-/", add_help=False, description=f'Finding practical paths in BloodHound')
    parser.add_argument('--blocking_dir', type = str)
    parser.add_argument('--mts_sampling', type = str)
    parser.add_argument('--mts_graph_sample', type = int)
    parser.add_argument('--mts_graph_sample_total', type = int)
    args = parser.parse_args()
    return args
args_input = parse_args()

number_of_runs = 1
args = {
    # "fn": "examplegraph",
    # "fn": "adsim05",
    "mts_sampling" : args_input.mts_sampling,
    "blocking_dir" : args_input.blocking_dir,
    "mts_graph_sample" : args_input.mts_graph_sample,
    "mts_graph_sample_total" : args_input.mts_graph_sample_total,
}


if __name__ == "__main__":
    print(args["blocking_dir"]) 
    print(args)
    score_list_flat = []
    score_list_competent = []
    score_list_lb_competent = []
    score_list_lb_flat = []
    thread_number = int(args["mts_graph_sample_total"]/args["mts_graph_sample"])
    for i in range(thread_number):
        dir = "/gpfs/users/a1798528/honeypot_dynamic_HPC/mts_par/" + args["blocking_dir"] + "_" + args["mts_sampling"] + "_" + str(args["mts_graph_sample"]) + "_" + str(i) + ".pkl" 
        with open(dir, 'rb') as f:
            score_list = pickle.load(f)
        score_list_competent.append(score_list["competent"])
        score_list_flat.append(score_list["flat"])
        score_list_lb_competent.append(score_list["lb_competent"])
        score_list_lb_flat.append(score_list["lb_flat"])
        os.remove(dir)

    save_list = (score_list_flat, score_list_competent, score_list_lb_competent, score_list_lb_flat)
        
    save_dir = "/gpfs/users/a1798528/honeypot_dynamic_HPC/mts_par/" + args["blocking_dir"] + "_" + args["mts_sampling"] + "_" + str(args["mts_graph_sample"]) + ".pkl" 
    with open(save_dir, 'wb') as f:
        pickle.dump(save_list, f)
    
    print("Average Flat: ", sum(score_list_flat) / len(score_list_flat))
    print("Variance: ",np.var(score_list_flat))
    print("STD: ", np.std(score_list_flat))
    
    print("Average Competent: ", sum(score_list_competent) / len(score_list_competent))
    print("Variance: ",np.var(score_list_competent))
    print("STD: ", np.std(score_list_competent))
