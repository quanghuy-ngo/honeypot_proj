#netflow not so simple now :((
from argparse import ArgumentParser
from ast import arg
import json
from linecache import lazycache
from setupgraph import flow_graph_setup, get_spath_graph, dynamic_graph_setup, flip_edge, temporal_graph_setup, merge_graph_setup
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


def parse_args():
    parser = ArgumentParser(prog="ShotHound", prefix_chars="-/", add_help=False, description=f'Finding practical paths in BloodHound')
    parser.add_argument('budget', type = int)
    parser.add_argument('start', type = int)
    # parser.add_argument('batch_number', type = int)
    parser.add_argument('fn', type = str)
    # parser.add_argument('interarrival_mean', type = int)
    # parser.add_argument('new_edge_rate_mean', type = int)
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
    # 'day_of_data': 400,
    # 'interarrival_mean': args_input.interarrival_mean,
    # 'new_edge_rate_mean': args_input.new_edge_rate_mean, 
    "seed": 1,
    "no_one_hop": False
}



if __name__ == "__main__":
    global output
    print(args)
    df = pd.DataFrame()
    # CG, auth, snapshot_list  = temporal_graph_setup(**args)
    CG, auth, snapshot_list  = merge_graph_setup(**args)
    seed = args["seed"]
    graph_name = args["fn"]
    # day_of_data = args['day_of_data']
    # interarrival_mean = args["interarrival_mean"]
    # new_edge_rate_mean = args['new_edge_rate_mean']
    with open(f"//Users/huyngo/Desktop/Research/honeypot_dynamic/processed_dygraph/m_auth_{graph_name}.pickle", "wb") as fp:   #Pickling
        pickle.dump(auth, fp)
    write_gpickle(CG, f"//Users/huyngo/Desktop/Research/honeypot_dynamic/processed_dygraph/m_temporal_{graph_name}.gpickle")
        
