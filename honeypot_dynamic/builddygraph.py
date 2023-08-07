#netflow not so simple now :((
from argparse import ArgumentParser
from ast import arg
import json
from linecache import lazycache
from setupgraph import merge_graph_setup, process_company_hassession, mapping_company_dygraph, sample_edge_type, is_path_to
from utility import report, evaluate_flow, draw_networkx, remove_dead_nodes
import timeit
from networkx.readwrite.gpickle import write_gpickle, read_gpickle
from timeout import timeout
import pandas as pd
import pickle
import numpy as np


def parse_args():
    parser = ArgumentParser(prog="ShotHound", prefix_chars="-/", add_help=False, description=f'Finding practical paths in BloodHound')
    parser.add_argument('--fn', type = str)
    parser.add_argument('--blockable_p', type = float)
    args = parser.parse_args()
    return args
args_input = parse_args()

number_of_runs = 1
args = {
    # "fn": "examplegraph",
    # "fn": "adsim05",
    "fn" : args_input.fn,
    "blockable_p": args_input.blockable_p,
    "no_one_hop": True,
    "seed": 0
}


# def analyse_CG(CG):
#     for n in CG.nodes():
#         try:
#             nx.all_shortest_paths(CG,source=entry,target=DA)


#TODO: 1. save the strategy into .pkl file for dyMIP and voting algorithm
#TODO: 2. write the monte carlo simulation 
#TODO: 3. if have time, write the 

if __name__ == "__main__":
    global output
    # if args["start_node_number"] == -1:
    #     number_of_runs = 1
    print(args)
    temporal = True
    data_dir = "/Users/huyngo/Desktop/Research/honeypot_dynamic/company_data/"
    df = pd.DataFrame()
    fn,  blockable_p, no_one_hop, seed = args.values()
    # CG = dynamic_graph_setup(fn, blockable_p)
    CG = merge_graph_setup(fn, blockable_p)
    label = dict()
    for u, v in CG.edges():
        if CG.edges[u, v]["label"] not in label:
            label[CG.edges[u, v]["label"]] = 1
        else: 
            label[CG.edges[u, v]["label"]] += 1
    print(label)
    
    user_nodes = []
    for n in CG.nodes():
        if CG.nodes[n]["label"] == "User":
            user_nodes.append(n)

    # print(len(is_path_to(CG, user_nodes, CG.graph["DA"])))
    setting = 6
    # CG = sample_edge_type(CG, "MemberOf", 1.5, seed)
    # CG = sample_edge_type(CG, "AdminTo", 10, seed)
    # CG = sample_edge_type(CG, "AllowedToDelegate", 10, seed)
    # CG = sample_edge_type(CG, "GpLink", 10, seed)

    # print(len(is_path_to(CG, user_nodes, CG.graph["DA"])))
    # graph 1 setting: 0 0 0 
    # graph 2 setting: 5, 5, 2
    # graph 3 setting: 10, 10, 10
    # graph 4 setting: 8, 8, 8
    label = dict()
    for u, v in CG.edges():
        if CG.edges[u, v]["label"] not in label:
            label[CG.edges[u, v]["label"]] = 1
        else: 
            label[CG.edges[u, v]["label"]] += 1
    print(label)
    write_gpickle(CG, f"//Users/huyngo/Desktop/Research/honeypot_dynamic/processed_dygraph/dy{fn}_{setting}_{blockable_p}.gpickle")
    # analyse_CG(CG)
    # logon_dict = process_company_hassession(data_dir, 0.25)
    # print(len(logon_dict["snapshot_dict"]))
    # seed = 0
    # logon_dict = mapping_company_dygraph(CG, logon_dict, seed)
    # with open('comp_data_auth.pkl', 'wb') as f:
    #     pickle.dump(logon_dict, f)
