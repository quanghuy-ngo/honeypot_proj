#netflow not so simple now :((
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


def parse_args():
    parser = ArgumentParser(prog="ShotHound", prefix_chars="-/", add_help=False, description=f'Finding practical paths in BloodHound')
    parser.add_argument('--budget', type = int)
    parser.add_argument('--start', type = int)
    parser.add_argument('--batch_number', type = int)
    parser.add_argument('--fn', type = str)
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
    'batch_number': args_input.batch_number,
    "graph_sample_number": 100,
    "no_one_hop": False,
    "seed": 1
}


def time_and_run(name, func, seed, CG):

    start_time = timeit.default_timer()

    
    if name == "greedy_v1":

        output[f"{name}_{seed}_score"], output[f"{name}_strat"] = func(CG)
    
    elif name == "greedy_v2":

        output[f"{name}_{seed}_score"], output[f"{name}_strat"] = func(CG)

    elif name == "mip_v1":

        output[f"{name}_{seed}_score"], output[f"{name}_strat"] = func(CG)

    elif name == "mip_v2":

        output[f"{name}_{seed}_score"], output[f"{name}_strat"] = func(CG)

    elif name == "dp":
        try: 
            output[f"{name}_{seed}_score"] = func(CG)
        except:
            output[f"{name}_{seed}_score"] = -1
            output[f"{name}_{seed}_time"] = -1


    output[f"{name}_{seed}_time"] = timeit.default_timer() - start_time
    return output[f"{name}_strat"]

def vote_blocking(graph_condensed, block_strategy):
    # ranking the nodes in block strategy 
    vote = dict()
    for strat in block_strategy:
        for i in strat:
            if i not in vote:
                vote[i] = 0
            else:
                vote[i] = vote[i] + 1

    return sorted(vote, key=vote.get, reverse=True)[:graph_condensed.graph["budget"]]


def dyMIP(graph, graph_batch_number, graph_sample):
    blocking_list = []
    average_score_list = []
    for batch in range(int(graph_sample/graph_batch_number)):
        print("Batch: ", batch)
        graph_batch = []
        for i in range(batch, batch+graph_batch_number):
            CG_sample = flip_edge(graph, graph.graph["hassession_mask"][i])
            graph_condensed = get_spath_graph(CG_sample, args["cross_path_p"])
            print("------------done SETUP graph_condensed-------------")
            graph_batch.append(graph_condensed)
            
        average_score, blocking_strat = mip_dygraph(graph_batch)
        average_score_list.append(average_score)
        blocking_list.append(blocking_strat)
    return average_score_list, blocking_list






#TODO: 1. save the strategy into .pkl file for dyMIP and voting algorithm
#TODO: 2. write the monte carlo simulation 
#TODO: 3. if have time, write the 
#TODO: 4. run on Uofa HPC

if __name__ == "__main__":
    global output
    # if args["start_node_number"] == -1:
    #     number_of_runs = 1
    print(args)
    skip_dp = True
    output = {}
    
    df = pd.DataFrame()
    seed = 0
    args["seed"] = seed
    graph_dir = "/Users/huyngo/Desktop/Research/honeypot_dynamic/processed_dygraph/" + args["fn"] + ".gpickle" # dyadsim025_all_10000_0.gpickle
    CG = read_gpickle(graph_dir)
    print(CG.graph.keys())
    dasdas
    # CG = dynamic_graph_setup(**args)
    # sample = args["graph_sample_number"]
    # seed = args["seed"]
    # graph_name = args["fn"]
    # write_gpickle(CG, f"//home/quanghuyngo/Desktop/CFR/Code/honeypot_dynamic/processed_graph/dy{graph_name}_{sample}_{seed}.gpickle")
    

    block_strategy_mip = []
    block_strategy_greedy = []
    score_list = []
    start_time = timeit.default_timer()
    average_score_list, blocking_list = dyMIP(CG, args["batch_number"], args["graph_sample_number"])
    time = timeit.default_timer() - start_time
    
    
    print("Average Score batches: ")
    print(sum(average_score_list) / len(average_score_list))
    # print("Score MIP(100): ")
    # print(average_score_all)
    print(np.var(average_score_list))
    print(np.std(average_score_list))
    
    save_dict = dict()
    save_dict["average_score_list"] = average_score_list
    save_dict["blocking_list"] = blocking_list
    save_dict["time"] = time
    save_dict["graph_dir"] = graph_dir
    pkl_dir = "/Users/huyngo/Desktop/Research/honeypot_dynamic/dygraph_blocking/dyMIP_" + str(args['fn']) + "_" + str(args['batch_number']) + "_" + str(args['budget']) + "_" + str(args["seed"]) + "_" + ".pkl"
    f = open(pkl_dir,"wb")

    # write the python object (dict) to pickle file
    pickle.dump(save_dict,f)
    # close file
    f.close()
    
    
    
