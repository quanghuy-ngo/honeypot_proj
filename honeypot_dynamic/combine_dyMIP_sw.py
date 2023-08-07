#m argparse import ArgumentParser
from argparse import ArgumentParser
from ast import arg
import json
from linecache import lazycache
from setupgraph import flow_graph_setup, get_spath_graph, dynamic_graph_setup, flip_edge
from netflow_simple_greedy import greedy_node_v1, greedy_node_v2
from netflow_simple_lp import mip_flow, mip_flow_2, mip_dygraph
# from netflow_simple_TD_v2 import dp_flow, tree_width
# from utility import report, evaluate_flow, draw_networkx, remove_dead_nodes
import timeit
from networkx.readwrite.gpickle import write_gpickle, read_gpickle
from timeout import timeout
import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm
import os
#python3 combine_dyMIP_sw.py --fn $a --budget $b --start $c --blockable $d --sampling_type $e --seed $f --spacing_window $g --batch_number $h --graph_number $i --graph_number_total $j --algo $k
def parse_args():
    parser = ArgumentParser(prog="ShotHound", prefix_chars="-/", add_help=False, description=f'Finding practical paths in BloodHound')
    parser.add_argument('--budget', type = int)
    parser.add_argument('--start', type = int)
    parser.add_argument('--fn', type = str)
    parser.add_argument('--blockable_p', type = float)
    parser.add_argument('--sampling_type', type = str)
    parser.add_argument('--seed', type = int)
    parser.add_argument('--spacing_window', type = int)
    parser.add_argument('--batch_number', type = int)
    parser.add_argument('--graph_number', type = int)
    parser.add_argument('--graph_number_total', type = int)
    parser.add_argument('--algo', type = str)
    args = parser.parse_args()
    return args
args_input = parse_args()

number_of_runs = 1
args = {
    # "fn": "examplegraph",
    # "fn": "adsim05",
    "fn" : args_input.fn,
    "budget": args_input.budget,
    "start": args_input.start,
    "blockable_p": args_input.blockable_p,
    "spacing_window": args_input.spacing_window,
    "sampling_type": args_input.sampling_type,
    'batch_number': args_input.batch_number,
    "graph_number": args_input.graph_number,
    "graph_number_total": args_input.graph_number_total,
    "seed": args_input.seed,
    "algo": args_input.algo, # 0: pure flat attacker, 1: pure competent attacker, 0.5: half attacker
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



if __name__ == "__main__":
    #print(args["dymip_dir"])
    blocking_list = []
    competent_list = []
    flat_list = []
    #seed = 0
    #args["seed"] = 0
    thread_number = int(args["graph_number_total"]/args["graph_number"])
    for i in range(thread_number):
        dir = "/gpfs/users/a1798528/honeypot_dynamic_HPC/dygraph_blocking/dymip_par/sw_" + str(args['fn']) + "_" + str(args['budget']) + "_" + str(args['start']) + "_" + str(args['blockable_p']) + "_" + str(args['sampling_type']) + "_" + str(args['spacing_window']) + "_" + str(args['batch_number']) + "_" + str(args['graph_number']) + "_" + str(args['graph_number_total']) + "_" + str(args['seed']) + "_" + str(args['algo']) + "_" + str(i) + ".pkl"
        with open(dir, 'rb') as f:
            data = pickle.load(f)
        blocking_list.extend(data["blocking_list"])
        competent_list.extend(data["competent_score_list"])
        flat_list.extend(data["flat_score_list"])
        os.remove(dir)

    vote_block = vote_blocking(args["budget"], blocking_list)

    print("Average Compentent Score")
    print("Average: ", sum(competent_list) / len(competent_list))
    print("Variance: ",np.var(competent_list))
    print("STD: ", np.std(competent_list))

    print("Average Compentent Score")
    print("Average: ", sum(flat_list) / len(flat_list))
    print("Variance: ",np.var(flat_list))
    print("STD: ", np.std(flat_list))

    average_list = [(x + y)*0.5 for x, y in zip(flat_list, competent_list)]
    
    print("Average Score")
    print("Average: ", sum(average_list) / len(average_list))
    print("Variance: ",np.var(average_list))
    print("STD: ", np.std(average_list))

    save_dict = dict()
    save_dict["competent_score_list"] = competent_list
    save_dict["flat_score_list"] = flat_list
    save_dict["blocking_list"] = vote_block
    save_dict["graph"] = data["graph"] 
    #pkl_dir = "/hpcfs/users/a1798528/honeypot_dynamic_HPC/dygraph_blocking/sw_" + str(data['fn']) +  "_" + str(args["sampling_type"]) + "_"  + str(args["graph_number"]*args["thread_number"]) + "_" + str(args['batch_number']) + "_" + str(args["spacing_window"]) + "_" + str(data['budget']) + "_" + str(data["seed"]) + ".pkl"
    pkl_dir = "/gpfs/users/a1798528/honeypot_dynamic_HPC/dygraph_blocking/dymip_par/sw_" + str(args['fn']) + "_" + str(args['budget']) + "_" + str(args['start']) + "_" + str(args['blockable_p']) + "_" + str(args['sampling_type']) + "_" + str(args['spacing_window']) + "_" + str(args['batch_number']) + "_" + str(args['graph_number_total']) + "_" + str(args['seed']) + "_" + str(args['algo']) + ".pkl"
    f = open(pkl_dir,"wb")

    # write the python object (dict) to pickle file
    pickle.dump(save_dict,f)
    #pkl_dir = "/hpcfs/users/a1798528/honeypot_dynamic_HPC/dygraph_blocking/" + str(args["voter"]) + "_" + str(args['fn']) + ".pkl"



