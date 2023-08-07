
#t so simple now :((
from argparse import ArgumentParser
from ast import arg
import json
from linecache import lazycache
from setupgraph import generate_hassession, get_spath_graph, dynamic_graph_setup, flip_edge, dynamic_graph_setup_inplace, get_shortest_graph_in_place, normal_hassession_prob, binomial_hassession_prob
from netflow_simple_lp import mip_dygraph_mixed_attack
# from netflow_simple_TD_v2 import dp_flow, tree_width
from utility import report, evaluate_flow, evaluate_flow_competent
import timeit
from networkx.readwrite.gpickle import write_gpickle, read_gpickle
from timeout import timeout
import pandas as pd
import pickle
import numpy as np
from pulp import *
import random

#python3 dyMIP_single.py --fn $a --budget $b --start $c --blockable $d --sampling_type $e --batch_number $f --graph_number_total $g --algo $h --seed $i
def parse_args():
    parser = ArgumentParser(prog="ShotHound", prefix_chars="-/", add_help=False, description=f'Finding practical paths in BloodHound')
    parser.add_argument('--budget', type = int)
    parser.add_argument('--start', type = int)
    parser.add_argument('--blockable_p', type = float)
    parser.add_argument('--fn', type = str)
    parser.add_argument('--sampling_type', type = str)
    parser.add_argument('--seed', type = int)
    parser.add_argument('--batch_number', type = int)
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
    "sampling_type": args_input.sampling_type,
    'batch_number': args_input.batch_number,
    "graph_number_total" : args_input.graph_number_total,
    "seed": args_input.seed,
    "algo": args_input.algo,
}


def dyMIP_single(graph, hasmask, algo):
    graph_batch = []
    #index_graph = []
    #x = 100
    for i in range(len(hasmask)):
        #i = i % graph_sample
        #index_graph.append(i)
        CG_sample = flip_edge(graph, hasmask[i])

        graph_original = get_shortest_graph_in_place(CG_sample)
        time = timeit.default_timer() - start_time
        print(time)
        print("------------done SETUP graph_condensed-------------")
        graph_batch.append(graph_original)

    blocking_strat = None
    if algo == "flat":
        blocking_strat = mip_dygraph_mixed_attack(graph_batch, 0)
    elif algo == "competent":
        blocking_strat = mip_dygraph_mixed_attack(graph_batch, 1)
    elif algo == "mixed":
        blocking_strat = mip_dygraph_mixed_attack(graph_batch, 0.5)
    
    flat_score_list = []
    competent_score_list = []
    for i in range(len(graph_batch)):
        flat_score = evaluate_flow(graph_batch[i], blocking_strat)
        competent_score = evaluate_flow_competent(graph_batch[i], blocking_strat)
        print("Flat Score for Graph: ", i, ": ", flat_score)
        print("Competent Score for Graph: ", i, ": ", competent_score)

        flat_score_list.append(flat_score)
        competent_score_list.append(competent_score)
    average_competent_score = sum(competent_score_list) / len(competent_score_list)
    average_flat_score = sum(flat_score_list) / len(flat_score_list)

    print("Average Flat Score: ", average_flat_score)
    print("Average Comepentent Score: ", average_competent_score)


    return  average_competent_score, average_flat_score, blocking_strat




if __name__ == "__main__":
    global output
    # if args["start_node_number"] == -1:
    #     number_of_runs = 1
    print(args)
    skip_dp = True
    output = {}
    print(listSolvers(onlyAvailable=True))
    df = pd.DataFrame()
    fn, budget, start, blockable_p, sampling_type, batch_number, graph_number_total, seed, algo = args.values()
    # CG = dynamic_graph_setup_inplace(CG, fn, seed, start_node_number, blockable_p, budget, no_one_hop)   
    CG = dynamic_graph_setup_inplace(fn, seed, start, blockable_p, budget, False)
    # CG = dynamic_graph_setup(**args)
    # sample = args["graph_sample_number"]
    # seed = args["seed"]
    # graph_name = args["fn"]
    # write_gpickle(CG, f"//home/quanghuyngo/Desktop/CFR/Code/honeypot_dynamic/processed_graph/dy{graph_name}_{sample}_{seed}.gpickle")

    block_strategy_mip = []
    block_strategy_greedy = []
    score_list = []
    start_time = timeit.default_timer()

    hasmask = None
    if args["sampling_type"] == "normal":
        p = normal_hassession_prob(CG)
        data = generate_hassession(graph_number_total, p, CG.graph["hassession_to_idx"], seed )
        hasmask = data
    elif args["sampling_type"] == "binomial":

        prob = binomial_hassession_prob(CG)
        p = prob
        data = generate_hassession(graph_number_total, p, CG.graph["hassession_to_idx"], seed )
        hasmask = data

    data = []
    print(len(hasmask))
    index_list = random.sample([i for i in range(len(hasmask))], args["batch_number"])
    temp = []
    for i in index_list:
        temp.append(hasmask[i])
    hasmask = temp
    
    average_competent_score, average_flat_score, blocking_list = dyMIP_single(CG, hasmask, algo)
    time = timeit.default_timer() - start_time


    print("Average Score batches: ")
    #print(sum(average_score_list) / len(average_score_list))
    # print("Score MIP(100): ")
    # print(average_score_all)
    #print(np.var(average_score_list))
    #print(np.std(average_score_list))
    print(time)

    save_dict = dict()
    save_dict["average_competent_score"] = average_competent_score
    save_dict["average_flat_score"] = average_flat_score
    save_dict["blocking_list"] = blocking_list
    save_dict["time"] = time
    save_dict["graph"] = CG
    pkl_dir = pkl_dir = "/gpfs/users/a1798528/honeypot_dynamic_HPC/dygraph_blocking/single_" + str(args['fn']) + "_" + str(args['budget']) + "_" + str(args['start']) + "_" + str(args['blockable_p']) + "_" + str(args['sampling_type']) + "_"  + str(args['batch_number']) +  "_" + str(args['graph_number_total']) + "_" + str(args['seed']) + "_" + str(args['algo']) + ".pkl"
    f = open(pkl_dir,"wb")
    # write the python object (dict) to pickle file
    pickle.dump(save_dict,f)
    # close file
    f.close()







