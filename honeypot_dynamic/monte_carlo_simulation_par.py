from argparse import ArgumentParser
from ast import arg
import json
from setupgraph import flip_edge, get_shortest_graph_in_place, dynamic_graph_setup_inplace, normal_hassession_prob, generate_hassession, binomial_hassession_prob
from netflow_simple_lp import mip_flow, mip_flow_2, mip_dygraph
from utility import report, evaluate_flow, evaluate_flow_competent
import timeit
from networkx.readwrite.gpickle import write_gpickle, read_gpickle
from timeout import timeout
import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm

# sbatch --export=ALL,a="dyadsimx05_all",b=20,c=-1,d=-2,e="normal",f=0,g=100,h=100,i=100,j=1000,k="mixed" dyMIP_sw_par.sh

# srun --ntasks 1 --exclusive -c 2 python3 monte_carlo_simulation_par.py --blocking_dir $a --mts_sampling $b --mts_thread $c --mts_graph_sample $d --mts_graph_sample_total $e

def parse_args():
    parser = ArgumentParser(prog="ShotHound", prefix_chars="-/", add_help=False, description=f'Finding practical paths in BloodHound')
    parser.add_argument('--blocking_dir', type = str)
    parser.add_argument('--mts_sampling', type = str)
    parser.add_argument('--mts_thread', type = int)
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
    "mts_thread" : args_input.mts_thread,
    "mts_graph_sample" : args_input.mts_graph_sample,
    "mts_graph_sample_total" : args_input.mts_graph_sample_total,
}



def monte_carlo_simulation_batch(graph, blocking_strategy, monte_carlo_mask):
    #monte_carlo_mask = graph.graph["hassession_mask_monte_carlo"]
    score_list_flat = []
    score_list_competent = []
    score_list_lb_competent = []
    score_list_lb_flat = []
    strat_lb = get_lb_strat(graph)

    for i in range(len(monte_carlo_mask)):
        CG_sample = flip_edge(graph, monte_carlo_mask[i])
        graph_original = get_shortest_graph_in_place(CG_sample)
        flat_score = evaluate_flow(graph_original, blocking_strategy)
        competent_score = evaluate_flow_competent(graph_original, blocking_strategy)
        score_lb_flat = evaluate_flow(graph_original, strat_lb)
        score_list_competent.append(competent_score)
        score_list_flat.append(flat_score)
        score_list_lb_flat.append(score_lb_flat)
        if i % (100) == 0:
            print("###########################")
            print(f"monte carlo iteration {i}")
            print("Average: ", sum(score_list_flat) / len(score_list_flat))
            print("Variance: ",np.var(score_list_flat))
            print("STD: ", np.std(score_list_flat))
            print("###########################")
            print("Average: ", sum(score_list_competent) / len(score_list_competent))
            print("Variance: ",np.var(score_list_competent))
            print("STD: ", np.std(score_list_competent))
    score_list = dict()
    score_list["competent"] = score_list_competent
    score_list["flat"] = score_list_flat
    score_list["lb_competent"] = score_list_lb_competent
    score_list["lb_flat"] = score_list_lb_flat
    return score_list

def get_lb_strat(graph):
    strat = []
    for n in graph.nodes():
        if graph.nodes[n]["blockable"] == True:
            strat.append(n)
    return strat

if __name__ == "__main__":
    mts_sampling, blocking_dir, mts_thread, mts_graph_sample, mts_graph_sample_total = args.values()
    dir = "/gpfs/users/a1798528/honeypot_dynamic_HPC/dygraph_blocking/" + args["blocking_dir"] + ".pkl"
    print(args["blocking_dir"])
    with open(dir, 'rb') as f:
        data = pickle.load(f)
    # average_score_list = data["average_score_list"]
    blocking_list = data["blocking_list"]
    CG = data["graph"]
    # CG.graph["budget"] = args["budget"]
    start = args["mts_thread"]*args["mts_graph_sample"]
    end = start + args["mts_graph_sample"]
    # graph = read_gpickle("/home/quanghuyngo/Desktop/CFR/Code/honeypot_dynamic/processed_dygraph/dyadsim025_all_10000_10000_1.gpickle")
    monte_carlo_mask = None
    if args["sampling_type"] == "normal":
        p = normal_hassession_prob(CG)
        data = generate_hassession(mts_graph_sample_total, p, CG.graph["hassession_to_idx"], 100)
        monte_carlo_mask = data[start:end]
    elif args["sampling_type"] == "binomial":
        prob = binomial_hassession_prob(CG)
        p = [prob]*len(CG.graph["hassession_to_idx"])
        data = generate_hassession(mts_graph_sample_total, p, CG.graph["hassession_to_idx"], 100)
        monte_carlo_mask = data[start:end]


    blocking_strategy = blocking_list

    score_list = monte_carlo_simulation_batch(CG, blocking_strategy, monte_carlo_mask)
    save_dir = "/gpfs/users/a1798528/honeypot_dynamic_HPC/mts_par/" + args["blocking_dir"] + "_" + args["mts_sampling"] + "_" + str(args["mts_graph_sample"]) + "_" + str(args["mts_thread"]) + ".pkl" 
    with open(save_dir, 'wb') as f:
        pickle.dump(score_list, f)
