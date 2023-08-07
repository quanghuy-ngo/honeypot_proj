# not so simple now :((
from argparse import ArgumentParser
from ast import arg
import json
from linecache import lazycache
from setupgraph import dynamic_graph_setup_inplace_large, generate_hassession, mapping_company_dygraph, process_company_hassession, flip_edge, dynamic_graph_setup_inplace, get_shortest_graph_in_place, normal_hassession_prob, binomial_hassession_prob
from netflow_simple_greedy import greedy_node_v1, greedy_node_v2
from netflow_simple_lp import mip_flow, mip_flow_2, mip_dygraph, mip_dygraph_mixed_attack
# from netflow_simple_TD_v2 import dp_flow, tree_width
from utility import evaluate_flow, evaluate_flow_competent
import timeit
from networkx.readwrite.gpickle import write_gpickle, read_gpickle
from timeout import timeout
import pandas as pd
import pickle
import numpy as np
from pulp import *
#srun --ntasks 1 --exclusive -c 1 python3 dyMIP_par_sw.py --fn $a --budget $b --start $c --blockable $d --sampling_type $e --seed $f --spacing_window $g --batch_number $h --thread 0 --graph_number $i --graph_number_total $j --algo $k
# python3 dyMIP_par_sw.py 


# sbatch --export=ALL,a="dyadsimx05_all",b=20,c=-1,d=-2,e="normal",f=0,g=100,h=100,i=100j=1000,k="mixed" dyMIP_sw_par.sh


# sbatch --export=ALL,a="dyadsimx05_all",b=20,c=-1,d=-2,e="normal",f=100,g=1000,g="mixed",i=0 dyMIP_single.sh
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
    parser.add_argument('--thread', type = int)
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
    "sampling_type": args_input.sampling_type,
    "spacing_window": args_input.spacing_window,
    'batch_number': args_input.batch_number,
    "thread": args_input.thread,
    "graph_number": args_input.graph_number,
    "graph_number_total": args_input.graph_number_total,
    "seed": args_input.seed,
    "algo": args_input.algo, # 0: pure flat attacker, 1: pure competent attacker, 0.5: half attacker
}




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


def dyMIP(graph, graph_batch_number, graph_sample, spacing_window, hasmask, algo):
    blocking_list = []
    average_competent_score_list = []
    average_flat_score_list = []
    for batch in range(0, len(hasmask), spacing_window):
        print("Batch: ", batch)
        graph_batch = []
        index_graph = []
        for i in range(batch, batch+graph_batch_number):
            i = i % graph_sample
            index_graph.append(i)
            import timeit
            start_time = timeit.default_timer()
            CG_sample = flip_edge(graph, hasmask[i])
            print("DONE flip")
            graph_original = get_shortest_graph_in_place(CG_sample)
            time = timeit.default_timer() - start_time
            print(time)
            print("------------done SETUP graph_condensed-------------")
            graph_batch.append(graph_original)
            print(len(graph_original.nodes()))
            print(graph_original.graph["DA"])
        # graph_batch = remove_reachable(graph_batch)
        blocking_strat = None
        if algo == "flat":
            blocking_strat = mip_dygraph_mixed_attack(graph_batch, 0)
        elif algo == "competent":
            blocking_strat = mip_dygraph_mixed_attack(graph_batch, 1)
        elif algo == "mixed":
            blocking_strat = mip_dygraph_mixed_attack(graph_batch, 0.5)


        flat_score_list = []
        competent_score_list = []
        print(blocking_strat)
        for i in range(len(graph_batch)):
            print(i)
            flat_score = evaluate_flow(graph_batch[i], blocking_strat)
            competent_score = evaluate_flow_competent(graph_batch[i], blocking_strat)
            print("Flat Score for Graph: ", i, ": ", flat_score)
            print("Competent Score for Graph: ", i, ": ", competent_score)
            flat_score_list.append(flat_score)
            competent_score_list.append(competent_score)
            #optimal
            optimal_block = []
            for n in graph_batch[i].nodes():
                if graph_batch[i].nodes[n]["blockable"] == True:
                    optimal_block.append(n)
            flat_score = evaluate_flow(graph_batch[i], blocking_strat)
            competent_score = evaluate_flow_competent(graph_batch[i], blocking_strat)
            print("Optimal Flat Score for Graph: ", i, ": ", flat_score)
            print("Optimal Competent Score for Graph: ", i, ": ", competent_score)
        average_competent_score = sum(competent_score_list) / len(competent_score_list)
        average_flat_score = sum(flat_score_list) / len(flat_score_list)
    # print(node_blocked)
    # The optimised objective function value is printed to the screen
        print("Average Flat Score: ", average_flat_score)
        print("Average Comepentent Score: ", average_competent_score)
        average_competent_score_list.append(average_competent_score)
        average_flat_score_list.append(average_flat_score)
        blocking_list.append(blocking_strat)
    return average_competent_score_list, average_flat_score_list, blocking_list

if __name__ == "__main__":
    global output
    # if args["start_node_number"] == -1:
    #     number_of_runs = 1
    
    print(args)
    skip_dp = True
    output = {}
    print(listSolvers(onlyAvailable=True))
    df = pd.DataFrame()

    fn, budget, start_node_number, blockable_p, sampling_type, spacing_window, batch_number, thread_number, graph_number, graph_number_total, seed, algo = args.values()
    # CG = dynamic_graph_setup_inplace(CG, fn, seed, start_node_number, blockable_p, budget, no_one_hop)   
    

    # print(sampling_type)
    if args["sampling_type"] == "factor":
        factor = 1
        if "adsim100" in fn: 
            factor = 10
        elif "3" in fn:
            factor = 3
        elif "5" in fn:
            factor = 5 
        print("Hassession Factor: ", factor)
        CG = dynamic_graph_setup_inplace_large(fn, seed, start_node_number, blockable_p, budget, True, factor)

    else:
        CG = dynamic_graph_setup_inplace(fn, seed, start_node_number, blockable_p, budget, True)

    # sample = args["graph_sample_number"]
    # seed = args["seed"]
    # graph_name = args["fn"]
    # write_gpickle(CG, f"//home/quanghuyngo/Desktop/CFR/Code/honeypot_dynamic/processed_graph/dy{graph_name}_{sample}_{seed}.gpickle")

    block_strategy_mip = []
    block_strategy_greedy = []
    score_list = []
    start_time = timeit.default_timer()

    hasmask = None
    

    # dasds
    start = args["thread"]*args["graph_number"]
    end = start + args["graph_number"]
    if args["sampling_type"] == "normal":
        p = normal_hassession_prob(CG)
        data = generate_hassession(graph_number_total, p, CG.graph["hassession_to_idx"], seed )
        hasmask = data[start: end]
    elif args["sampling_type"] == "binomial":

        prob = binomial_hassession_prob(CG)
        p = [prob]*len(CG.graph["hassession_to_idx"])
        data = generate_hassession(graph_number_total, p, CG.graph["hassession_to_idx"], seed )
        hasmask = data[start: end]
    elif args["sampling_type"] == "lb":
        hasmask_dir = "/gpfs/users/a1798528/honeypot_dynamic_HPC/processed_dygraph/" + args["fn"] + "_hasmask_mts" + ".pkl"
        with open(hasmask_dir, 'rb') as f:
            data = pickle.load(f)
        hasmask = data["sample_mts"]
        args["graph_number"] = len(hasmask[0])
        args["seed"] = 0
    elif args['sampling_type'] == "company":
        data_dir = "/Users/huyngo/Desktop/Research/honeypot_dynamic/company_data/"
        logon_dict = process_company_hassession(data_dir, 0.25)
        # print(len(logon_dict["snapshot_dict"]))
        logon_dict = mapping_company_dygraph(CG, logon_dict, seed)

        hasmask = logon_dict[100:1100]
    elif args['sampling_type'] == "factor":
        logon_dict = CG.graph["logon_dict"]
        hasmask = logon_dict[100:1100]

    # print(hasmask[1])
    # print(CG.graph["hassession_to_idx"])
    data = []
    print(len(hasmask))
    print("Number of Nodes: ", len(CG.nodes()))
    print("Number of edges: ", len(CG.edges()))

    competent_score_list, flat_score_list, blocking_list = dyMIP(CG, args["batch_number"], args["graph_number"], args["spacing_window"], hasmask, algo)
    time = timeit.default_timer() - start_time


    print("Average Competent Score batches: ")
    print(sum(competent_score_list) / len(competent_score_list))

    print(np.var(competent_score_list))
    print(np.std(competent_score_list))

    print("Average Flat Score batches: ")
    print(sum(flat_score_list) / len(flat_score_list))

    print(np.var(flat_score_list))
    print(np.std(flat_score_list))
    print(time)

    save_dict = dict()
    save_dict["competent_score_list"] = competent_score_list
    save_dict["flat_score_list"] = flat_score_list
    save_dict["blocking_list"] = blocking_list
    save_dict["time"] = time
    save_dict["graph"] = CG
    pkl_dir = "/gpfs/users/a1798528/honeypot_dynamic_HPC/dygraph_blocking/dymip_par/sw_" + str(args['fn']) + "_" + str(args['budget']) + "_" + str(args['start']) + "_" + str(args['blockable_p']) + "_" + str(args['sampling_type']) + "_" + str(args['spacing_window']) + "_" + str(args['batch_number']) + "_" + str(args['graph_number']) + "_" + str(args['graph_number_total']) + "_" + str(args['seed']) + "_" + str(args['algo']) + "_" + str(args['thread_number']) + ".pkl"
    f = open(pkl_dir,"wb")

    

    # write the python object (dict) to pickle file
    pickle.dump(save_dict,f)
    # close file
    f.close()



