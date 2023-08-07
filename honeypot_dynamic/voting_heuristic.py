#w not so simple now :((
from argparse import ArgumentParser
from ast import arg
import json
from linecache import lazycache
from setupgraph import flow_graph_setup, get_spath_graph, dynamic_graph_setup, flip_edge
from netflow_simple_greedy import greedy_node_v1, greedy_node_v2
from netflow_simple_lp import mip_flow, mip_flow_2
from netflow_simple_TD_v2 import dp_flow, tree_width
from utility import report, evaluate_flow, draw_networkx, remove_dead_nodes
import timeit
from networkx.readwrite.gpickle import write_gpickle, read_gpickle
from timeout import timeout
import pandas as pd
import pickle



def parse_args():
    parser = ArgumentParser(prog="ShotHound", prefix_chars="-/", add_help=False, description=f'Finding practical paths in BloodHound')
    parser.add_argument('fn', type = str) #voter
    parser.add_argument('budget', type = int)
    parser.add_argument('voter', type = str)
    args = parser.parse_args()

    return args
args_input = parse_args()








number_of_runs = 1
args = {
    # "fn": "examplegraph",
    # "fn": "adsim05",
    "fn" : args_input.fn,
    "budget" : args_input.budget,
    "voter" : args_input.voter,
    "blockable_p": 0.8,
    "double_edge_p": 0,
    "multi_block_p": 1,
    "cross_path_p": 0,
    "graph_sample": 10000,
    "no_one_hop": False,
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



def get_best_k_index(data, k):
    sorted(range(len(data["average_score_list"])), key=lambda i: data["average_score_list"][i])[:k]

def get_worst_k_index(data, k):
    sorted(range(len(data["average_score_list"])), key=lambda i: data["average_score_list"][i], reverse=True)[:k]


if __name__ == "__main__":
    global output
    print(args)
    skip_dp = True
    output = {}

    df = pd.DataFrame()
    seed = 0
    args["seed"] = seed
    blocking_dir = "/hpcfs/users/a1798528/honeypot_dynamic_HPC/dygraph_blocking/" + args["fn"] + ".pkl" # dyadsim025_all_10000_0.gpickle
    with open(blocking_dir, 'rb') as f:
        data = pickle.load(f)

    blocking_strat = data["blocking_list"]
    graph_dir = data["graph_dir"]
    
    
    
    vote_block = []
    if args["voter"] == "vote_all":
        vote_block = vote_blocking(args["budget"], blocking_strat)
        
    elif args["voter"] == "best10":
        temp = []
        temp = get_best_k_index(data, 10)
        temp_strat = []
        for i in temp:
            temp_strat.append(blocking_strat[i])
        vote_block = vote_blocking(args["budget"], temp_strat)
            
    elif args["voter"] == "worst10":
        temp = []
        temp = get_worst_k_index(data, 10)
        temp_strat = []
        for i in temp:
            temp_strat.append(blocking_strat[i])
        vote_block = vote_blocking(args["budget"], temp_strat)
            
    elif args["voter"] == "best5worst5":
        temp = []
        temp = get_best_k_index(data, 5)
        temp_strat = []
        for i in temp:
            temp_strat.append(blocking_strat[i])
        temp = get_best_k_index(data, 5)
        for i in temp:
            temp_strat.append(blocking_strat[i])
        vote_block = vote_blocking(args["budget"], temp_strat)
    
    
    
    
    print(vote_block)
    # print("Score Vote for MIP:", evaluate_flow(graph_condensed, vote_block))

    temp = []
    temp.append(vote_block)
    save_dict = dict()
    #graph_dir = "/hpcfs/users/a1798528/honeypot_dynamic_HPC/processed_dygraph/" + args["graph_dir"] + ".gpickle"
    save_dict["blocking_list"] = temp
    save_dict["graph_dir"] = graph_dir
    save_dict["time"] = data["time"]
    save_dict["average_score_list"] = data["average_score_list"]
    pkl_dir = "/hpcfs/users/a1798528/honeypot_dynamic_HPC/dygraph_blocking/" + str(arg["voter"]) + "_" + str(args['fn']) + ".pkl"
    f = open(pkl_dir,"wb")
    # write the python object (dict) to pickle file
    pickle.dump(save_dict,f)
    # close file
    f.close()








# if __name__ == "__main__":
#     global output
#     # if args["start_node_number"] == -1:
#     #     number_of_runs = 1
#     print(args)
#     skip_dp = True
#     output = {}
    
#     df = pd.DataFrame()
#     seed = 0
#     args["seed"] = seed
#     graph_dir = "/home/quanghuyngo/Desktop/CFR/Code/honeypot_dynamic/processed_dygraph/" + args["fn"] + ".gpickle" # dyadsim025_all_10000_0.gpickle
#     CG = read_gpickle(graph_dir)
#     block_strategy_mip = []
#     block_strategy_greedy = []
    
#     score_list = []
#     for i in range(args["graph_sample"]):
#         print("Current Sample: ", i)
#         CG_copy = flip_edge(CG, CG.graph["hassession_mask"][i])
        
#         graph_condensed = get_spath_graph(CG_copy, args["cross_path_p"])
        
#         # new_graph_info = report(graph_condensed)
        
#         block_strategy_mip.append(time_and_run("mip_v2", mip_flow_2, i, graph_condensed))
        
#         # block_strategy_greedy.append(time_and_run("greedy_v2", greedy_node_v2, i, graph_condensed))
        
#         # Evaluate strategy every 10 step
#         if i % 10 == 0:
#             CG_copy = CG.copy()
#             graph_condensed = get_spath_graph(CG, args["cross_path_p"])    
#             vote_block = vote_blocking(graph_condensed, block_strategy_mip)
#             score_list.append(evaluate_flow(graph_condensed, vote_block))
            

#     graph_condensed = get_spath_graph(CG, args["cross_path_p"])    
    
#     all_block_mip = time_and_run("mip_v2", mip_flow_2, seed, graph_condensed)
    
#     # all_block_greedy = time_and_run("greedy_v2", greedy_node_v2, seed, graph_condensed)
    
#     out_json = '/home/quanghuyngo/Desktop/CFR/Code/honeypot_dynamic/result/dynamic' + "_" + str(args['fn']) + "_" + str(args['budget']) + "_" + str(args['start_node_number']) + "_" + str(args['blockable_p'])  +".json"
#     with open(out_json, 'w') as outfile:
#         json.dump(output, outfile)
    
#     print("############RESULT###############")
#     # print(block_strategy_mip)
#     vote_block = vote_blocking(graph_condensed, block_strategy_mip)
#     print(vote_block)
#     print("Score Vote for MIP:", evaluate_flow(graph_condensed, vote_block))
    
#     # vote_block = vote_blocking(graph_condensed, block_strategy_greedy)
#     # print(vote_block)s
#     # print("Score Vote for greedy:", evaluate_flow(graph_condensed, vote_block))
#     # print(all_block_greedy)
#     # score_all = evaluate_flow(graph_condensed, all_block_greedy)
#     # print("Score Greedy: ", score_all)
    
#     print(all_block_mip)
#     score_all = evaluate_flow(graph_condensed, all_block_mip)
#     print("Score Optimal: ", score_all)
    
#     temp = []
#     temp.append(vote_block)
#     save_dict = dict()
#     save_dict["Score_List"] = score_list
#     save_dict["blocking_list"] = temp
#     save_dict["Optimal_Score"] = score_all
#     pkl_dir = "/home/quanghuyngo/Desktop/CFR/Code/honeypot_dynamic/result/dynamic_" + str(args["graph_sample"]) + "_" + str(args['fn']) + "_" + str(args['budget']) + "_" + str(args["seed"]) + "_" + str(args['blockable_p']) + ".pkl"
#     f = open(pkl_dir,"wb")
#     # write the python object (dict) to pickle file
#     pickle.dump(save_dict,f)
#     # close file
#     f.close()
        

        
        
        