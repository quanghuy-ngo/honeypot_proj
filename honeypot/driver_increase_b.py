from argparse import ArgumentParser
from ast import arg
from gettext import npgettext
from itertools import permutations
import pickle as pkl
from linecache import lazycache
from setupgraph import random_setup, flow_graph_setup, get_spath_graph, get_shortest_graph_in_place
from netflow_simple_greedy import greedy_node_v1, greedy_node_v2, greedy_competent
from netflow_simple_lp import mip_flow, mip_flow_2, mip_flow_competent, mip_flow_phiattack, mip_flow_phiattack_wm
from netflow_simple_TD_v2 import dp_flow, tree_width
from utility import upper_lower_bounds, report, evaluate_flow, draw_networkx, evaluate_flow_competent
import timeit
from networkx.readwrite.gpickle import write_gpickle
from networkx import minimum_node_cut, shortest_path
from timeout import timeout
import pandas as pd



# "sample.gpickle",
# "sample-dag.gpickle",
# "r100.gpickle",
# "r100-dag.gpickle",
# "r200.gpickle",
# "r200-dag.gpickle",
# "r500.gpickle",
# "r500-dag.gpickle",
# "r1000.gpickle",
# "r1000-dag.gpickle",
# "r2000.gpickle",
# "r2000-dag.gpickle",

def parse_args():
    parser = ArgumentParser(prog="ShotHound", prefix_chars="-/", add_help=False, description=f'Finding practical paths in BloodHound')
    # parser.add_argument('--budget', type = int)
    parser.add_argument('--start', type = int)
    parser.add_argument('--fn', type = str)
    # parser.add_argument('--algo', type = str)
    args = parser.parse_args()

    return args
args_input = parse_args()

max_budget = 50
number_of_runs = 10
args = {
    # "fn": "examplegraph",
    # "fn": "adsim05",
    "fn" : args_input.fn,
    "budget": 0,
    "start_node_number": args_input.start,
    # "algo": args_input.algo,
    "blockable_p": -2, # -1 AAAI-23 blockable policy, if -2 then can only block computer node
    "double_edge_p": 0,
    "multi_block_p": 1,
    "cross_path_p": 0,
    "no_one_hop": True,
}
def extract_block_edge_from_TD(TD_block, G):
    for i in TD_block:
        TD_node = i[0]
        budget = i[1]
        temp = list(TD_node[1])
        temp.append(TD_node[0])
        print(temp)
        perm = list(permutations(temp,2))
        print(perm)
        # for j in range(budget):
            
    # for i in list 
    
# def check_if_higher_layer()
def mixed_evaluate_helper(graph_original, node_blocked):
    # print(node_blocked)
    flat_score = evaluate_flow(graph_original, node_blocked)
    competent_score = evaluate_flow_competent(graph_original, node_blocked)
    true_score = (flat_score+competent_score)/2
    # print("Flat Score: ", flat_score)
    # print("Competent Score: ",competent_score)
    # print("True Score: ", true_score)
    score = [true_score, flat_score, competent_score]
    return score

def print_mixed_attack(score_list, time_list):
    true_score = [row[0] for row in score_list]
    flat_score = [row[1] for row in score_list]
    competent_score = [row[2] for row in score_list]
    
    print(args)
    mean = sum(true_score) / len(true_score)
    variance = sum([((x - mean) ** 2) for x in true_score]) / len(true_score)
    std_dev = variance ** 0.5

    print("Average True Score: ", sum(true_score)/len(true_score))
    print("Std Score:", std_dev)


    mean = sum(flat_score) / len(flat_score)
    variance = sum([((x - mean) ** 2) for x in flat_score]) / len(flat_score)
    std_dev = variance ** 0.5

    print("Average Flat Score: ", sum(flat_score)/len(flat_score))
    print("Std Score:", std_dev)


    mean = sum(competent_score) / len(competent_score)
    variance = sum([((x - mean) ** 2) for x in competent_score]) / len(competent_score)
    std_dev = variance ** 0.5

    print("Average Competent Score: ", sum(competent_score)/len(competent_score))
    print("Std Score:", std_dev)

    print("Average Time: ", sum(time_list)/len(time_list))

def time_and_run(name, func, graph_original, graph_condensed, phi=1):
    print(phi)

    start_time = timeit.default_timer()
    print("RUNNING: ", name)
    score = 0
    if name == "greedy_flat":

        strat = func(graph_condensed)

    elif name == "greedy_competent":

        strat = func(graph_original)

    elif name == "mip":

        strat = func(graph_original)

    elif name == "mixed_attack":

        strat = func(graph_original, phi)

    elif name == "mixed_attack_wm":

        strat = func(graph_original, 0.5, 1)    
    
    time = timeit.default_timer() - start_time

    score = mixed_evaluate_helper(graph_original, strat)

    return score, time






if __name__ == "__main__":
    global output
    
    
    # if args["start_node_number"] == -1:
    #     number_of_runs = 1
    print(args)

    output = {}
    result = []
    greedy_details = []
    dp_details = []

    dp_win = 0
    mip_v1_win = 0
    mip_v2_win = 0
    greedy_v1_win = 0
    greedy_v2_win = 0
    
    df = pd.DataFrame()

    
    score_1_avg = []
    score_2_avg = []
    score_3_avg = []

    time_1_avg = []
    time_2_avg = []
    time_3_avg = []
    for b in range(10,11):
        score_1 = []
        score_2 = []
        score_3 = []
        score_4 = []

        time_1 = []
        time_2 = []
        time_3 = []
        time_4 = []
        args["budget"] = b

        print("BUDGET: ", b)
        for seed in range(number_of_runs):

            df_temp = pd.DataFrame()
            args["seed"] = seed + 10
            
            
            # print(args.values())
            fn, budget, start_node_number, blockable_p, double_edge_p, multi_block_p, cross_path_p, no_one_hop, seed = args.values()
            CG = flow_graph_setup(fn, seed, start_node_number, blockable_p, double_edge_p, multi_block_p, budget, cross_path_p, no_one_hop)
            
            # ori_graph_info = report(CG)
            # ori_graph_info = ori_graph_info
            graph_original = get_shortest_graph_in_place(CG)
            graph_condensed = get_spath_graph(CG, args["cross_path_p"])

            # print("Phi = 0")
            score, time = time_and_run("mixed_attack", mip_flow_phiattack, graph_original, graph_condensed, 0)
            score_1.append(score)
            time_1.append(time)
            # print("Phi = x")
            score, time = time_and_run("mixed_attack", mip_flow_phiattack, graph_original, graph_condensed, 0.5)
            score_2.append(score)
            time_2.append(time)
            # print("Phi = 1")
            score, time = time_and_run("mixed_attack", mip_flow_phiattack, graph_original, graph_condensed, 1)
            score_3.append(score)
            time_3.append(time)
            # score, time = time_and_run("mixed_attack_wm", mip_flow_phiattack_wm, graph_original, graph_condensed, 1)
            # score_4.append(score)
            # time_4.append(time)
        score_1_avg.append(score_1)
        score_2_avg.append(score_2)
        score_3_avg.append(score_3)
        
        time_1_avg.append(sum(time_1)/len(time_1))
        time_2_avg.append(sum(time_2)/len(time_2))
        time_3_avg.append(sum(time_3)/len(time_3))
        
    print(score_1_avg)
    dsadas
    save_dict = dict()
    save_dict["0_score"] = score_1_avg
    save_dict["05_score"] = score_2_avg
    save_dict["1_score"] = score_3_avg
    save_dict["0_time"] =  time_1_avg
    save_dict["05_time"] =  time_2_avg
    save_dict["1_time"] =  time_3_avg

    save_dir = "/Users/huyngo/Desktop/Research/honeypot/result_budget/" + str(args['fn']) + "_budget_analysis.pkl"
    with open(save_dir, 'wb') as f:
        pkl.dump(save_dict, f)
    # if args["algo"] == "mixed_attack":
            
    #     print("Phi = 0")
    #     print_mixed_attack(score_1, time_1)
    #     print("Phi = 0.5")
    #     print_mixed_attack(score_2, time_2)
    #     print("Phi = 1")
    #     print_mixed_attack(score_3, time_3)
    #     # print("Weighted Metric Method")
    #     # print_mixed_attack(score_4, time_4)


    