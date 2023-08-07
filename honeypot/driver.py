from argparse import ArgumentParser
from ast import arg
from gettext import npgettext
from itertools import permutations
import json
from linecache import lazycache
from setupgraph import random_setup, flow_graph_setup, get_spath_graph
from netflow_simple_greedy import greedy_node_v1, greedy_node_v2
from netflow_simple_lp import mip_flow, mip_flow_2
from netflow_simple_TD_v2 import dp_flow, tree_width
from utility import upper_lower_bounds, report, evaluate_flow, draw_networkx
import timeit
from networkx.readwrite.gpickle import write_gpickle
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
    parser.add_argument('--budget', type = int)
    parser.add_argument('--start', type = int)
    parser.add_argument('--fn', type = str)
    args = parser.parse_args()

    return args
args_input = parse_args()

number_of_runs = 10
args = {
    # "fn": "examplegraph",
    # "fn": "adsim05",
    "fn" : args_input.fn,
    "budget": args_input.budget,
    "start_node_number": args_input.start,
    "blockable_p": -2,
    "double_edge_p": 0,
    "multi_block_p": 1,
    "cross_path_p": 0,
    "no_one_hop": False,
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






if __name__ == "__main__":
    global output
    
    
    # if args["start_node_number"] == -1:
    #     number_of_runs = 1
    print(args)
    skip_dp = True
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

    for seed in range(number_of_runs):
        # if seed == 4:
        #     asdsadsa
        df_temp = pd.DataFrame()
        args["seed"] = seed + 10
        
        

        CG = flow_graph_setup(**args)
        
        ori_graph_info = report(CG)
        ori_graph_info = ori_graph_info
        graph_condensed = get_spath_graph(CG, args["cross_path_p"])
        new_graph_info = report(graph_condensed)
        new_graph_info = new_graph_info + tree_width(graph_condensed)

        graph_dir = "/Users/huyngo/Desktop/Research/honeypot/graph/" + "graph_condensed" + "_" + str(seed)
        draw_networkx(graph_condensed, graph_dir)
        # if seed == 1:
        #     sadsdsadds
        if skip_dp == False:
            time_and_run("dp", dp_flow, seed, graph_condensed)
        else:    
            output[f"dp_{seed}_score"] = 100
            output[f"dp_{seed}_time"] = 100
        time_and_run("greedy_v1", greedy_node_v1, seed, graph_condensed)
        time_and_run("greedy_v2", greedy_node_v2, seed, graph_condensed)
        # time_and_run("mip_v1", mip_flow, seed, graph_condensed)
        time_and_run("mip_v2", mip_flow_2, seed, graph_condensed)
        # win if the score is lower (score is attacker chance to the DA)

        # if output[f"dp_{seed}_score"] != -1:
        score_list = []
        score_list.append(output[f"greedy_v1_{seed}_score"])
        score_list.append(output[f"greedy_v2_{seed}_score"])
        if output[f"dp_{seed}_score"] != -1:
            score_list.append(output[f"dp_{seed}_score"])
            
        # score_list.append(output[f"mip_v1_{seed}_score"])
        score_list.append(output[f"mip_v2_{seed}_score"])
        
        
        min_score = min(score_list)
        if output[f"greedy_v1_{seed}_score"] <= min_score + 0.000001:
            greedy_v1_win += 1
        if output[f"greedy_v2_{seed}_score"] <= min_score + 0.000001:
            greedy_v2_win += 1
        if output[f"dp_{seed}_score"] != -1:
            if output[f"dp_{seed}_score"] <= min_score + 0.000001:
                dp_win += 1
        # if output[f"mip_v1_{seed}_score"] <= min_score + 0.000001:
        #     mip_v1_win += 1
        if output[f"mip_v2_{seed}_score"] <= min_score + 0.000001:
            mip_v2_win += 1
            
        
        result.append(output.copy())
        # else:
        #     output[f"greedy_v1_{seed}_score"] = -1
        #     output[f"greedy_v2_{seed}_score"] = -1
        #     output[f"dp_{seed}_score"] = -1
        #     output[f"mip_v1_{seed}_score"] = -1
        #     output[f"mip_v2_{seed}_score"] = -1
        #     result.append(output.copy())
        
        algo_list = ["greedy_v1", "greedy_v2", "dp", "mip_v2"]
        df_temp["seed"] = [seed for i in range(len(algo_list))]   
        df_temp["algo"] = algo_list
        df_temp["original graph info"] =  [ori_graph_info for i in range(len(df_temp["algo"]))]
        df_temp["new graph info"] = [new_graph_info for i in range(len(df_temp["algo"]))]
        df_temp["score"] = [output[f"greedy_v1_{seed}_score"], output[f"greedy_v2_{seed}_score"],
                            output[f"dp_{seed}_score"], 
                            output[f"mip_v2_{seed}_score"]]
        df_temp["time"] = [output[f"greedy_v1_{seed}_time"], output[f"greedy_v2_{seed}_time"],
                            output[f"dp_{seed}_time"], 
                            output[f"mip_v2_{seed}_time"]]

        df = df.append(df_temp, ignore_index=True)


    # df = df.groupby(df["seed"])
    out_csv = '/Users/huyngo/Desktop/Research/honeypot/result/result' + "_" + str(args['fn']) + "_" + str(args['budget']) + "_" + str(args['start_node_number']) + "_" + str(args['blockable_p'])  +".csv"
    df.to_csv(out_csv)
    print(df)
    output[f"greedy_v1_win_time"] = greedy_v1_win
    output[f"greedy_v2_win_time"] = greedy_v2_win
    if skip_dp == False:
        output[f"dp"] = dp_win
    # output[f"mip_v1"] = mip_v1_win
    output[f"mip_v2"] = mip_v2_win
    
    
    print("Result length: -----------------------------------", len(result))
    print('--------------------------------------------------')
    print('--------------------------------------------------')
    print('--------------------------------------------------')
    print("First: ", result[0])
    print("Greedy_v2 win: " , output[f"greedy_v2_win_time"])
    print("mip_v2 win: ", output[f"mip_v2"])
    # print(result)
    store = {"items": []}
    out_json = '/Users/huyngo/Desktop/Research/honeypot/result/result' + "_" + str(args['fn']) + "_" + str(args['budget']) + "_" + str(args['start_node_number']) + "_" + str(args['blockable_p'])  +".json"
    with open(out_json, 'w') as outfile:
        json.dump(output, outfile)
    
    
