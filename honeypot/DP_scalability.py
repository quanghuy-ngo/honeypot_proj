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
    parser.add_argument('budget', type = int)
    parser.add_argument('start', type = int)
    parser.add_argument('fn', type = str)
    args = parser.parse_args()

    return args
args_input = parse_args()

number_of_runs = 50
args = {
    # "fn": "examplegraph",
    # "fn": "adsim05",
    "fn" : args_input.fn,
    "budget": args_input.budget,
    "start_node_number": args_input.start,
    "blockable_p": 0.8,
    "double_edge_p": 0,
    "multi_block_p": 1,
    "cross_path_p": 0, # probability of the 
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

def plot_score(df, png_dir):
    import matplotlib.pyplot as plt
    dp = df[df["algo"] == "dp"]


    
    plt.plot(dp["start_node_number"], dp["time"] , color='blue', marker='o', label = "greedy_v2")
    plt.title('Score', fontsize=14)
    plt.xlabel('# of Honeypot', fontsize=14)
    plt.ylabel('Attacker Probability', fontsize=14)
    plt.grid(True)
    plt.legend()
    # plt.show()
    plt.savefig(png_dir)
# in this test, we try to find point where the optimal score converge


if __name__ == "__main__":
    global output
    
    
    # if args["start_node_number"] == -1:
    #     number_of_runs = 1
    
    # 278 starting nodes for r4000_alledges
    
    print(args)
    skip_dp = False
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
    df_temp = pd.DataFrame()
    seed = 0
    st_range = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 278]
    for st in st_range:

        args["seed"] = seed
        args["start_node_number"] = st
        CG = flow_graph_setup(**args) 
        
        ori_graph_info = report(CG)
        ori_graph_info = ori_graph_info
        graph_condensed = get_spath_graph(CG, args["cross_path_p"])
        new_graph_info = report(graph_condensed)
        new_graph_info = new_graph_info + tree_width(graph_condensed)


        
        time_and_run("dp", dp_flow, st, graph_condensed)
        # win if the score is lower (score is attacker chance to the DA)

        algo_list = ["dp"]
        # if output[f"greedy_v1_{seed}_score"] <= min_score + 0.000001:
        #     greedy_v1_win += 1
        df_temp["win"] = [0 for i in range(len(algo_list))]
        df_temp["start_node_number"] = [st for i in range(len(algo_list))]   
        df_temp["algo"] = algo_list
        df_temp["original graph info"] =  [ori_graph_info for i in range(len(df_temp["algo"]))]
        df_temp["new graph info"] = [new_graph_info for i in range(len(df_temp["algo"]))]
        df_temp["score"] = [ output[f"dp_{st}_score"]]
        df_temp["time"] =  [output[f"dp_{st}_time"]]

        df = df.append(df_temp, ignore_index=True)
        out_csv = '/home/quanghuyngo/Desktop/CFR/Code/honeypot/result/DP_scalability' + "_" + str(args['fn']) + "_" + str(args['blockable_p'])  +".csv"
        df.to_csv(out_csv)
  

    # df = df.groupby(df["seed"])
    png_dir = '/home/quanghuyngo/Desktop/CFR/Code/honeypot/result/DP_scalability' + "_" + str(args['fn']) + "_" + str(args['blockable_p'])  + ".png"
    plot_score(df, png_dir)
    out_csv = '/home/quanghuyngo/Desktop/CFR/Code/honeypot/result/DP_scalability' + "_" + str(args['fn']) + ".csv"
    df.to_csv(out_csv)
    
    
