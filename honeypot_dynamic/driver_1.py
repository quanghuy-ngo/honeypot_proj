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
    "cross_path_p": 0,
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
    df_greedy_v2 = df[df["algo"] == "greedy_v2"]
    df_mip_v2 = df[df["algo"] == "mip_v2"]
    df_greedy_v1 = df[df["algo"] == "greedy_v1"]

    
    plt.plot(df_greedy_v2["budget"], df_greedy_v2["score"] , color='red', marker='o', label = "greedy_v2")
    plt.plot(df_greedy_v1["budget"], df_greedy_v1["score"] , color='black', marker='^', label = "greedy_v1")
    plt.plot(df_mip_v2["budget"], df_mip_v2["score"] , color='blue', marker='x', label = "mip")
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
    
    print(args)
    skip_dp = False
    output = {}
    result = []
    greedy_details = []
    dp_details = []
    # all_methods = ["greedy_v1","greedy_v2", "dp"]
    all_methods = ["greedy_v1","greedy_v2", "mip"]
    dp_win = 0
    mip_v1_win = 0
    mip_v2_win = 0
    greedy_v1_win = 0
    greedy_v2_win = 0
    
    df = pd.DataFrame()
    seed = 99
    honeypot_range = 100

    args["seed"] = seed
    CG = flow_graph_setup(**args)
    
    ori_graph_info = report(CG)
    ori_graph_info = ori_graph_info
    graph_condensed = get_spath_graph(CG, args["cross_path_p"])
    new_graph_info = report(graph_condensed)
    new_graph_info = new_graph_info + tree_width(graph_condensed)
    graph_dir = "/home/quanghuyngo/Desktop/CFR/Code/honeypot/graph/" + "graph_condensed" + "_" + str(seed)
    draw_networkx(graph_condensed, graph_dir)

    for n in range(honeypot_range):
    # for seed in range(number_of_runs):
        # if seed == 4:
        #     asdsadsa
        df_temp = pd.DataFrame()
        
        args["budget"] = n
        graph_condensed.graph["budget"] = n

        
        time_and_run("greedy_v1", greedy_node_v1, n, graph_condensed)
        time_and_run("greedy_v2", greedy_node_v2, n, graph_condensed)
        time_and_run("mip_v1", mip_flow, n, graph_condensed)
        time_and_run("mip_v2", mip_flow_2, n, graph_condensed)
        # win if the score is lower (score is attacker chance to the DA)

        # if output[f"dp_{seed}_score"] != -1:
        score_list = []
        score_list.append(output[f"greedy_v1_{n}_score"])
        score_list.append(output[f"greedy_v2_{n}_score"])
        score_list.append(output[f"mip_v1_{n}_score"])
        score_list.append(output[f"mip_v2_{n}_score"])
        
        
        min_score = min(score_list)
        algo_list = ["greedy_v1", "greedy_v2", "mip_v1", "mip_v2"]
        # if output[f"greedy_v1_{seed}_score"] <= min_score + 0.000001:
        #     greedy_v1_win += 1
        df_temp["win"] = [0 for i in range(len(algo_list))]


        if output[f"greedy_v1_{n}_score"] <= min_score + 0.000001:
            greedy_v1_win += 1
            df_temp["win"][0] = 1        
        if output[f"greedy_v2_{n}_score"] <= min_score + 0.000001:
            greedy_v2_win += 1
            df_temp["win"][1] = 1
            

        if output[f"mip_v1_{n}_score"] <= min_score + 0.000001:
            mip_v1_win += 1
            df_temp["win"][2] = 1
        if output[f"mip_v2_{n}_score"] <= min_score + 0.000001:
            mip_v2_win += 1
            df_temp["win"][3] = 1

        
        result.append(output.copy())

        

        df_temp["budget"] = [n for i in range(len(algo_list))]   
        df_temp["algo"] = algo_list
        df_temp["original graph info"] =  [ori_graph_info for i in range(len(df_temp["algo"]))]
        df_temp["new graph info"] = [new_graph_info for i in range(len(df_temp["algo"]))]
        df_temp["score"] = [output[f"greedy_v1_{n}_score"],
                            output[f"greedy_v2_{n}_score"],
                            output[f"mip_v1_{n}_score"], 
                            output[f"mip_v2_{n}_score"]]
        df_temp["time"] =  [output[f"greedy_v1_{n}_time"],
                            output[f"greedy_v2_{n}_time"],
                            output[f"mip_v1_{n}_time"],
                            output[f"mip_v2_{n}_time"]]

        df = df.append(df_temp, ignore_index=True)
        print(seed)

    # df = df.groupby(df["seed"])
    png_dir = '/home/quanghuyngo/Desktop/CFR/Code/honeypot/result/test1_result' + "_" + str(args['fn']) + "_" + str(args['start_node_number']) + "_" + str(args['blockable_p'])  +".png"
    plot_score(df, png_dir)
    out_csv = '/home/quanghuyngo/Desktop/CFR/Code/honeypot/result/test1_result' + "_" + str(args['fn']) + "_" + str(args['start_node_number']) + "_" + str(args['blockable_p'])  +".csv"
    df.to_csv(out_csv)
    
    # output[f"greedy_v1_win_time"] = greedy_v1_win
    output[f"greedy_v1_win_time"] = greedy_v1_win
    output[f"greedy_v2_win_time"] = greedy_v2_win
    output[f"mip_v1"]             = mip_v1_win
    output[f"mip_v2"]             = mip_v2_win
    
    
    print("Result length: -----------------------------------", len(result))
    print('--------------------------------------------------')
    print('--------------------------------------------------')
    print('--------------------------------------------------')
    print("First: ", result[0])
    
    # print(result)
    store = {"items": []}
    out_json = '/home/quanghuyngo/Desktop/CFR/Code/honeypot/result/test1_result' + "_" + str(args['fn']) + "_" + str(args['start_node_number']) + "_" + str(args['blockable_p'])  +".json"
    with open(out_json, 'w') as outfile:
        json.dump(output, outfile)
    
    

    # write_gpickle(
    #     (args, result),
    #     f"/home/andrewngo/Desktop/Research-Express/routes/AAAI/{args['fn']}-{args['budget']}-{args['start_node_number']}-{args['blockable_p']}-{args['double_edge_p']}.gpickle",
    # )

    # all_methods = ["trivialFPT"]
    # "trivialFPT", "greedy", "dp", "cf", "gnn"
    # for method in ["lb"] + all_methods:
    #     if method + "_res" in result[0]:
    #         time_sum = sum([r[method + "_time"] for r in result])
    #         perf_sum = sum([r[method + "_res"] for r in result])
    #         print(method, time_sum / number_of_runs, perf_sum / number_of_runs)
    # for i, r in enumerate(result):
    #     best_methods = []
    #     for method in all_methods:
    #         if method + "_res" not in r:
    #             continue
    #         is_best = True
    #         for other_method in all_methods:
    #             if other_method == method or other_method + "_res" not in r:
    #                 continue
    #             if r[other_method + "_res"] + 0.000001 < r[method + "_res"]:
    #                 is_best = False
    #         if is_best:
    #             best_methods.append(method)
    #     temp_store = {}
    #     temp_store["id"] = i
    #     temp_store["best"] = best_methods[:]
    #     # temp_store["greedy"] = greedy_details[i][:]
    #     # temp_store["gnn"] = gnn_details[i][:]
    #     # temp_store["trivial"] = trivial_details[i][:]
    #     # temp_store["cf"] = class_details[i][:]

    #     store["items"].append(temp_store)
    #     print(f"round {i} best methods", best_methods)
    #     # print("CF: ", trivial_details[i])
    #     # print("Greedy: ", greedy_details[i])
    #     # print("GNN: ", gnn_details[i])
    #     # print("Trivial: ", trivial_details[i])

    # # write_gpickle(
    # #     (args, result),
    # #     f"/home/m/Dropbox/{args['fn']}-{args['budget']}-{args['start_node_number']}-{args['blockable_p']}-{args['double_edge_p']}.gpickle",
    # # )
    # with open('/home/andrewngo/Desktop/Research-Express/routes/AAAI/best_methods.json', 'w') as outfile:
    #     json.dump(store, outfile)
    # write_gpickle(
    #     (args, result),
    #     f"/home/andrewngo/Desktop/Research-Express/routes/AAAI/{args['fn']}-{args['budget']}-{args['start_node_number']}-{args['blockable_p']}-{args['double_edge_p']}.gpickle",
    # )
