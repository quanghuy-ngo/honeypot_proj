#rgparse import ArgumentParser
from argparse import ArgumentParser
from ast import arg
import json
from linecache import lazycache
from timeout import timeout
import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm
import os



# srun --ntasks 1 --exclusive -c 2 python3 combine_feature_par.py --fn $a --budget $b --start $c --blockable $d --sampling_type $e --seed $f --thread_number 10 --graph_number $g --graph_number_total $h --algo $k


# python3 combine_feature_par.py --fn dyadsimx05_all --budget 20 --start -1 --blockable -2 --sampling_type normal --seed 0 --graph_number 100 --algo mixed --thread_number 10
def parse_args():
    parser = ArgumentParser(prog="ShotHound", prefix_chars="-/", add_help=False, description=f'Finding practical paths in BloodHound')
    parser.add_argument('--budget', type = int)
    parser.add_argument('--start', type = int)
    parser.add_argument('--fn', type = str)
    parser.add_argument('--blockable_p', type = str)
    parser.add_argument('--sampling_type', type = str)
    parser.add_argument('--seed', type = int)
    parser.add_argument('--graph_number', type = int)
    parser.add_argument('--graph_number_total', type = int)
    parser.add_argument('--algo', type = str)
    # parser.add_argument('--thread_number', type = int)
    args = parser.parse_args()
    return args
args_input = parse_args()

number_of_runs = 1
args = {
    "fn" : args_input.fn,
    "budget": args_input.budget,
    "start": args_input.start,
    "graph_number": args_input.graph_number,
    "graph_number_total": args_input.graph_number_total,
    "sampling_type": args_input.sampling_type,
    "blockable_p": args_input.blockable_p,
    "seed": args_input.seed,
    "algo": args_input.algo,
    # "thread_number": args_input.thread_number,
}


if __name__ == "__main__":
    #print(args["dymip_dir"])
    feature_list = []
    #seed = 0
    #args["seed"] = 0
    CG = None
    thread_number = int(args["graph_number_total"]/args["graph_number"])
    for i in range(thread_number):
        thread = i
        # graph_number = args["graph_number"]
        dir = "/gpfs/users/a1798528/honeypot_dynamic_HPC/dymip_feature/feature_" + str(args['fn']) + "_"  + str(args['budget']) + "_" + str(args["start"]) + "_" + str(args["blockable_p"])+ "_" + str(args["sampling_type"]) + "_" + str(args["graph_number"]) + "_" +  str(args["algo"]) + "_" + str(args["seed"]) + "_" + str(thread) + ".pkl"
        with open(dir, 'rb') as f:
            data = pickle.load(f)
        CG = data["graph"]
        features = data["features"]
        feature_list.extend(features)
        os.remove(dir)


    # print("Average: ", sum(average_list) / len(average_list))
    # print("Variance: ",np.var(average_list))
    # print("STD: ", np.std(average_list))
    # save_dict = dict()
    # save_dict["average_score_list"] = average_list
    # save_dict["blocking_list"] = blocking_list
    # save_dict["time"] = data["time"]
    # save_dict["graph_dir"] = data["graph_dir"]
    save_dict = dict()
    save_dict["graph"] = CG
    save_dict["features"] = feature_list
    pkl_dir = "/gpfs/users/a1798528/honeypot_dynamic_HPC/dymip_feature/feature_" + str(args['fn']) + "_"  + str(args['budget']) + "_" + str(args["start"]) + "_" + str(args["blockable_p"])+ "_" + str(args["sampling_type"]) + "_" + str(args["graph_number_total"]) + "_" +  str(args["algo"]) + "_" + str(args["seed"]) + ".pkl"
    f = open(pkl_dir,"wb")

    # write the python object (dict) to pickle file
    pickle.dump(save_dict,f)
