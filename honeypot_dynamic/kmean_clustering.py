#rgparse import ArgumentParser
from argparse import ArgumentParser
from ast import arg
import json
from linecache import lazycache
from setupgraph import get_shortest_graph_in_place, get_spath_graph, get_spath_graph_2, dynamic_graph_setup, flip_edge, generate_hassession, normal_hassession_prob, binomial_hassession_prob, dynamic_graph_setup_inplace
from netflow_simple_lp import mip_flow_phiattack, mip_dygraph_mixed_attack
from utility import report, evaluate_flow, draw_networkx, remove_dead_nodes
import timeit
from networkx.readwrite.gpickle import write_gpickle, read_gpickle
from timeout import timeout
import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances
import random
from sklearn.metrics import silhouette_score
# python3 kmean_clustering.py --fn dyadsimx05_all --budget 20 --start -1 --blockable -2 --sampling_type normal --seed 0 --graph_number 100 --algo mixed --cluster_number 10
# python3 combine_feature_par.py --fn $a --budget $b --start $c --blockable $d --sampling_type $e --seed $f --graph_number_total $h --algo $i --cluster_number $j

# sbatch --export=ALL,a="dyadsimx05",b=20,c=-1,d=-2,e="normal",f=0,g=100,h=1000,i="mixed",j=10 dyMIP_sw_par.sh
#sbatch --export=ALL,a="dyadsimx05_all",b=20,c=-1,d=-2,e="normal",f=0,g=100,h=1000,i="mixed",j=10 kmean_clustering.sh
def parse_args():
    parser = ArgumentParser(prog="ShotHound", prefix_chars="-/", add_help=False, description=f'Finding practical paths in BloodHound')    
    parser.add_argument('--budget', type = int)
    parser.add_argument('--start', type = int)
    parser.add_argument('--fn', type = str)
    parser.add_argument('--blockable_p', type = float)
    parser.add_argument('--sampling_type', type = str)
    parser.add_argument('--seed', type = int)
    parser.add_argument('--graph_number_total', type = int)
    parser.add_argument('--algo', type = str)
    parser.add_argument('--cluster_number', type = int)
    args = parser.parse_args()
    return args
args_input = parse_args()

number_of_runs = 1
args = {
    "fn" : args_input.fn,
    "budget": args_input.budget,
    "start": args_input.start,
    "graph_number_total": args_input.graph_number_total,
    "sampling_type": args_input.sampling_type,
    "blockable_p": args_input.blockable_p,
    "seed": args_input.seed,
    "algo": args_input.algo,
    "cluster_number" : args_input.cluster_number,
}

def pred_count_dict(pred_y):
    counts = dict()
    for i in pred_y:
        counts[i] = counts.get(i, 0) + 1
    return counts
def apply_kmean_2(features, n_clusters=10, mip_number=10, seed=0):
    # print(features)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=seed)
    km_obj = kmeans.fit(scaled_features)
    pred_y = km_obj.labels_
    count_dict = pred_count_dict(pred_y)
    sum_count_dict_value = sum(count_dict.values())
    weight = [count_dict[i]/sum_count_dict_value for i in range(n_clusters)]
    centroid_prob = random.choices([i for i in range(n_clusters)], weight, k=mip_number)
    centroid_number = pred_count_dict(centroid_prob)
    # print(count_dict)
    print(centroid_number)

    cluster_centers = km_obj.cluster_centers_
    # get every centroid graph
    # centroid_feature = ()
    # centroid_index, _ = pairwise_distances_argmin_min(cluster_centers, scaled_features)
    distances = pairwise_distances(cluster_centers, scaled_features, metric='euclidean')
    centroid_graph = []
    for centroid in centroid_number:
        number_of_point = centroid_number[centroid]
        ind = np.argpartition(distances[centroid], number_of_point)[:number_of_point]
        centroid_graph.extend(ind)
        
        # closest = [X_emb[indexes] for indexes in ind]

    print(centroid_graph)
    # dadas
    return centroid_graph


def apply_kmean_3(features, n_clusters=10, mip_number=10, seed=0):
    # print(features)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=seed)
    km_obj = kmeans.fit(scaled_features)
    pred_y = km_obj.labels_
    cluster_centers = km_obj.cluster_centers_
    centroid_index, _ = pairwise_distances_argmin_min(cluster_centers, scaled_features)
    count_dict = pred_count_dict(pred_y)
    sum_count_dict_value = sum(count_dict.values())
    min_count_dict = min(count_dict.values())
     
    weight_dict = [count_dict[i]/min_count_dict for i in range(n_clusters)]
    #print(centroid_number)

    weight = dict()
    for i in centroid_index:
        weight[i] = weight_dict[pred_y[i]]
    #weight 
    return centroid_index, weight



def apply_kmean(features, n_clusters=10, seed=0):
    print(features)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=seed)
    km_obj = kmeans.fit(scaled_features)
    pred_y = km_obj.labels_

    pred_y = km_obj.labels_
    count_dict = pred_count_dict(pred_y)
    smallest_cluster_idx = min(count_dict, key=count_dict.get)
    print(count_dict)
    cluster_centers = km_obj.cluster_centers_
    # get every centroid graph
    centroid_feature = ()
    centroid_index, _ = pairwise_distances_argmin_min(cluster_centers, scaled_features)

    return centroid_index
if __name__ == "__main__":
    pkl_dir = "/gpfs/users/a1798528/honeypot_dynamic_HPC/dymip_feature/feature_" + str(args['fn']) + "_"  + str(args['budget']) + "_" + str(args["start"]) + "_" + str(args["blockable_p"])+ "_" + str(args["sampling_type"]) + "_" + str(args["graph_number_total"]) + "_" +  str(args["algo"]) + "_" + str(args["seed"]) + ".pkl"
    with open(pkl_dir, 'rb') as f:
        save_dict = pickle.load(f)
    CG = save_dict["graph"]
    features = save_dict["features"]
    fn, budget, start_node_number, graph_number_total, sampling_type, blockable_p, seed, algo, cluster_number = args.values()
    
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
        p = [prob]*len(CG.graph["hassession_to_idx"])
        data = generate_hassession(graph_number_total, p, CG.graph["hassession_to_idx"], seed )
        hasmask = data
    elif args["sampling_type"] == "lb":
        hasmask_dir = "/gpfs/users/a1798528/honeypot_dynamic_HPC/processed_dygraph/" + args["fn"] + "_hasmask_mts" + ".pkl"
        with open(hasmask_dir, 'rb') as f:
            data = pickle.load(f)
        hasmask = data["sample_mts"]
        args["graph_sample_number"] = len(hasmask[0])
        args["seed"] = 0
    #centroid_index = apply_kmean(features, args['cluster_number'])

    kmax = 50
    sil = []
    for k in range(2, kmax+1):
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        kmeans = KMeans(n_clusters = k).fit(scaled_features)
        labels = kmeans.labels_
        sil.append(silhouette_score(scaled_features, labels, metric = 'euclidean'))
    print(sil)
    #dassadas
    #centroid_index, weight_dict = apply_kmean_3(features, args['cluster_number'], args['cluster_number'], 0)
    centroid_index= apply_kmean_2(features, args['cluster_number'], args['cluster_number'], 0)
    graph_batch = []
    #weight = []
    for i in centroid_index:
        CG_sample = flip_edge(CG, hasmask[i])
        graph_original = get_shortest_graph_in_place(CG_sample)
        graph_batch.append(graph_original)
    #    weight.append(weight_dict[i])

    #average_score, blocking_strat = mip_dygraph_weight(graph_batch, weight)
    if algo == "flat":
        blocking_strat = mip_dygraph_mixed_attack(graph_batch, 0)
    elif algo == "competent":   
        blocking_strat = mip_dygraph_mixed_attack(graph_batch, 1)
    elif algo == "mixed":
        blocking_strat = mip_dygraph_mixed_attack(graph_batch, 0.5)
    save_dict = dict()
    #graph_dir = "/hpcfs/users/a1798528/honeypot_dynamic_HPC/processed_dygraph/" + args["graph_dir"] + ".gpickle"
    save_dict["blocking_list"] = blocking_strat
    save_dict["graph"] = CG
    # save_dict["graph_dir"] = graph_dir
    save_dict["time"] = 0
    save_dict["average_score_list"] = []
    pkl_dir = "/gpfs/users/a1798528/honeypot_dynamic_HPC/dygraph_blocking/kmean_" + str(args['fn']) + "_"  + str(args['budget']) + "_" + str(args["start"]) + "_" + str(args["blockable_p"])+ "_" + str(args["sampling_type"]) + "_" + str(args["graph_number_total"]) + "_" +  str(args["algo"]) + "_" + str(args["seed"]) + "_" + str(args["cluster_number"]) + ".pkl"
    f = open(pkl_dir,"wb")
    # write the python object (dict) to pickle file
    pickle.dump(save_dict,f)
    print(pkl_dir)
    # close file
    f.close()




