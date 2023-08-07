from networkx.readwrite.gpickle import read_gpickle, write_gpickle
import random
from utility import is_blockable, is_start, remove_dead_nodes
import networkx as nx
import math
import numpy as np
from numpy.random import default_rng
import random
from temporal_generate import generate_temporal_sample_2, get_snapshot
import sys
import pandas as pd
from datetime import datetime, timedelta
import pytz
# from networkx.exception import NetworkXNoPath
# graph_file_names = [
#     "sample.gpickle",
#     "sample-dag.gpickle",
#     "r100.gpickle",
#     "r100-dag.gpickle",
#     "r200.gpickle",
#     "r200-dag.gpickle",
#     "r500.gpickle",
#     "r500-dag.gpickle",
#     "r1000.gpickle",
#     "r1000-dag.gpickle",
#     "r2000.gpickle",
#     "r2000-dag.gpickle",
# ]
# graph_file_names = [
#     "r500-dag.gpickle"
# ]

# graphs = {}
# for fn in graph_file_names:
#     graphs[fn] = read_gpickle("/home/andrewngo/Desktop/Research-Express/routes/AAAI/" + fn)


def rand_bin_array(K, N):
    arr = np.zeros(N)
    arr[:K]  = 1
    np.random.shuffle(arr)
    return arr

def random_setup(
    fn, seed, start_node_number, blockable_p, double_edge_p, multi_block_p, budget
):
    # graph_file_names = [
    #     "r500-dag.gpickle"
    # ]
    graph_file_names = fn + ".gpickle"
    graphs = {}
    graphs[graph_file_names] = read_gpickle("/hpcfs/users/a1798528/honeypot_HPC/" + graph_file_names)

    CG = graphs[fn + ".gpickle"].copy()
    random.seed(seed)
    CG.graph["budget"] = budget
    for v in CG.nodes():
        if v == CG.graph["DA"]:
            CG.nodes[v]["node_type"] = "DA"
        else:
            CG.nodes[v]["node_type"] = ""
    # modification in the way of taking entry node
    start_nodes = random.sample(list(CG.nodes()), k=min(start_node_number+1, len(list(CG.nodes()))))
    print(list(CG.nodes()))
    print(start_nodes)
    if CG.graph["DA"] in start_nodes:
        start_nodes.remove(CG.graph["DA"])
    start_nodes = start_nodes[:start_node_number]
    assert len(start_nodes) == start_node_number
    for v in start_nodes:
        CG.nodes[v]["node_type"] = "S"
    double_edge_list = []
    for u, v in CG.edges():
        ran = random.random()
        if ran <= blockable_p:
            if random.random() <= double_edge_p and CG.nodes[v]["node_type"] == "":
                double_edge_list.append((u, v))
            else:
                CG[u][v]["blockable"] = True
        else:
            CG[u][v]["blockable"] = False

    #####################
    for u in CG.nodes():
        if CG.out_degree(u) > 1 and random.random() <= multi_block_p:
            for v in list(CG.successors(u)):
                CG[u][v]["blockable"] = True

    for u, v in double_edge_list:
        a = hash((u, v))
        CG.add_edge(u, a)
        CG.nodes[a]["node_type"] = ""
        CG.nodes[a]["layer"] = CG.nodes[v]["layer"]
        CG[u][v]["blockable"] = True
        CG[u][a]["blockable"] = True
        for x in list(CG.successors(v)):
            CG[v][x]["blockable"] = False
            CG.add_edge(a, x)
            CG[a][x]["blockable"] = False
    preprocess(CG)
    ################################
    # CG[2][0]["blockable"] = False
    # CG[1][0]["blockable"] = False
    ################################
    return CG


def flow_graph_setup(
    fn, seed, start_node_number, blockable_p, double_edge_p, multi_block_p, budget, cross_path_p, no_one_hop
):
    # graph_file_names = [
    #     "r500-dag.gpickle"
    # ]
    graph_file_names = fn + ".gpickle"
    graphs = {} #/Users/huyngo/Desktop/Research/honeypot/
    graphs[graph_file_names] = read_gpickle("/Users/huyngo/Desktop/Research/honeypot/" + graph_file_names)

    CG = graphs[fn + ".gpickle"].copy()
    random.seed(seed)
    CG.graph["budget"] = budget
    for v in CG.nodes():
        if v == CG.graph["DA"]:
            CG.nodes[v]["node_type"] = "DA"
        else:
            CG.nodes[v]["node_type"] = ""
    # modification in the way of taking entry node
    start_nodes = []
    
    if start_node_number == -1:
        for n in CG.nodes():
            if CG.nodes[n]["label"] == "User":
                if no_one_hop == True:
                    hop_to_DA = nx.shortest_path_length(CG, source=n, target=CG.graph["DA"])
                    
                    if hop_to_DA >= 2:
                        start_nodes.append(n)
                else:
                    start_nodes.append(n)
    else:
        user_nodes = []
        for n in CG.nodes():
            if CG.nodes[n]["label"] == "User":
                if no_one_hop == True:
                    hop_to_DA = nx.shortest_path_length(CG, source=n, target=CG.graph["DA"])
                    print(hop_to_DA)
                    if hop_to_DA >= 2:
                        user_nodes.append(n)
                else:
                    user_nodes.append(n)
        start_nodes = random.sample(list(user_nodes), k=min(start_node_number+1, len(list(CG.nodes()))))
        # print(list(CG.nodes()))
        if CG.graph["DA"] in start_nodes:
            start_nodes.remove(CG.graph["DA"])
        start_nodes = start_nodes[:start_node_number]
        assert len(start_nodes) == start_node_number
        
    print("Starting nodes:", start_nodes)
    print(len(start_nodes))
    print("DA nodes: ", CG.graph["DA"])
    
    # remove starting nodes that 1 hop away to DA (this is for honeypot allocation on nodes, we basically can not do anything is does not have intermediate nodes)
    
    for v in start_nodes:
        CG.nodes[v]["node_type"] = "S"
    double_edge_list = []
    for u, v in CG.edges():
        ran = random.random()
        if ran <= blockable_p:
            if random.random() <= double_edge_p and CG.nodes[v]["node_type"] == "":
                double_edge_list.append((u, v))
            else:
                CG[u][v]["blockable"] = True
        else:
            CG[u][v]["blockable"] = False

    
    #assign honey allocatable nodes
    # for v in CG.nodes():
    #     ran = random.random()
    #     if ran <= blockable_p:
    #         if random.random() <= double_edge_p and CG.nodes[v]["node_type"] == "":
    #             double_edge_list.append((u, v))
    #         else:
    #             CG.nodes[v]["blockable"] = True
    #     else:
    #         CG.nodes[v]["blockable"] = False
    #####################
    for u in CG.nodes():
        if CG.out_degree(u) > 1 and random.random() <= multi_block_p:
            for v in list(CG.successors(u)):
                CG[u][v]["blockable"] = True
    # create second edge if
    for u, v in double_edge_list:
        # print("herhehreerre")
        a = hash((u, v))
        # print(a)
        CG.add_edge(u, a)
        CG.nodes[a]["node_type"] = ""
        CG.nodes[a]["layer"] = CG.nodes[v]["layer"]
        CG[u][v]["blockable"] = True
        CG[u][a]["blockable"] = True
        for x in list(CG.successors(v)):
            CG[v][x]["blockable"] = False
            CG.add_edge(a, x)
            CG[a][x]["blockable"] = False
    preprocess(CG)
    
    ################################
    # CG[2][0]["blockable"] = False
    # CG[1][0]["blockable"] = False
    ################################
    return CG



def flip_edge(CG, mask):
    CG_copy = CG.copy()
    edge_to_idx = CG_copy.graph["hassession_to_idx"]
    for idx, edge in enumerate(edge_to_idx):
        if mask[idx] == False:
            if CG_copy.has_edge(edge[0], edge[1]) == True:
                CG_copy.remove_edge(edge[0], edge[1])
        elif mask[idx] == True: 
            if CG_copy.has_edge(edge[0], edge[1]) == False:
                CG_copy.add_edge(edge[0], edge[1])
    remove_dead_nodes(CG_copy)
    CG_copy.graph["hassession_to_idx"] = []
    CG_copy.graph["hassession_mask"] = []
    return CG_copy
    
    
def process_company_hassession(data_dir, hour_snapshot):
    snapshot_dict = dict()
    df = pd.read_csv(data_dir + "export_4647_event_1.csv", header=None)
    temp = pd.read_csv(data_dir + "export_4647_event_2.csv", header=None)
    df = df.append(temp)
    temp = pd.read_csv(data_dir + "export_4647_event_3.csv", header=None)
    df = df.append(temp)
    temp = pd.read_csv(data_dir + "export_4647_event_4.csv", header=None)
    df = df.append(temp)
    temp = pd.read_csv(data_dir + "export_4647_event_5.csv", header=None)
    df = df.append(temp)
    df = df.rename(columns={0: 'time', 1: 'eventID', 2: 'Name', 3: 'Computer'})
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%dT%H:%M:%S')
    # convert time zone

    adelaide = pytz.timezone('Australia/Adelaide')
    df.index = df["time"]
    df.index = df.index.tz_convert(adelaide)
    df["time"] = df.index
    df = df.reset_index(drop=True)
    #4624 log on
    #4647 log off
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    df_list = []
    t = timedelta(days=30)
    start = (min(df["time"]) + t*11)

    for i in range(0,len(df),2):
        row1 = df.iloc[i]
        row2 = df.iloc[i+1]
        start_time = row2["time"]
        end_time = row1["time"]
        if start_time < start:
            continue
        if end_time == None or start_time == None:
            continue
        if start_time >= end_time:
            continue
        delta = end_time - start_time
        df_list.append((start_time, end_time, row1["Name"], row1["Computer"], delta))
    df_list.sort()


    df = df.reset_index(drop=True)
    # start_time = Timestamp('2022-07-13 10:00:00.000000+0930', tz='Australia/Adelaide')
    t = timedelta(days=30)
    start_period = min(df["time"]) + t*11
    # snapshot_time = min(df["time"])
    end_period = max(df["time"])
    delta_time = timedelta(hours=hour_snapshot)
    online_list = []
    current_time = start_period
    online_list = []
    computer_list = sorted(list(set(df["Computer"])))
    user_list = sorted(list(set(df["Name"])))
    # print(df["Computer"])
    # print(computer_list)
    # dsadsad
    comp_mask_id = {computer_list[i]: i for i in range(len(computer_list))}
    user_mask_id = {user_list[i]: i for i in range(len(user_list))}

    edge_dict = dict()
    time_unit = 1
    while(current_time < end_period):
        online_session = 0
    #     print(snapshot_time)
        current_time += delta_time
        snapshot_dict[time_unit] = []
        for i in range(len(df_list)):
            row = df_list[i]
            start = row[0]
            end = row[1]
            user = row[2]
            comp = row[3]
            if current_time > start and current_time < end:
                edge = (comp_mask_id[comp], user_mask_id[user])
                snapshot_dict[time_unit].append(edge)
        time_unit += 1
    
    logon_dict = dict()
    logon_dict["comp_mask"] = comp_mask_id
    logon_dict["user_mask"] = user_mask_id
    logon_dict["snapshot_dict"] = snapshot_dict
    return logon_dict


def mapping_factor_dygraph(CG, logon_dict, seed, factor_size):
    comp_mask_id = logon_dict["comp_mask"]
    user_mask_id = logon_dict["user_mask"]
    snapshot_dict = logon_dict["snapshot_dict"]
    CG.graph["hassession_to_idx"] = []
    user_nodes = []
    comp_nodes = []
    other_nodes = []
    # temp = dict()
    DA = CG.graph["DA"]
    for n in CG.nodes():
        if CG.nodes[n]["label"] == "User":
            user_nodes.append(n)
        elif CG.nodes[n]["label"] == "Computer":
            comp_nodes.append(n)
        else:
            other_nodes.append(n)
    random.seed(seed)
    random_user = random.sample(user_nodes, len(user_mask_id)*factor_size)
    random_computer = random.sample(comp_nodes, len(comp_mask_id)*factor_size)

    shuffled_user = random_user
    shuffled_comp = random_computer
    random.shuffle(shuffled_user)
    random.shuffle(shuffled_comp)
    new_user_mask = []
    new_comp_mask = []
    for i in range(factor_size):
        picked_user = shuffled_user[:len(user_mask_id)] # first 80% of shuffled list
        remain_user = shuffled_user[len(user_mask_id):] # last 20% of shuffled list


        picked_comp = shuffled_comp[:len(comp_mask_id)] # first 80% of shuffled list
        remain_comp = shuffled_comp[len(comp_mask_id):] # last 20% of shuffled list


        user_mask = {i : picked_user[i] for i in range(len(user_mask_id))}
        comp_mask = {i : picked_comp[i] for i in range(len(comp_mask_id))}
        shuffled_user = remain_user
        shuffled_comp = remain_comp
        new_user_mask.append(user_mask)
        new_comp_mask.append(comp_mask)
    print(len(new_comp_mask))

    hasmask = []
    hassesion_to_idx = dict()
    snapshot_new = []
    for time in snapshot_dict:
        snapshot = [snapshot_new[i] for i in range(len(snapshot_new))]
        for edge in snapshot_dict[time]:
            for dup in range(factor_size):
                new_edge = (new_comp_mask[dup][edge[0]], new_user_mask[dup][edge[1]])
                if new_edge in hassesion_to_idx:
                    snapshot[hassesion_to_idx[new_edge]] = 1
                else:
                    hassesion_to_idx[new_edge] = len(snapshot)
                    snapshot.append(1)
        hasmask.append(snapshot)
        snapshot_new = [0 for i in range(len(snapshot))]

    for i in range(len(hasmask)):
        for j in range(len(hasmask[i]), len(hassesion_to_idx)):
            hasmask[i].append(0)                 
    
    

    return hasmask, hassesion_to_idx     

def mapping_company_dygraph(CG, logon_dict, seed):

    comp_mask_id = logon_dict["comp_mask"]
    user_mask_id = logon_dict["user_mask"]
    snapshot_dict = logon_dict["snapshot_dict"]
    hassesion_to_idx = CG.graph["hassession_to_idx"]
    user_nodes = []
    comp_nodes = []
    other_nodes = []
    # temp = dict()
    DA = CG.graph["DA"]
    for n in CG.nodes():
        if CG.nodes[n]["label"] == "User":
            user_nodes.append(n)
        elif CG.nodes[n]["label"] == "Computer":
            comp_nodes.append(n)
        else:
            other_nodes.append(n)
    random.seed(seed)
    random_user = random.sample(user_nodes, len(user_mask_id))
    random_computer = random.sample(comp_nodes, len(comp_mask_id))
    # print(random_user)
    # print(user_mask_id)
    # dsadsadas

    new_user_mask = {i : random_user[i] for i in range(len(user_mask_id))}
    new_comp_mask = {i : random_computer[i] for i in range(len(comp_mask_id))}

    hasmask = []
    for time in snapshot_dict:
        snapshot = [0 for i in range(len(hassesion_to_idx))]
        for edge in snapshot_dict[time]:
            new_edge = (new_comp_mask[edge[0]], new_user_mask[edge[1]])
            if new_edge in hassesion_to_idx:
                snapshot[hassesion_to_idx[new_edge]] = 1
            # print("hererer")
        hasmask.append(snapshot)
    return hasmask




def generate_hassession(sample_number, propability, hassession_to_idx, seed=0):
    n = len(hassession_to_idx)
    m = sample_number
    mask_list = []
    import timeit

    start_time = timeit.default_timer()
    np.random.seed(seed)
    edge_idx_list = []
    
    probabilities = np.array(propability)

    # Generate random numbers for each simulation
    random_numbers = np.random.rand(m, n)
    print(random_numbers.shape)

    # Use vectorized comparison to get indices of on coins for each simulation
    edge_idx_list = (random_numbers < probabilities)
    print(edge_idx_list)
    size_bytes = sys.getsizeof(edge_idx_list)
    # Convert bytes to megabytes
    size_mb = size_bytes / (1024 * 1024)
    print(f"Memory size of my_list: {size_mb:.2f} MB")
    # Reshape indices into list of lists for each simulation

    time = timeit.default_timer() - start_time
    print(time)
    return edge_idx_list



def  merge_graph_setup(fn, blockable_p):
    # graph_file_names = [
    #     "r500-dag.gpickle"
    # ]
    graph_file_names = fn + ".gpickle"
    graphs = {} # /Users/huyngo/Desktop/Research/honeypot/
    graphs[graph_file_names] = read_gpickle("/Users/huyngo/Desktop/Research/honeypot_dynamic/processed_graph/" + graph_file_names)

    CG = graphs[fn + ".gpickle"].copy()


    
    user_nodes = []
    comp_nodes = []
    other_nodes = []
    # temp = dict()
    DA = CG.graph["DA"]
    for n in CG.nodes():
        if CG.nodes[n]["label"] == "User":
            user_nodes.append(n)
        elif CG.nodes[n]["label"] == "Computer":
            comp_nodes.append(n)
        else:
            other_nodes.append(n)

    user_nodes.append(DA)
    new_edge = []

    remove_edge = []
    for u, v in CG.edges():
        if CG.edges[u, v]["label"] == "HasSession":
            remove_edge.append((u, v))
    CG.remove_edges_from(remove_edge)



    label = dict()
    for u, v in CG.edges():
        if CG.edges[u, v]["label"] not in label:
            label[CG.edges[u, v]["label"]] = 1
        else: 
            label[CG.edges[u, v]["label"]] += 1
    print(label)
    print(len(is_path_to(CG, user_nodes, CG.graph["DA"])))

    reachable_nodes = is_path_to(CG, list(CG.nodes()), CG.graph["DA"])

    print("Number of Reachable node: ", len(reachable_nodes))
    reachable_edge = dict()
    for u, v in CG.edges():
       if u in reachable_nodes:
           label = CG.edges[u, v]["label"]
           if label not in reachable_edge:
               reachable_edge[label] = []
           reachable_edge[label].append((u,v))
    
    for label in reachable_edge:
        print(label, len(reachable_edge[label]))
    
    for label in reachable_edge:
        random_remove_edge(CG, reachable_edge[label], 6, 1)

    print("after ")
    label = dict()
    for u, v in CG.edges():
        if CG.edges[u, v]["label"] not in label:
            label[CG.edges[u, v]["label"]] = 1
        else: 
            label[CG.edges[u, v]["label"]] += 1
    print(label)
    print(len(is_path_to(CG, user_nodes, CG.graph["DA"])))
    print("-----------")


    for u in comp_nodes:
        for v in user_nodes:
            # hassession_edges.append((u,v))   
            
            if CG.has_edge(u, v) == False:
                CG.add_edge(u, v)
                CG.edges[u, v]["label"] = "HasSession"
                new_edge.append((u,v))
            # else:
            #     if CG.edges[u, v]["label"] == "HaSession":
            #         CG.edges[u, v]["label"] == "HaSession_DA"

    # print(len(comp_nodes))
    # print(len(user_nodes))
    label = dict()
    for u, v in CG.edges():
        if CG.edges[u, v]["label"] not in label:
            label[CG.edges[u, v]["label"]] = 1
        else: 
            label[CG.edges[u, v]["label"]] += 1
    print(label)
    print(len(is_path_to(CG, user_nodes, CG.graph["DA"])))
    remove_dead_nodes(CG)
    # for n in CG.nodes():


    nodes_to_remove = [node for node, in_degree in CG.in_degree() if in_degree == 0 and (node in comp_nodes or node in other_nodes)]
    CG.remove_nodes_from(nodes_to_remove)
    remove_dead_nodes(CG)
    user_nodes = []
    comp_nodes = []
    other_nodes = []
    for n in CG.nodes():
        if CG.nodes[n]["label"] == "User":
            user_nodes.append(n)
        elif CG.nodes[n]["label"] == "Computer":
            comp_nodes.append(n)
        else:
            other_nodes.append(n)
    print(len(comp_nodes))
    print(len(user_nodes))


    hassession_to_idx = dict()
    count = 0
    count_not = 0
    for u, v in CG.edges():
        if CG.edges[u, v]["label"] == "HasSession":
            hassession_to_idx[(u,v)] = count
            count += 1
        else:
            count_not += 1
    print(count)

    blockable_nodes = []
    if blockable_p == -1:
        max_hop = 1 
        print(max_hop)
        #find max hop
        for n in CG.nodes():
            if CG.nodes[n]["label"] != "User":
                hop_to_DA = nx.shortest_path_length(CG, source=n, target=CG.graph["DA"])
                if hop_to_DA > max_hop:
                    max_hop = hop_to_DA
        
        for n in CG.nodes():
            if CG.nodes[n]["label"] != "User":
                hop_to_DA = nx.shortest_path_length(CG, source=n, target=CG.graph["DA"])
                ran = random.random()
                p = hop_to_DA/max_hop
                if ran <= p:
                    CG.nodes[n]["blockable"] = True
                    blockable_nodes.append(n)
                else:
                    CG.nodes[n]["blockable"] = False
            else:
                CG.nodes[n]["blockable"] = False
                
    # Can only block computer
    elif blockable_p == -2:
        for n in CG.nodes():
            if CG.nodes[n]["label"] == "Computer":
                CG.nodes[n]["blockable"] = True
                blockable_nodes.append(n)
            else:
                CG.nodes[n]["blockable"] = False
    else:
        for v in CG.nodes():
            ran = random.random()
            if CG.nodes[v]["label"] != "":
                if ran <= blockable_p and CG.nodes[v]["label"] != "User" and v != DA:
                    CG.nodes[v]["blockable"] = True
                    blockable_nodes.append(v)
                else:
                    CG.nodes[v]["blockable"] = False
    CG.graph["hassession_to_idx"] = hassession_to_idx
    CG.graph["blockable_nodes"] = blockable_nodes

    # if fn == "adsim100":
    CG.remove_edges_from(new_edge)
    CG.graph["hassession_to_idx"] = []

    hasedge = []
    for u, v in CG.edges():
        if CG.edges[u, v]["label"] == "HasSession":
            hasedge.append((u,v))
    CG.remove_edges_from(hasedge)

    for v in CG.nodes():
        if v == CG.graph["DA"]:
            CG.nodes[v]["node_type"] = "DA"
        else:
            CG.nodes[v]["node_type"] = ""
    graph_name = fn
    # modification in the way of taking entry node
    blockable_p = str(blockable_p)
    print("Number of nodes: ", len(CG.nodes()))
    print("Number of edges: ", count_not)
    # dasds
   
    
    return CG






def dynamic_graph_setup(fn, blockable_p):
    # graph_file_names = [
    #     "r500-dag.gpickle"
    # ]
    graph_file_names = fn + ".gpickle"
    graphs = {} # /Users/huyngo/Desktop/Research/honeypot/
    graphs[graph_file_names] = read_gpickle("/Users/huyngo/Desktop/Research/honeypot_dynamic/processed_graph/" + graph_file_names)

    CG = graphs[fn + ".gpickle"].copy()


    
    user_nodes = []
    comp_nodes = []
    other_nodes = []
    # temp = dict()
    DA = CG.graph["DA"]
    for n in CG.nodes():
        if CG.nodes[n]["label"] == "User":
            user_nodes.append(n)
        elif CG.nodes[n]["label"] == "Computer":
            comp_nodes.append(n)
        else:
            other_nodes.append(n)

    user_nodes.append(DA)

    for u in comp_nodes:
        for v in user_nodes:
            # hassession_edges.append((u,v))   
            if CG.has_edge(u, v) == False:
                CG.add_edge(u, v)
                CG.edges[u, v]["label"] = "HasSession"

    # print(len(comp_nodes))
    # print(len(user_nodes))
    remove_dead_nodes(CG)
    # for n in CG.nodes():


    nodes_to_remove = [node for node, in_degree in CG.in_degree() if in_degree == 0 and (node in comp_nodes or node in other_nodes)]
    CG.remove_nodes_from(nodes_to_remove)
    remove_dead_nodes(CG)
    user_nodes = []
    comp_nodes = []
    other_nodes = []
    for n in CG.nodes():
        if CG.nodes[n]["label"] == "User":
            user_nodes.append(n)
        elif CG.nodes[n]["label"] == "Computer":
            comp_nodes.append(n)
        else:
            other_nodes.append(n)
    print(len(comp_nodes))
    print(len(user_nodes))



    hassession_to_idx = dict()
    count = 0
    for u, v in CG.edges():
        if CG.edges[u, v]["label"] == "HasSession":
            hassession_to_idx[(u,v)] = count
            count += 1
    print(count)

    blockable_nodes = []
    if blockable_p == -1:
        max_hop = 1 
        print(max_hop)
        #find max hop
        for n in CG.nodes():
            if CG.nodes[n]["label"] != "User":
                hop_to_DA = nx.shortest_path_length(CG, source=n, target=CG.graph["DA"])
                if hop_to_DA > max_hop:
                    max_hop = hop_to_DA
        
        for n in CG.nodes():
            if CG.nodes[n]["label"] != "User":
                hop_to_DA = nx.shortest_path_length(CG, source=n, target=CG.graph["DA"])
                ran = random.random()
                p = hop_to_DA/max_hop
                if ran <= p:
                    CG.nodes[n]["blockable"] = True
                    blockable_nodes.append(n)
                else:
                    CG.nodes[n]["blockable"] = False
            else:
                CG.nodes[n]["blockable"] = False
                
    # Can only block computer
    elif blockable_p == -2:
        for n in CG.nodes():
            if CG.nodes[n]["label"] == "Computer":
                CG.nodes[n]["blockable"] = True
                blockable_nodes.append(n)
            else:
                CG.nodes[n]["blockable"] = False
    else:
        for v in CG.nodes():
            ran = random.random()
            if CG.nodes[v]["label"] != "":
                if ran <= blockable_p:
                    CG.nodes[v]["blockable"] = True
                    blockable_nodes.append(v)
                else:
                    CG.nodes[v]["blockable"] = False
    CG.graph["hassession_to_idx"] = hassession_to_idx
    CG.graph["blockable_nodes"] = blockable_nodes

    
    for v in CG.nodes():
        if v == CG.graph["DA"]:
            CG.nodes[v]["node_type"] = "DA"
        else:
            CG.nodes[v]["node_type"] = ""
    graph_name = fn
    # modification in the way of taking entry node
    blockable_p = str(blockable_p)
    write_gpickle(CG, f"//Users/huyngo/Desktop/Research/honeypot_dynamic/processed_dygraph/dy{graph_name}_{blockable_p}.gpickle")
    
    return CG


def temporal_graph_setup(
    fn, seed, start_node_number, blockable_p, double_edge_p, multi_block_p, budget, cross_path_p, day_of_data,
    interarrival_mean, new_edge_rate_mean, no_one_hop):
    # graph_file_names = [
    #     "r500-dag.gpickle"
    # ]
    graph_file_names = fn + ".gpickle"
    graphs = {}
    graphs[graph_file_names] = read_gpickle("/hpcfs/users/a1798528/honeypot_dynamic_HPC/processed_graph/" + graph_file_names)

    CG = graphs[fn + ".gpickle"].copy()
    random.seed(seed)
    CG.graph["budget"] = budget

    
    user_nodes = []
    comp_nodes = []
    # temp = dict()
    for n in CG.nodes():
        if CG.nodes[n]["label"] == "User":
            user_nodes.append(n)
        elif CG.nodes[n]["label"] == "Computer":
            comp_nodes.append(n)
    
    print(CG.graph["DA"])
    user_nodes.append(CG.graph["DA"])
    hassession_to_idx = dict()
    for u in comp_nodes:
        for v in user_nodes:
            # hassession_edges.append((u,v))   
            if CG.has_edge(u, v) == False:
                CG.add_edge(u, v)
                CG.edges[u, v]["label"] = "HasSession"
    
    print("***")
    print(len(user_nodes))
    print(len(comp_nodes))
    
    remove_dead_nodes(CG)
    
    user_nodes = []
    comp_nodes = []
    # temp = dict()
    for n in CG.nodes():
        if CG.nodes[n]["label"] == "User":
            user_nodes.append(n)
        elif CG.nodes[n]["label"] == "Computer":
            comp_nodes.append(n)  
    user_nodes.append(CG.graph["DA"])
    print(len(user_nodes))
    print(len(comp_nodes))
    print("*******")

    
    count = 0
    for u, v in CG.edges():
        if CG.edges[u, v]["label"] == "HasSession":
            hassession_to_idx[(u,v)] = count
            count += 1

    # check random sample if duplicate or not
    temp = set()
    hassession_mask = []
    count = 0
    end_time = day_of_data*24*60*60
    auth_list = generate_temporal_sample_2(CG, user_nodes, comp_nodes, hassession_to_idx, end_time,
                                        new_edge_rate_mean, interarrival_mean)
    snapshot_list = get_snapshot(auth_list, hassession_to_idx, end_time, hour_per_snapshot = 1)
    
    CG.graph["hassession_mask"] = snapshot_list
    CG.graph["hassession_to_idx"] = hassession_to_idx
    for v in CG.nodes():
        if v == CG.graph["DA"]:
            CG.nodes[v]["node_type"] = "DA"
        else:
            CG.nodes[v]["node_type"] = ""
            
    # modification in the way of taking entry node
    start_nodes = []
    
    if start_node_number == -1:
        for n in CG.nodes():
            if CG.nodes[n]["label"] == "User":
                if no_one_hop == True:
                    hop_to_DA = nx.shortest_path_length(CG, source=n, target=CG.graph["DA"])
                    
                    if hop_to_DA >= 2:
                        start_nodes.append(n)
                else:
                    start_nodes.append(n)
    else:
        user_nodes = []
        for n in CG.nodes():
            if CG.nodes[n]["label"] == "User":
                if no_one_hop == True:
                    hop_to_DA = nx.shortest_path_length(CG, source=n, target=CG.graph["DA"])
                    print(hop_to_DA)
                    if hop_to_DA >= 2:
                        user_nodes.append(n)
                else:
                    user_nodes.append(n)
        start_nodes = random.sample(list(user_nodes), k=min(start_node_number+1, len(list(CG.nodes()))))
        if CG.graph["DA"] in start_nodes:
            start_nodes.remove(CG.graph["DA"])
        start_nodes = start_nodes[:start_node_number]
        assert len(start_nodes) == start_node_number
        
    print("Starting nodes:", start_nodes)
    print(len(start_nodes))
    print("DA nodes: ", CG.graph["DA"])
    
    # remove starting nodes that 1 hop away to DA (this is for honeypot allocation on nodes, we basically can not do anything is does not have intermediate nodes)
    CG.graph["start_nodes"] = start_nodes
    for v in start_nodes:
        CG.nodes[v]["node_type"] = "S"
    double_edge_list = []
    for u, v in CG.edges():
        ran = random.random()
        if ran <= blockable_p:
            if random.random() <= double_edge_p and CG.nodes[v]["node_type"] == "":
                double_edge_list.append((u, v))
            else:
                CG[u][v]["blockable"] = True
        else:
            CG[u][v]["blockable"] = False
            
            
    #assign honey allocatable nodes
    blockable_nodes = []
    for v in CG.nodes():
        ran = random.random()
        if ran <= blockable_p:
            if random.random() <= double_edge_p and CG.nodes[v]["node_type"] == "":
                double_edge_list.append((u, v))
            else:
                CG.nodes[v]["blockable"] = True
                blockable_nodes.append(v)
        else:
            CG.nodes[v]["blockable"] = False
    CG.graph["blockable_nodes"] = blockable_nodes
    #assign honey allocatable nodes
    # 2^(user*comp)

    #####################
    
    for u in CG.nodes():
        if CG.out_degree(u) > 1 and random.random() <= multi_block_p:
            for v in list(CG.successors(u)):
                CG[u][v]["blockable"] = True

    return CG, auth_list, snapshot_list


def get_spath_graph(CG, cross_path_p):
    # global path_all, DA, path_number_dict, starting_nodes
    # report(CG)
    spliting_nodes = {v:CG.out_degree(v) for v in CG.nodes() if (CG.out_degree(v) > 1)}
    combining_nodes = {v:CG.out_degree(v) for v in CG.nodes() if (CG.in_degree(v) > 1)}

    # print(CG.nodes())
    DA = CG.graph["DA"]
    # draw_networkx(CG, "CG_example_netflow")
    starting_nodes = []
    for v in CG.nodes():
        # print(v)
        if CG.nodes[v]["node_type"] == "S":
            starting_nodes.append(v)

    path_all = []
    path_number_dict = dict()
    path_dict = dict()
    layers = dict()
    for entry in starting_nodes:
        # print([p for p in nx.all_shortest_paths(CG,source=entry,target=DA)])
        temp = list(nx.all_shortest_paths(CG,source=entry,target=DA))
        path_number_dict[entry] = len(temp)
        path_all = path_all + temp
        for path in temp:
            layer_id = 1
            # print(path)
            for i in range(len(path)-1, -1, -1):
                # print(i)
                # print(layer_id)
                if layer_id not in layers:
                    layers[layer_id] = set()
                layers[layer_id].add(path[i])
                layer_id += 1
    
    # print(layers)s
    # print(path_all)
    edge_list = extract_edge(path_all)
    graph_condensed = nx.DiGraph()
    graph_condensed.add_edges_from(edge_list)

    
    path_all = []
    path_number_dict = dict()
    path_dict = dict()
    for entry in starting_nodes:
        # print([p for p in nx.all_shortest_paths(CG,source=entry,target=DA)])
        temp = list(nx.all_shortest_paths(graph_condensed,source=entry,target=DA))
        path_number_dict[entry] = len(temp)
        path_all = path_all + temp
    # print(path_all)
    edge_list = extract_edge(path_all)
    graph_condensed = nx.DiGraph()
    graph_condensed.add_edges_from(edge_list)
    
    
    
    for v in graph_condensed.nodes():
        graph_condensed.nodes[v]["node_type"] = CG.nodes[v]["node_type"]
        graph_condensed.nodes[v]["blockable"] = CG.nodes[v]["blockable"]
    # for u, v in graph_condensed.edges():
    #     graph_condensed[u][v]["blockable"] = CG[u][v]["blockable"]
    graph_condensed.graph["DA"] = CG.graph["DA"]
    graph_condensed.graph["budget"] = CG.graph["budget"]
    graph_condensed.graph["all_user"] = CG.graph["start_nodes"]
    graph_condensed.graph["path_all"] = path_all
    graph_condensed.graph["path_number_dict"] = path_number_dict
    graph_condensed.graph["starting_nodes"] = starting_nodes
    graph_condensed.graph["blockable_nodes"] = CG.graph["blockable_nodes"]
    # print(report(graph_condensed))
    return graph_condensed


def get_spath_graph_2(CG):
    # global path_all, DA, path_number_dict, starting_nodes
    # report(CG)
    # print(CG.nodes())
    DA = CG.graph["DA"]



    # Welp, to be honest, i dont know why finding all shortest path again work, 
    # although it produce the same graph (same node set and edges set),
    # if have time please check it 
    if "starting_nodes" in CG.graph:
        starting_nodes = CG.graph["starting_nodes"]
    else:
        starting_nodes = []
        for v in CG.nodes():
            if CG.nodes[v]["node_type"] == "S":
                starting_nodes.append(v)
    path_all = []
    path_number_dict = dict()
    path_dict = dict()
    node_path_dict = {}
    # print(len(starting_nodes))
    node_no_path = []
    for entry in starting_nodes:
        # print([p for p in nx.all_shortest_paths(CG,source=entry,target=DA)])

        try:
            temp = list(nx.all_shortest_paths(CG,source=entry,target=DA))
        except NodeNotFound:
            temp = []
            node_no_path.append(entry)

        path_number_dict[entry] = len(temp)
        path_all = path_all + temp
        node_path_dict[entry] = temp
    # print(path_all)
    edge_list = extract_edge(path_all)
    graph_condensed = nx.DiGraph()
    graph_condensed.add_edges_from(edge_list)
    for v in graph_condensed.nodes():
        graph_condensed.nodes[v]["node_type"] = CG.nodes[v]["node_type"]
        graph_condensed.nodes[v]["blockable"] = CG.nodes[v]["blockable"]
    # for u, v in graph_condensed.edges():
    #     graph_condensed[u][v]["blockable"] = CG[u][v]["blockable"]
    graph_condensed.graph["DA"] = CG.graph["DA"]
    graph_condensed.graph["budget"] = CG.graph["budget"]
    # graph_condensed.graph["all_user"] = CG.graph["start_nodes"]
    if  "possible_starting_nodes" in CG.graph:
        graph_condensed.graph["possible_starting_nodes"] = CG.graph["possible_starting_nodes"]
    graph_condensed.graph["node_no_path"] = node_no_path
    graph_condensed.graph["path_all"] = path_all
    graph_condensed.graph["path_number_dict"] = path_number_dict
    graph_condensed.graph["starting_nodes"] = starting_nodes
    graph_condensed.graph["blockable_nodes"] = CG.graph["blockable_nodes"]
    if  "user_nodes" in CG.graph:
        graph_condensed.graph["user_nodes"] = CG.graph["user_nodes"]
    graph_condensed.graph["node_path_dict"] = node_path_dict 
    if  "nodes_in_use" in CG.graph:
        graph_condensed.graph["nodes_in_use"] = CG.graph["nodes_in_use"]
    if "node_to_feature_id" in CG.graph:
        graph_condensed.graph["node_to_feature_id"] = CG.graph["node_to_feature_id"]  
    # print(report(graph_condensed))
    return graph_condensed


def extract_edge(path_list):
    edge_list = []
    for path in path_list:
        for i in range(len(path)):
            if i == len(path)-1:
                break
            edge_list.append((path[i], path[i+1]))
    return edge_list



def preprocess(CG):
    for u in CG.nodes():
        if CG.out_degree(u) <= 1:
            continue
        unblockable_next = {}
        unblockable_len = {}
        for v in list(CG.successors(u)):
            path = uncached_path_to_split_node_or_DA(CG, u, v)
            path_len = len(path) - 1
            if not is_path_blockable(CG, path):
                if path[-1] not in unblockable_next:
                    unblockable_next[path[-1]] = v
                    unblockable_len[path[-1]] = path_len
                elif path_len < unblockable_len[path[-1]]:
                    unblockable_next[path[-1]] = v
                    unblockable_len[path[-1]] = path_len
        for v in list(CG.successors(u)):
            path = uncached_path_to_split_node_or_DA(CG, u, v)
            path_len = len(path) - 1
            if path[-1] not in unblockable_len:
                continue
            if (
                path_len >= unblockable_len[path[-1]]
                and v != unblockable_next[path[-1]]
            ):
                CG.remove_edge(u, v)

    has_blockable = False
    for u, v in CG.edges():
        if is_blockable(CG, u, v):
            has_blockable = True
    assert has_blockable

    remove_in_degree_0(CG)

    start_nodes = []
    split_nodes = []
    for u in CG.nodes():
        if CG.out_degree(u) > 1:
            split_nodes.append(u)
        if is_start(CG, u):
            start_nodes.append(u)
    CG.graph["start_nodes"] = tuple(sorted(start_nodes))
    CG.graph["split_nodes"] = tuple(sorted(split_nodes))


def remove_in_degree_0(CG):
    keep_deleting = True
    while keep_deleting:
        keep_deleting = False
        for u in list(CG.nodes()):
            if not is_start(CG, u) and CG.in_degree(u) == 0:
                CG.remove_node(u)
                keep_deleting = True


def uncached_path_to_split_node_or_DA(CG, u, v):
    if CG.out_degree(v) > 1 or v == CG.graph["DA"]:
        return [u, v]
    else:
        future = list(CG.successors(v))
        assert len(future) == 1
        return [u] + uncached_path_to_split_node_or_DA(CG, v, future[0])


def is_path_blockable(CG, path):
    for i in range(len(path) - 1):
        if is_blockable(CG, path[i], path[i + 1]):
            return True
    return False



def plain_dynamic_graph_setup(
    fn, seed, start_node_number, blockable_p, double_edge_p, multi_block_p, budget, cross_path_p, no_one_hop):
    # graph_file_names = [
    #     "r500-dag.gpickle"
    # ]
    graph_file_names = fn + ".gpickle"
    graphs = {}
    graphs[graph_file_names] = read_gpickle("/hpcfs/users/a1798528/honeypot_dynamic_HPC/processed_graph/" + graph_file_names)

    CG = graphs[fn + ".gpickle"].copy()
    random.seed(seed)
    CG.graph["budget"] = budget


    user_nodes = []
    comp_nodes = []
    # temp = dict()
    DA = CG.graph["DA"]
    for n in CG.nodes():
        if CG.nodes[n]["label"] == "User":
            user_nodes.append(n)
        elif CG.nodes[n]["label"] == "Computer":
            comp_nodes.append(n)
    user_nodes.append(DA)

    hassession_to_idx = dict()
    for u in comp_nodes:
        for v in user_nodes:
            # hassession_edges.append((u,v))
            if CG.has_edge(u, v) == False:
                CG.add_edge(u, v)
                CG.edges[u, v]["label"] = "HasSession"
    remove_dead_nodes(CG)


    count = 0
    for u, v in CG.edges():
        if CG.edges[u, v]["label"] == "HasSession":
            hassession_to_idx[(u,v)] = count
            count += 1
    CG.graph["hassession_to_idx"] = hassession_to_idx
    for v in CG.nodes():
        if v == CG.graph["DA"]:
            CG.nodes[v]["node_type"] = "DA"
        else:
            CG.nodes[v]["node_type"] = ""

    # modification in the way of taking entry node
    start_nodes = []

    if start_node_number == -1:
        for n in CG.nodes():
            if CG.nodes[n]["label"] == "User":
                if no_one_hop == True:
                    hop_to_DA = nx.shortest_path_length(CG, source=n, target=CG.graph["DA"])

                    if hop_to_DA >= 2:
                        start_nodes.append(n)
                else:
                    start_nodes.append(n)
    else:
        user_nodes = []
        for n in CG.nodes():
            if CG.nodes[n]["label"] == "User":
                if no_one_hop == True:
                    hop_to_DA = nx.shortest_path_length(CG, source=n, target=CG.graph["DA"])
                    print(hop_to_DA)
                    if hop_to_DA >= 2:
                        user_nodes.append(n)
                else:
                    user_nodes.append(n)
        start_nodes = random.sample(list(user_nodes), k=min(start_node_number+1, len(list(CG.nodes()))))
        if CG.graph["DA"] in start_nodes:
            start_nodes.remove(CG.graph["DA"])
        start_nodes = start_nodes[:start_node_number]
        assert len(start_nodes) == start_node_number

    print("Starting nodes:", start_nodes)
    print(len(start_nodes))
    print("DA nodes: ", CG.graph["DA"])
    # remove starting nodes that 1 hop away to DA (this is for honeypot allocation on nodes, we basically can not do anything is does not have intermediate nodes)
    CG.graph["start_nodes"] = start_nodes
    for v in start_nodes:
        CG.nodes[v]["node_type"] = "S"
    double_edge_list = []
    for u, v in CG.edges():
        ran = random.random()
        if ran <= blockable_p:
            if random.random() <= double_edge_p and CG.nodes[v]["node_type"] == "":
                double_edge_list.append((u, v))
            else:
                CG[u][v]["blockable"] = True
        else:
            CG[u][v]["blockable"] = False


    #assign honey allocatable nodes
    blockable_nodes = []
    for v in CG.nodes():
        ran = random.random()
        if ran <= blockable_p:
            if random.random() <= double_edge_p and CG.nodes[v]["node_type"] == "":
                double_edge_list.append((u, v))
            else:
                CG.nodes[v]["blockable"] = True
                blockable_nodes.append(v)
        else:
            CG.nodes[v]["blockable"] = False
    CG.graph["blockable_nodes"] = blockable_nodes
    #assign honey allocatable nodes
    # 2^(user*comp)

    #####################

    for u in CG.nodes():
        if CG.out_degree(u) > 1 and random.random() <= multi_block_p:
            for v in list(CG.successors(u)):
                CG[u][v]["blockable"] = True

    return CG

def sample_edge_type(CG, edge_type, keep_fraction, seed):
    edges = []
    for u, v in CG.edges():
        if CG.edges[u, v]["label"] == edge_type:
            edges.append((u, v))
    random.seed(seed)
    remove_edges = random.sample(edges, len(edges) - int(len(edges)/keep_fraction) )
    CG.remove_edges_from(remove_edges)
    return CG

def random_remove_edge(CG, edge_list, keep_fraction, seed):
    random.seed(seed)
    remove_edges = random.sample(edge_list, len(edge_list) - int(len(edge_list)/keep_fraction) )
    CG.remove_edges_from(remove_edges)
    return CG



def is_path_to(CG, node_list, target):
    reachable = []
    for n in node_list:
        try:
            hop_to_DA = nx.shortest_path_length(CG, source=n, target=target)
            reachable.append(n)
        except:
            # print("herree")
            continue
    return reachable

def dynamic_graph_setup_inplace_large(
    fn, seed, start_node_number, blockable_p, budget, no_one_hop, factor
):
    
    data_dir = "/Users/huyngo/Desktop/Research/honeypot_dynamic/company_data/"
    logon_dict = process_company_hassession(data_dir, 0.25)

    print("DONE")

    graph_file_names = fn + "_" + str(blockable_p) + ".gpickle"
    graphs = {} # /Users/huyngo/Desktop/Research/honeypot/
    graphs[graph_file_names] = read_gpickle("/Users/huyngo/Desktop/Research/honeypot_dynamic/processed_dygraph/" + graph_file_names)
    random.seed(seed)
    CG = graphs[graph_file_names].copy()
    CG.graph["budget"] = budget





    edge_remove = []
    for u, v in CG.edges():
        if CG.edges[u, v]["label"] == "HasSession":
            edge_remove.append((u,v))
    CG.remove_edges_from(edge_remove)


    logon_dict, hassesion_to_idx = mapping_factor_dygraph(CG, logon_dict, seed, factor_size=factor)
    # modification in the way of taking entry node
    CG.graph["hassession_to_idx"] = hassesion_to_idx
    CG.graph["logon_dict"] = logon_dict
    
    label = dict()

    edge_remove = []
    for u, v in CG.edges():
        if CG.edges[u, v]["label"] == "HasSession":
            edge_remove.append((u,v))
    CG.remove_edges_from(edge_remove)
    label = dict()
    for u, v in CG.edges():
        if CG.edges[u, v]["label"] not in label:
            label[CG.edges[u, v]["label"]] = 1
        else: 
            label[CG.edges[u, v]["label"]] += 1
    print(label)
    user_nodes = []
    for n in CG.nodes():
        if CG.nodes[n]["label"] == "User":
            user_nodes.append(n)

    print(len(user_nodes))
    per_reachable_user = is_path_to(CG, user_nodes, CG.graph["DA"])


    for u, v in  hassesion_to_idx.keys():
        CG.add_edge(u,v)
        CG.edges[u, v]["label"] = "HasSession"
    label = dict()
    for u, v in CG.edges():
        if CG.edges[u, v]["label"] not in label:
            label[CG.edges[u, v]["label"]] = 1
        else: 
            label[CG.edges[u, v]["label"]] += 1
    print(label)
    reachable_user = is_path_to(CG, user_nodes, CG.graph["DA"])

    print(len(per_reachable_user))
    print(len(reachable_user))

    



    sadsaasdd
    # label = dict()
    # for u, v in CG.edges():
    #     if CG.edges[u, v]["label"] not in label:
    #         label[CG.edges[u, v]["label"]] = 1
    #     else: 
    #         label[CG.edges[u, v]["label"]] += 1
    # print(label)
    
    # CG = sample_edge_type(CG, "AdminTo", 5, seed)
    # CG = sample_edge_type(CG, "AllowedToDelegate", 5, seed)
    # CG = sample_edge_type(CG, "GpLink", 5, seed)

    label = dict()
    for u, v in CG.edges():
        if CG.edges[u, v]["label"] not in label:
            label[CG.edges[u, v]["label"]] = 1
        else: 
            label[CG.edges[u, v]["label"]] += 1
    print(label)



    for u, v in CG.graph["hassession_to_idx"]:
        CG.add_edge(u, v)
        CG.edges[u, v]["label"] = "HasSession"
    # remove_dead_nodes(CG)
    for u, v in CG.graph["hassession_to_idx"]:
        CG.add_edge(u, v)
        CG.edges[u, v]["label"] = "HasSession"
    edge_list = dict()
    for n in CG.nodes():
        try:
            path = nx.shortest_path(CG, source=n, target=CG.graph["DA"])
            if len(path) >= 4:
                # print(path)
                for i in range(len(path)-1):
                    edge = CG.edges[path[i], path[i+1]]["label"]
                    if edge not in edge_list:
                        edge_list[edge] = 1
                    else:
                        edge_list[edge] += 1
        except:
            continue
    print(edge_list)
    # dasdsa


    start_nodes = []
    
    if start_node_number == -1:
        for n in CG.nodes():
            if CG.nodes[n]["label"] == "User":
                if no_one_hop == True:
                    hop_to_DA = nx.shortest_path_length(CG, source=n, target=CG.graph["DA"])
                    
                    if hop_to_DA >= 3:
                        start_nodes.append(n)
                else:
                    start_nodes.append(n)
    else:
        user_nodes = []
        user_dist = dict()
        for n in CG.nodes():
            if CG.nodes[n]["label"] == "User":
                if no_one_hop == True:
                    hop_to_DA = nx.shortest_path_length(CG, source=n, target=CG.graph["DA"])
                    # print(hop_to_DA)
                    if hop_to_DA >= 3:
                        user_nodes.append(n)
                        user_dist[n] = hop_to_DA
                else:
                    user_nodes.append(n)
        start_nodes = random.sample(list(user_nodes), k=min(start_node_number+1, len(list(CG.nodes()))))
        if CG.graph["DA"] in start_nodes:
            start_nodes.remove(CG.graph["DA"])
        start_nodes = start_nodes[:start_node_number]
        assert len(start_nodes) == start_node_number

    # print("Starting nodes:", start_nodes)
    print(len(start_nodes))
    print("DA nodes: ", CG.graph["DA"])
    
    # remove starting nodes that 1 hop away to DA (this is for honeypot allocation on nodes, we basically can not do anything is does not have intermediate nodes)
    CG.graph["start_nodes"] = start_nodes
    CG.graph["starting_nodes"] = start_nodes
    for n in CG.nodes():
        if n in start_nodes:
            CG.nodes[n]["node_type"] = "S"
        else:
            CG.nodes[n]["node_type"] = ""

    return CG



def dynamic_graph_setup_inplace(
    fn, seed, start_node_number, blockable_p, budget, no_one_hop
):
    

    graph_file_names = fn + "_" + str(blockable_p) + ".gpickle"
    graphs = {} # /Users/huyngo/Desktop/Research/honeypot/
    graphs[graph_file_names] = read_gpickle("/Users/huyngo/Desktop/Research/honeypot_dynamic/processed_dygraph/" + graph_file_names)
    random.seed(seed)
    CG = graphs[graph_file_names].copy()
    CG.graph["budget"] = budget

    # modification in the way of taking entry node
    start_nodes = []
    
    if start_node_number == -1:
        for n in CG.nodes():
            if CG.nodes[n]["label"] == "User":
                if no_one_hop == True:
                    hop_to_DA = nx.shortest_path_length(CG, source=n, target=CG.graph["DA"])
                    
                    if hop_to_DA >= 2:
                        start_nodes.append(n)
                else:
                    start_nodes.append(n)
    else:
        user_nodes = []
        for n in CG.nodes():
            if CG.nodes[n]["label"] == "User":
                if no_one_hop == True:
                    hop_to_DA = nx.shortest_path_length(CG, source=n, target=CG.graph["DA"])
                    print(hop_to_DA)
                    if hop_to_DA >= 2:
                        user_nodes.append(n)
                else:
                    user_nodes.append(n)
        start_nodes = random.sample(list(user_nodes), k=min(start_node_number+1, len(list(CG.nodes()))))
        if CG.graph["DA"] in start_nodes:
            start_nodes.remove(CG.graph["DA"])
        start_nodes = start_nodes[:start_node_number]
        assert len(start_nodes) == start_node_number
        
    print("Starting nodes:", start_nodes)
    print(len(start_nodes))
    print("DA nodes: ", CG.graph["DA"])
    
    # remove starting nodes that 1 hop away to DA (this is for honeypot allocation on nodes, we basically can not do anything is does not have intermediate nodes)
    CG.graph["start_nodes"] = start_nodes
    CG.graph["starting_nodes"] = start_nodes
    for n in CG.nodes():
        if n in start_nodes:
            CG.nodes[n]["node_type"] = "S"
        else:
            CG.nodes[n]["node_type"] = ""
            
            
    #assign honey allocatable nodes


    # graph_name = fn + ".gpickle"
    # # modification in the way of taking entry node
    # write_gpickle(CG, f"//Users/huyngo/Desktop/Research/honeypot_dynamic/processed_dygraph/dy{graph_name}.gpickle")
    
    return CG



def get_shortest_graph_in_place(CG):
    # an alternative for the get_spath_graph with out finding all shortest path graph
    graph_original = nx.DiGraph()
    graph_original.add_edges_from(CG.edges)



    
    starting_nodes = []
    for v in CG.nodes():
        if CG.nodes[v]["node_type"] == "S":
            starting_nodes.append(v)
    DA = CG.graph["DA"]
    print(len(starting_nodes))

    path_all = []
    path_number_dict = dict()

    for entry in starting_nodes:
        # print([p for p in nx.all_shortest_paths(CG,source=entry,target=DA)])
        temp = list(nx.all_shortest_paths(CG,source=entry,target=DA))
        path_number_dict[entry] = len(temp)
        path_all = path_all + temp

    edge_list = extract_edge(path_all)
    graph_condensed = nx.DiGraph()
    graph_condensed.add_edges_from(edge_list)



    shortest_nodes = []
    for n in graph_condensed.nodes():
        shortest_nodes.append(n)
    for u, v in graph_original.edges():
        if (u, v) in graph_condensed.edges():
            graph_original[u][v]["shortest_edges"] = True
        else: 
            graph_original[u][v]["shortest_edges"] = False



    for v in graph_original.nodes():
        graph_original.nodes[v]["node_type"] = CG.nodes[v]["node_type"]
        graph_original.nodes[v]["blockable"] = CG.nodes[v]["blockable"]
    # for u, v in graph_condensed.edges():
    #     graph_condensed[u][v]["blockable"] = CG[u][v]["blockable"]
    
    graph_original.graph["DA"] = CG.graph["DA"]
    graph_original.graph["budget"] = CG.graph["budget"]
    graph_original.graph["start_nodes"] = CG.graph["start_nodes"]



    graph_original.graph["starting_nodes"] = starting_nodes
    graph_original.graph["path_all"] = path_all
    graph_original.graph["path_number_dict"] = path_number_dict
    graph_original.graph["shortest_nodes"] = shortest_nodes
    if "blockable_nodes" in CG.graph.keys():
        graph_original.graph["blockable_nodes"] = CG.graph["blockable_nodes"]
    if "node_to_feature_id" in CG.graph:
        graph_original.graph["node_to_feature_id"] = CG.graph["node_to_feature_id"]  
    return graph_original


def normal_hassession_prob(CG):
    return [0.5]*len(CG.graph["hassession_to_idx"])

def binomial_hassession_prob(CG):
    user_nodes = []
    comp_nodes = []
    other_nodes = []
    DA = CG.graph["DA"]
    hassession_to_idx = CG.graph["hassession_to_idx"]
    for n in CG.nodes():
        if CG.nodes[n]["label"] == "User":
            user_nodes.append(n)
        elif CG.nodes[n]["label"] == "Computer":
            comp_nodes.append(n)
        else:
            other_nodes.append(n)
    user_nodes.append(DA)
    max_sessions_per_user = 3
    prob = int(math.ceil(math.log10(len(user_nodes))))/len(comp_nodes)
    prob = [prob]*len(CG.graph["hassession_to_idx"])
    for i in range(len(CG.graph["DA_user"])):
        num_sessions = random.randrange(0, max_sessions_per_user)
        num_sessions = max(num_sessions, 1)
        for c in random.sample(comp_nodes, num_sessions):
            if (c, DA) not in hassession_to_idx:
                continue
            else:
                prob[hassession_to_idx[(c, DA)]] = 1
    return prob
