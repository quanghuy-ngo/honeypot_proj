from platform import node
from re import S
from tracemalloc import start
from networkx.readwrite.gpickle import read_gpickle
import random
from utility import is_blockable, is_start, report, remove_dead_nodes
import networkx as nx
from patch import topological_generations
import copy
import itertools as it
import numpy as np
import math
from scipy.stats import lognorm, expon, truncnorm
import time

def get_snapshot(auth_list, hassession_to_idx, end, hour_per_snapshot = 1):
    current_time = 0
    snapshot_list = []
    while(current_time < end):
        hassession_list = [0 for i in range(len(hassession_to_idx))]
        for i in range(len(auth_list)):
            start_time = auth_list[i][0]
            end_time = auth_list[i][1]
            user = auth_list[i][2]
            comp = auth_list[i][3]
            if current_time > start_time and current_time < end_time:
                hassession_list[hassession_to_idx[(comp, user)]] = 1
        snapshot_list.append(hassession_list)
        current_time += hour_per_snapshot*60*60
    return snapshot_list

def get_online_node(online_edge):
    online_node = []
    for node in online_edge:
        if len(online_edge[node]) != 0:
            online_node.append(node)
    return online_node

def get_frequent_edge_uniform(comp_nodes, frequent_comp, online_comp, isAdd = True):
    prob = []
        #set prob for newedge
    # print(frequent_comp)
    for i in range(len(comp_nodes)):
        if isAdd == True:
            if comp_nodes[i] in frequent_comp:
                prob.append(0)
            else:
                prob.append(1./(len(comp_nodes)-len(frequent_comp)))
        else:
            if comp_nodes[i] not in frequent_comp :
                prob.append(0)
            elif comp_nodes[i] in online_comp:
                prob.append(0)
            else:
                prob.append(1./(len(frequent_comp)-len(online_comp)))
    return prob

def get_new_user_uniform(user_nodes, frequent_edge, online_edge):
    prob = []
    available_user = 0
    for i in range(len(user_nodes)):
        if len(online_edge[user_nodes[i]]) < len(frequent_edge[user_nodes[i]]):
            available_user += 1
        
    for i in range(len(user_nodes)):
        if len(online_edge[user_nodes[i]]) < len(frequent_edge[user_nodes[i]]):
            prob.append(1./available_user)
        else:
            prob.append(0)
    return prob


def get_new_comp_uniform(cur_user, comp_nodes, frequent_edge, online_edge):
    prob = []
    
    frequent_comp = frequent_edge[cur_user]
    online_comp = online_edge[cur_user]
    for i in range(len(comp_nodes)):
        if comp_nodes[i] in frequent_comp and comp_nodes[i] not in online_comp:
            prob.append(1./(len(frequent_comp)-len(online_comp)))
        else: 
            prob.append(0)
    return prob


def set_single_prob_uniform(user_nodes, online_edge):
    online_nodes = get_online_node(online_edge)
    prob = []
    for i in range(len(user_nodes)):
        if user_nodes[i] in online_nodes:
            prob.append(0)
        else:
            prob.append(1./(len(user_nodes) - len(online_nodes)))
    return prob

def set_multi_prob_uniform(user_nodes, online_edge):
    online_nodes = get_online_node(online_edge)
    prob = []
    for i in range(len(user_nodes)):
        if user_nodes[i] not in online_nodes:
            prob.append(0)
        else:
            prob.append(1./(len(online_nodes)))
    return prob



def set_prob_decay(cur_user, comp_nodes, online_edge, edge_node_repeat):
    edge_repeat = edge_node_repeat[cur_user]
    count_list = [None for i in range(len(comp_nodes))]
    count_rep = []
    for i in range(len(comp_nodes)):
        edge = (cur_user, comp_nodes[i])
        if edge not in online_edge:
            count_list[i] = f(edge_repeat[i])
        else:
            count_list[i] = 0
    prob = []
    for i in range(len(count_list)):
        prob.append(count_list[i]/sum(count_list))
    return prob
            
            
def bimodal_exp_norm_rv(mu_norm, sigma_norm, scale_exp, lower, upper, ratio=0.65):
    X_1 = expon(scale=scale_exp)
    X_2 = truncnorm(
        (lower - mu_norm) / sigma_norm, (upper - mu_norm) / sigma_norm, loc=mu_norm, scale=sigma_norm)
    X = np.concatenate((X_1.rvs(1), X_2.rvs(1)), axis=None)
    rv = np.random.choice(X, size=1, p=[1-ratio, ratio])
    return rv[0]


def bimodal_exp_norm_rv_list(mu_norm, sigma_norm, scale_exp, lower, upper, length, ratio=0.5):
    X_1 = expon(scale=scale_exp)
    X_2 = truncnorm(
        (lower - mu_norm) / sigma_norm, (upper - mu_norm) / sigma_norm, loc=mu_norm, scale=sigma_norm)
    X_1_length = length*0.5
    X_2_length = length*X_1_length
    X = np.concatenate((X_1.rvs(X_1_length), X_2.rvs(X_2_length)), axis=None)
    X = random.shuffle(X)
    return X

def f(x, m = 10000, k = 2, s = 10):
#     if x == 0:
#         return 1000
    func = m/(1+np.exp(k*(x-s)))
    if func < 1:
        func = 1
    return func


'''
generate edges with interarrival first create a first snap shot of network
The idea is each node will have 1 primary or frequent edge, this frequent edge will be reconnected
with a decay probability, i.e, after a while of connecting to a computer, user will "switched" to a new primary
computer, a user can assumed to have many primary computer, but here, we only assume each will have 1 primary
User can also have connection to multiple machine at a time (node degree > 1) but with shorter session time and 

The idea of version 2 is to improve the model in term of additional of new edge. The idea is network will have a 
set of frequent "edge" which in time will be gradually subtituted by new edges. The new model able us to modify
the new edge rate of the network by changing new_edge_rate parameter. And by adding new edge to a network, we 
also remove the frequent edge in the network by the same rate to balance out the number of frequent edge in the
network.

5 parameter: interarrival distribution, session duration distribution, new_edge_rate, multiedge probability
maximum user node degree. 
'''



def generate_temporal_sample_2(G, user_nodes, comp_nodes, hassession_to_idx, duration, new_edge_rate_mean, 
                            interarrival_mean):
    # initial snapshot
    # session_duration = 4*60*60
    # new_edge_rate = 30 # new edge per day 
    # generate interarrival list 
    interarrival_list = []
    cur_primary_edge = dict() # list of previous primary edge mapping to user id
    prev_primary_edge = dict() #list of previous primary edge with starting time
    primary_node_prob = [1./(len(user_nodes)) for i in range(len(user_nodes))]
    
    user_id = {user_nodes[i] : i for i in range(len((user_nodes)))}
    comp_id = {comp_nodes[i] : i for i in range(len((comp_nodes)))}
    
    online_edge = {user_nodes[i]: [] for i in range(len(user_nodes))}

    user_to_list = {user_nodes[i]: i for i in range(len(user_nodes))}
    list_to_user = {i: user_nodes[i] for i in range(len(user_nodes))}
    online_auth = []
    edge_node_repeat = {user_nodes[i]: [0 for i in range(len(comp_nodes))] for i in range(len(user_nodes))}
    auth_list = []
    
    hassession_comp = dict()
    for edge in hassession_to_idx:
        user = edge[1] 
        if user not in hassession_comp:
            hassession_comp[user] = []
        hassession_comp[user].append(edge[0])
    
    # initialise a list of "frequent" edge  
    frequent_edge = {user: [] for user in user_nodes}
    frequent_da_edge = {user: [] for user in user_nodes}
    used_edge = {user: [] for user in user_nodes}
    num_session = int(math.ceil(math.log10(len(user_nodes))))
    DA_session = (len(G.graph["DA_user"])*num_session)/2
    if DA_session > len(hassession_comp[G.graph["DA"]])/2:
        DA_session = len(hassession_comp[G.graph["DA"]])/2
    
    for user in user_nodes:
        if user == G.graph["DA"]:
            for c in random.sample(hassession_comp[user], DA_session):
                frequent_da_edge[user].append(c)
                frequent_edge[user].append(c)
                # used_edge[user].append(c)
        else:
            for c in random.sample(hassession_comp[user], num_session):
                frequent_edge[user].append(c)
                # used_edge[user].append(c)
    interarrival_list = []
    auth_list = []
    #generate interarrival data for new edge
    ia_newedge_list = []
    sum_interarrival = 0
    print(new_edge_rate_mean)
    while(sum_interarrival < duration):
        interarrival = lognorm.rvs(1.5, scale = (24*60*60)/new_edge_rate_mean,  size=1)
        sum_interarrival += interarrival[0]
        ia_newedge_list.append(sum_interarrival)
    print(ia_newedge_list[-1])
    print(len(ia_newedge_list))
    # dsadsdasa
    new_edge_count = 0
    sum_interarrival = 0
    while(sum_interarrival < duration):
        # start_time = time.time()
        interarrival = lognorm.rvs(1.5, scale = interarrival_mean,  size=1)
        sum_interarrival += interarrival[0]
        current_time = sum_interarrival
    
        print(current_time/(24*60*60))
        # generate sesion_duration 
        mu_norm, sigma_norm = 8, 1.6
        scale_exp = 4
        lower, upper = 0, 100
        session_duration = bimodal_exp_norm_rv(mu_norm, sigma_norm, scale_exp, lower, upper, ratio=0.35)*60*60
        end_time = current_time + session_duration
        # interarrival_list.append(interarrival)
        # renew the online_auth every iteration
        temp = []
        # print("--- %s seconds ---" % (time.time() - start_time))
        # print(len(auth_list))
        for i in range(len(online_auth)):
            end = online_auth[i][1]
            user = online_auth[i][2]
            comp = online_auth[i][3]
            if end < current_time:
                online_edge[user].remove(comp)
            else:
                temp.append(online_auth[i])

        online_auth = temp
        online_nodes = get_online_node(online_edge)
        # change frequent edge set 
        
        while (new_edge_count < len(ia_newedge_list) and float(ia_newedge_list[new_edge_count]) < current_time):
            #choosing new edge
            temp = []
            for i in range(len(user_nodes)):
                if len(online_edge[user_nodes[i]]) == len(frequent_edge[user_nodes[i]]):
                    temp.append(0)
                else:
                    temp.append(1)
            prob = [temp[i]/sum(temp) for i in range(len(temp))]
            user = np.random.choice(user_nodes, p = prob)
            
            frequent_comp = frequent_edge[user]
            online_comp = online_edge[user]
            # Choose edge to remove   
            # print(len(frequent_comp))
            prob = get_frequent_edge_uniform(hassession_comp[user], frequent_comp, online_comp, False)
            rm_comp = np.random.choice(hassession_comp[user], p = prob)
            frequent_edge[user].remove(rm_comp)
            # Cho0se edge too add
            prob = get_frequent_edge_uniform(hassession_comp[user], frequent_comp, online_comp, True)
            new_comp = np.random.choice(hassession_comp[user], p = prob)
            frequent_edge[user].append(new_comp)
            # used_edge[user].append()
            new_edge_count += 1

        # small tweak to make sure there is enough edge to AD
        if len(online_edge[G.graph["DA"]]) < len(G.graph["DA_user"])//2:
            cur_user = G.graph["DA"]
        
        else:
            prob = get_new_user_uniform(user_nodes, frequent_edge, online_edge)
            cur_user = np.random.choice(user_nodes, p = prob)
        print(cur_user)
        print(len(G.graph["DA_user"]))
        prob = get_new_comp_uniform(cur_user, hassession_comp[cur_user], frequent_edge, online_edge)
        cur_comp = np.random.choice(hassession_comp[cur_user], p = prob)
        # print("cur_comp: ", cur_comp)
        edge = (cur_user, cur_comp)
        online_edge[cur_user].append(cur_comp)
        

        auth_list.append((current_time, end_time, cur_user, cur_comp))
        online_auth.append((current_time, end_time, cur_user, cur_comp))
    return auth_list

