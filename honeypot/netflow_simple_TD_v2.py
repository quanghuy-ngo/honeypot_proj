# simple but yet not so easy :(
# import graphlib
from re import L
from xxlimited import new
import networkx as nx
from itertools import combinations
from patch import topological_generations
from utility import is_start, is_blockable, report, is_node_blockable, evaluate, draw_networkx, f
from functools import lru_cache
from setupgraph import random_setup, flow_graph_setup, get_spath_graph
from networkx.readwrite.gpickle import write_gpickle, read_gpickle
import networkx as nx
from argparse import ArgumentParser
from timeout import timeout
# def parse_args():
#     parser = ArgumentParser(prog="ShotHound", prefix_chars="-/", add_help=False, description=f'Finding practical paths in BloodHound')
#     parser.add_argument('budget', type = int)
#     parser.add_argument('start', type = int)
#     parser.add_argument('seed', type = int)
#     args = parser.parse_args()
#     return args

# args_input = parse_args()
# args = {
#     # "fn": "examplegraph_2",
#     "fn": "r2000",
#     "budget": args_input.budget,
#     "start_node_number": args_input.start,
#     "seed": args_input.seed,
#     "blockable_p": 1,   # blockable_p = 1 means that all are blockable
#     "double_edge_p": 0, # if double_edge = 0, no double edge appear
#     # "cross_path_p": 0,
#     "multi_block_p": 1,
# }

def tree_width(CG):
    tree_nodes = []
    G = CG.to_undirected()
    for layer in topological_generations(CG):
        for v in layer:
            G.add_edges_from(combinations(G.neighbors(v), 2))
            tree_nodes.append((v, tuple(sorted(G.neighbors(v))), 0))
            G.remove_node(v)
    # print(tree_nodes)
    print("tree width:", max([len(x[1]) for x in tree_nodes]))

    return "tree width: " + str(max([len(x[1]) for x in tree_nodes]))

def build_tree_decomposition(CG):
    tree_nodes = []
    G = CG.to_undirected()
    for layer in topological_generations(CG):
        for v in layer:
            G.add_edges_from(combinations(G.neighbors(v), 2))
            tree_nodes.append((v, tuple(sorted(G.neighbors(v))), 0))
            G.remove_node(v)
    # print(tree_nodes)
    print("tree width:", max([len(x[1]) for x in tree_nodes]))

    TD = nx.DiGraph()
    n = len(tree_nodes)
    for i in range(n - 1):
        for j in range(i + 1, n):
            if tree_nodes[j][0] in tree_nodes[i][1]:
                TD.add_edge(tree_nodes[i], tree_nodes[j])
                break
    # print(tree_nodes)
    return TD


def add_aux_nodes(TD):
    for u in list(TD.nodes()):
        assert u[2] == 0
        u_prime = (u[0], u[1], 1)
        for pre in list(TD.predecessors(u)):
            TD.remove_edge(pre, u)
            TD.add_edge(pre, u_prime)
        TD.add_edge(u_prime, u)
    # make tree binary
    not_binary = True
    while not_binary:
        not_binary = False
        for u in list(TD.nodes()):
            pres = list(TD.predecessors(u))
            if len(pres) > 2:
                u_prime = (u[0], u[1], u[2] + 1)
                TD.add_edge(u_prime, u)
                for pre in pres[1:]:
                    TD.remove_edge(pre, u)
                    TD.add_edge(pre, u_prime)
                not_binary = True
                break

    return TD


#####################################################################
# TODO: fixed paramter algorithm style for dynamic programing on tree decomposition for flow Problem
# not quite a flow problem

def moveon_flow(new_node,  num_blocked, num_all, current_num_blocked, current_num_all, budget):
    new_myself, new_knowledge_nodes, new_aux_flag = new_node
    new_num_blocked = tuple(num_blocked[i] for i in new_knowledge_nodes)
    new_num_all = tuple(num_all[i] for i in new_knowledge_nodes)
    # new_aux_knowledge = aux_knowledge
    if new_aux_flag == 0:
        new_aux_knowledge = -1
    return go_flow(new_node, new_num_blocked, new_num_all, current_num_blocked, current_num_all,  budget)


# need to clear cache
# haskell style terminology: go is the dp function
# knowledge value will show how many 


@lru_cache(maxsize=None)
def go_flow(node, num_blocked, num_all, current_num_blocked, current_num_all, budget):
    myself, knowledge_nodes, aux_flag = node
    assert len(knowledge_nodes) == len(num_blocked)
    assert len(knowledge_nodes) == len(num_all)
    # print("##############")
    # print(knowledge_values)
    # print("##############")
    num_all_dict = dict(zip(knowledge_nodes, num_all))
    num_blocked_dict = dict(zip(knowledge_nodes, num_blocked))
    if aux_flag == 0:
        # if current node have out degree == 1: remain the infomation
        # if current node have out degree > 1: 
        # other node in bag only remain and pass to the later node.      
        current_num_blocked = 0
        current_num_all = 0
        for dest in DG.successors(myself):
            if dest == DG.graph["DA"]:
                current_num_all = 1
            else:
                current_num_blocked += num_blocked_dict[dest]
                current_num_all += num_all_dict[dest]
            
        max_spend = 0
        if is_node_blockable(DG, myself) == True and budget > 0:
            max_spend = 1
            
        res_list = []
        for spend in range(max_spend + 1):
            if spend == 1:
                # if current node is being blocked then it mean nu_blocked_path = to all_path_num
                current_num_blocked = current_num_all
            # realised_distance = shortest_distances[spend][0]
            pre = list(TD.predecessors(node))[0]
            res = go_flow(pre, num_blocked, num_all, current_num_blocked, current_num_all, budget - spend)
            if is_start(DG, myself):
                res += f(DG, current_num_blocked, myself)
            res_list.append(res)
        
        return min(res_list)
    
    else:
        # assert aux_knowledge != -1
        num_blocked_dict[myself] = current_num_blocked
        num_all_dict[myself] = current_num_all
        pres = list(TD.predecessors(node))
        if len(pres) == 0:
            # leaf aux node doesn't contribute
            return 0
        elif len(pres) == 1:
            return moveon_flow(pres[0], num_blocked_dict, num_all_dict, current_num_blocked, current_num_all, budget)
        elif len(pres) == 2:
            res_list = []
            
            for budget0 in range(budget + 1):
                budget1 = budget - budget0
                res0 = moveon_flow(pres[0], num_blocked_dict, num_all_dict, current_num_blocked, current_num_all, budget0)
                res1 = moveon_flow(pres[1], num_blocked_dict, num_all_dict, current_num_blocked, current_num_all, budget1)
                res_list.append(res0 + res1)

            return min(res_list)
        else:
            assert False

# @timeout(300)
def dp_flow(CG):
    global DG, TD
    DG = CG
    # draw_networkx(DG, "condensed_graph")
    TD = add_aux_nodes(build_tree_decomposition(DG))
    # draw_networkx(TD, "TD_condensed_graph")
    go_flow.cache_clear()
    # wlog to start from the aux node for DA

    score = go_flow((DG.graph["DA"], (), 1), (), (), 0, 1, DG.graph["budget"])
    print("dp score:", score)
    return score






#######################################################################################
# TODO: Approximate algorithm version for tree decomposition based DP technique

def moveon_flow_v2(new_node,  num_blocked, num_all, current_num_blocked, current_num_all, budget):
    new_myself, new_knowledge_nodes, new_aux_flag = new_node
    new_num_blocked = tuple(num_blocked[i] for i in new_knowledge_nodes)
    new_num_all = tuple(num_all[i] for i in new_knowledge_nodes)
    # new_aux_knowledge = aux_knowledge
    if new_aux_flag == 0:
        new_aux_knowledge = -1
    return go_flow_v2(new_node, new_num_blocked, new_num_all, current_num_blocked, current_num_all,  budget)


# need to clear cache
# haskell style terminology: go is the dp function
# knowledge value will show how many 

@lru_cache(maxsize=None)
def go_flow_v2(node, num_blocked, num_all, current_num_blocked, current_num_all, budget):
    myself, knowledge_nodes, aux_flag = node
    assert len(knowledge_nodes) == len(num_blocked)
    assert len(knowledge_nodes) == len(num_all)
    # print("##############")
    # print(knowledge_values)
    # print("##############")
    num_all_dict = dict(zip(knowledge_nodes, num_all))
    num_blocked_dict = dict(zip(knowledge_nodes, num_blocked))
    if aux_flag == 0:
        # if current node have out degree == 1: remain the infomation
        # if current node have out degree > 1: 
        # other node in bag only remain and pass to the later node.      
        current_num_blocked = 0
        current_num_all = 0
        for dest in DG.successors(myself):
            if dest == DG.graph["DA"]:
                current_num_all = 1
            else:
                current_num_blocked += num_blocked_dict[dest]
                current_num_all += num_all_dict[dest]
            
        max_spend = 0
        if is_node_blockable(DG, myself) == True and budget > 0:
            max_spend = 1
        res_list = []
        for spend in range(max_spend + 1):
            if spend == 1:
                # if current node is being blocked then it mean nu_blocked_path = to all_path_num
                current_num_blocked = current_num_all
            # realised_distance = shortest_distances[spend][0]
            pre = list(TD.predecessors(node))[0]
            res = go_flow_v2(pre, num_blocked, num_all, current_num_blocked, current_num_all, budget - spend)
            if is_start(DG, myself):
                res += f(DG, current_num_blocked, myself)
            res_list.append(res)
        
        return min(res_list)


    else:
        # assert aux_knowledge != -1
        num_blocked_dict[myself] = current_num_blocked
        num_all_dict[myself] = current_num_all
        pres = list(TD.predecessors(node))
        if len(pres) == 0:
            # leaf aux node doesn't contribute
            return 0
        elif len(pres) == 1:
            return moveon_flow_v2(pres[0], num_blocked_dict, num_all_dict, current_num_blocked, current_num_all, budget)
        elif len(pres) == 2:
            res_list = []
            
            for budget0 in range(budget + 1):
                budget1 = budget - budget0
                res0 = moveon_flow_v2(pres[0], num_blocked_dict, num_all_dict, current_num_blocked, current_num_all, budget0)
                res1 = moveon_flow_v2(pres[1], num_blocked_dict, num_all_dict, current_num_blocked, current_num_all, budget1)
                res_list.append(res0 + res1)

            return min(res_list)
        else:
            assert False


def dp_flow_v2(CG):
    global DG, TD
    DG = CG
    draw_networkx(DG, "condensed_graph")
    TD = add_aux_nodes(build_tree_decomposition(DG))
    draw_networkx(TD, "TD_condensed_graph")
    go_flow.cache_clear()
    # wlog to start from the aux node for DA
    score = go_flow_v2((DG.graph["DA"], (), 1), (), (), 0, 1, DG.graph["budget"])
    print("dp score:", score)
    return score

# if __name__ == "__main__":
#     global CG, starting_nodes
#     # args["seed"] = 2
#     CG = flow_graph_setup(**args)

#     # CG = read_gpickle("/home/andrewngo/Desktop/CFR/Code/AAAI/examplegraph_2.gpickle")

#     graph_condensed = get_spath_graph(CG)
#     report(graph_condensed)
#     print("Start Tree Decomposition")
#     node_block_list = dp_flow(graph_condensed)
#     print(node_block_list)
#     # print(evaluate(graph_condensed, node_block_list))
