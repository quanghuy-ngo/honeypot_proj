# import graphlib
from re import L
from xxlimited import new
import networkx as nx
from itertools import combinations
from patch import topological_generations
from utility import is_start, is_blockable, report, is_node_blockable, evaluate_flow, draw_networkx
from functools import lru_cache
from setupgraph import random_setup, flow_graph_setup, get_spath_graph
from networkx.readwrite.gpickle import write_gpickle, read_gpickle
import networkx as nx
from argparse import ArgumentParser
import signal

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

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
    return TD, 


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








##########################condense_graph####################################################
#TODO Write TD algorithm for path number problem 
# In path number problem, when alocate 1 budget on a node, we need to re-calculate the path remaining
#


def moveon_flow(new_node, node_block_list_global, node_block_list_local, budget):
    # this funcion intended to 
    new_myself, new_knowledge_nodes, new_aux_flag = new_node
    # print(new_knowledge_vals)
    if new_aux_flag == 0:
        new_aux_knowledge = -1
    return go_flow(new_node, node_block_list_global, node_block_list_local, budget)

# node block list local will return the block list to the nearest join node
# node block list global use to evaluate the blocking strategy


@lru_cache(maxsize=None)
def go_flow(node, node_block_list_global, node_block_list_local, budget):
    myself, knowledge_nodes, aux_flag = node
    # print(node)
    # print(len(node_block_list_global))
    # print(budget)
    # print(node_block_list_local)
    # assert len(knowledge_nodes) == len(knowledge_values)
    # knowledge_dict = dict(zip(knowledge_nodes, knowledge_values))
    if budget == 0 or is_start(DG, myself):
        return list(node_block_list_local)
    if aux_flag == 0:
        blockable_node = is_node_blockable(graph_condensed, myself)
        pre = list(TD.predecessors(node))[0]
        # 2 option, spend 1 to block current node or pass
        res_list = []
        max_spend = 0
        if blockable_node == True:
            max_spend = 1
        new_node_block_list = []
        for spend in range(max_spend + 1):
            node_block_list_global = list(node_block_list_global)
            node_block_list_local = list(node_block_list_local)
            if spend == 1:
                node_block_list_global.append(myself)
                node_block_list_local.append(myself)
            node_block_list_global = tuple(node_block_list_global)
            node_block_list_local = tuple(node_block_list_local)
            res = go_flow(pre, node_block_list_global, node_block_list_local, budget - spend)
            new_node_block_list.append(res)
        
        best_block_strat = new_node_block_list[0]
        best_current_score = evaluate_flow(graph_condensed, new_node_block_list[0])
        for i in range(1,len(new_node_block_list)):
            temp_block = new_node_block_list[i]
            temp = evaluate_flow(graph_condensed, temp_block)
            if temp < best_current_score:
                best_block_strat = new_node_block_list[i]
                best_current_score = temp
        # print(node)
        # print("no spend","actual: ",len(new_node_block_list[0]),"nodetoblocks:",new_node_block_list[0])
        # if blockable_node == True:
        #     print("spend 1","actual: ",len(new_node_block_list[1]),"nodetoblocks:",new_node_block_list[1])
        # print(best_current_score) 

        return list(best_block_strat)

    else:
        pres = list(TD.predecessors(node))
        node_block_list_global = tuple(node_block_list_global)
        node_block_list_local = tuple(node_block_list_local)
        if len(pres) == 0:
            # leaf aux node doesn't contribute
            # print(node)
            return list(node_block_list_local)
        elif len(pres) == 1:
            # just pass to the next node
            return moveon_flow(pres[0], node_block_list_global, node_block_list_local, budget)
        elif len(pres) == 2:
            new_node_block_list = []
            for budget0 in range(budget + 1):
                budget1 = budget - budget0
                node_block_list_1 = moveon_flow(pres[0], node_block_list_global, (), budget0)
                node_block_list_2 = moveon_flow(pres[1], node_block_list_global, (), budget1) 
                # print("budget0:",budget0,"actual: ",len(node_block_list_1),"nodetoblocks:",node_block_list_1)
                # print("budget1:",budget1,"actual: ",len(node_block_list_2),"nodetoblocks:",node_block_list_2)
                assert len(node_block_list_1) <= budget0
                assert len(node_block_list_2) <= budget1
                # print(len(node_block_list_1 + node_block_list_2)) 
                new_node_block_list.append(list(node_block_list_local) + node_block_list_1 + node_block_list_2)
            
            best_block_strat = new_node_block_list[0]
            temp_block = new_node_block_list[0]
            best_current_score = evaluate_flow (graph_condensed, temp_block)
            for i in range(1,len(new_node_block_list)):
                temp_block = new_node_block_list[i]
                temp = evaluate_flow(graph_condensed, temp_block)
                # print(temp_block)
                # print(temp)
                if temp < best_current_score:
                    best_block_strat = new_node_block_list[i]
                    best_current_score = temp
            return best_block_strat
        else:
            assert False




def dp_flow(CG):
    global DG, TD, graph_condensed
    DG = CG
    graph_condensed = CG
    
    draw_networkx(DG, "condensed_graph")
    TD = add_aux_nodes(build_tree_decomposition(DG))
    draw_networkx(TD, "TD_condensed_graph")
    go_flow.cache_clear()
    node_block_list = go_flow((DG.graph["DA"], (), 1), (), (),DG.graph["budget"])
    final_score = evaluate_flow(graph_condensed, node_block_list)
    print("score:", final_score)
    
    return final_score, node_block_list


###############################################################################


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
#     print(evaluate(graph_condensed, node_block_list))
