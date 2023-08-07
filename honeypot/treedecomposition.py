import networkx as nx
from itertools import combinations
from patch import topological_generations
from utility import f, is_start, is_blockable, report, display
from functools import lru_cache
from pdb import set_trace as bp


# Algorithm 2
def build_tree_decomposition(CG):
    tree_nodes = []
    G = CG.to_undirected()
    print(list(topological_generations(CG)))
    for layer in topological_generations(CG):
        print("edge list", list(G.edges()))
        for v in layer:
            # print(list(G.neighbors(v)))
            # print(list(combinations(G.neighbors(v), 2)))
            # print(tuple(sorted(G.neighbors(v))))
            G.add_edges_from(combinations(G.neighbors(v), 2))
            tree_nodes.append((v, tuple(sorted(G.neighbors(v))), 0))
            G.remove_node(v)
    # print(tree_nodes)
    print([x for x in tree_nodes])
    print("tree width:", max([len(x[1]) for x in tree_nodes]))
    # ADSDADSSDA
    TD = nx.DiGraph()
    n = len(tree_nodes)
    for i in range(n - 1):
        for j in range(i + 1, n):
            if tree_nodes[j][0] in tree_nodes[i][1]:
                TD.add_edge(tree_nodes[i], tree_nodes[j])
                break
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


def moveon(new_node, knowledge_dict, aux_knowledge, budget):
    new_myself, new_knowledge_nodes, new_aux_flag = new_node
    new_knowledge_vals = tuple(knowledge_dict[i] for i in new_knowledge_nodes)
    new_aux_knowledge = aux_knowledge
    if new_aux_flag == 0:
        new_aux_knowledge = -1
    return go(new_node, new_knowledge_vals, new_aux_knowledge, budget)

def node_best_spending_strat(spending_bag):
    index = -1
    min = 100000
    for i in range(len(spending_bag)):
        if spending_bag[i] != None:
            if spending_bag[i][0] < min:
                index = i
                min = spending_bag[i][0]
        else:
            continue
    return spending_bag[index]    
# need to clear cache
# haskell style terminology: go is the dp function
# @lru_cache(maxsize=None)
def go(node, knowledge_values, aux_knowledge, budget):
    myself, knowledge_nodes, aux_flag = node
    assert len(knowledge_nodes) == len(knowledge_values)
    knowledge_dict = dict(zip(knowledge_nodes, knowledge_values))
    print("current node: ", node)
    print("Budget for this round: ", budget)

    if aux_flag == 0:
        assert aux_knowledge == -1
        shortest_distances = []
        print("successors: ", list(DG.successors(myself)))
        print("myself", myself)
        print("predessessor: ", list(DG.predecessors(myself)))
        print(knowledge_dict)
        for dest in DG.successors(myself):
            # print("dest:", dest)
            distance = knowledge_dict[dest] + 1
            blockable = is_blockable(DG, myself, dest)
            shortest_distances.append((distance, dest, blockable))
        shortest_distances.append((1000000, -1, False)) # infinite distance
        shortest_distances = sorted(shortest_distances)
        # print("shortest_distance:" ,shortest_distances)
        max_spend = 0
        for _, _, blockable in shortest_distances:
            if blockable:
                max_spend += 1
            else:
                break
        assert len(list(TD.predecessors(node))) == 1
        res_list = []
        all_strat = []
        for spend in range(min(budget, max_spend) + 1):
            print("IfSpend: ", spend, "at node: ", node)
            realised_distance = shortest_distances[spend][0]
            pre = list(TD.predecessors(node))[0]
            print("From: ", node, "go to node: ", pre)
            res, node_blocking_strat = go(pre, knowledge_values, realised_distance, budget - spend)
            print("Comback from: ", pre, "back to to: ", node)
            print("res: ", res, "retrieved strat: ", node_blocking_strat)
            all_strat.append(node_blocking_strat)
            if is_start(DG, myself):
                print("Entry Node !!!!!!!!!!!")
                res += f(realised_distance, DG)
            res_list.append(res)
        print("res_list: ",res_list)
        print("node blocking strat: ", all_strat)
        print("At node: ",node, "best spending is:", res_list.index(min(res_list)))
        node_best_spend = res_list.index(min(res_list))
        if node_best_spend != 0:
            # if TD_node_map[node][0] != 0:
            # print(pre)
            # print(TD_node_map[pre])
            # TD_node_map[node][node_best_spend] = node_best_spending_strat(TD_node_map[pre])
            # TD_node_map[node][node_best_spend][1].append((node, res_list.index(min(res_list))))
            # TD_node_map[node][node_best_spend][0] = min(res_list)

            all_strat[res_list.index(min(res_list))].append((node, res_list.index(min(res_list))))
        best_strat = all_strat[res_list.index(min(res_list))]
        print("best current strat: ", best_strat)

        return min(res_list), best_strat
    else:
        assert aux_knowledge != -1
        knowledge_dict[myself] = aux_knowledge
        pres = list(TD.predecessors(node))
        if len(pres) == 0:
            # leaf aux node doesn't contribute
            TD_node_map[node][budget] = [1, []]
            return 0, []
        elif len(pres) == 1:
            print("move on: ", pres[0])
            TD_node_map[node] = TD_node_map[pres[0]]
            return moveon(pres[0], knowledge_dict, aux_knowledge, budget)
        elif len(pres) == 2:
            res_list = []
            possible_node_block = []
            for budget0 in range(budget + 1):
                budget1 = budget - budget0
                temp = []
                res0, node_block_list_1 = moveon(pres[0], knowledge_dict, aux_knowledge, budget0)
                print("current res0: ", res0, "block node: ", node_block_list_1, "budget on this branch:", budget0)
                res1, node_block_list_2 = moveon(pres[1], knowledge_dict, aux_knowledge, budget1)
                print("current res1: ", res1, "block node: ", node_block_list_2, "budget on this branch:", budget1)
                temp = node_block_list_1 + node_block_list_2
                possible_node_block.append(temp)
                res_list.append(res0 + res1)
            print(possible_node_block)
            print(res_list.index(min(res_list)))
            best_blocking_strat = possible_node_block[res_list.index(min(res_list))]
                
            return min(res_list), best_blocking_strat
        else:
            assert False


def dp(CG):
    global DG, TD, TD_node_map, result_strat
    
    # save as a list tuple: (node, budget_spend_on_node)
    DG = CG
    TD = add_aux_nodes(build_tree_decomposition(DG))
    node_list = list(TD.nodes())
    TD_node_map = {node_list[i] : [None for i in range(DG.graph["budget"]+1)] for i in range(len(list(TD.nodes())))}
    # TD_node_map[(6, (4, 5), 0)] = [1, []]
    result_strat = [[] for i in range(len(list(TD.nodes())))]
    print(list(TD.nodes()))
    print(list(CG.nodes()))
    draw_networkx(CG, "CG_example")
    draw_networkx(TD, "TD_example")
    print("Entry Nodes:", DG.graph["start_nodes"])
    print("DA node", DG.graph["DA"])
    # go.cache_clear()
    # bp()
    # wlog to start from the aux node for DA
    unblockable_node = []
    blockable_node = []
    for u, v in DG.edges():
        if DG[u][v]["blockable"] == False:
            unblockable_node.append((u,v))
        else:
            blockable_node.append((u,v))
    print("UNBLOCKABLE NODES", unblockable_node)
    print("BLOCKABLE NODES", blockable_node)
    # print(TD_node_map)
    temp = go((DG.graph["DA"], (), 1), (), 0, DG.graph["budget"])
    print("Entry Nodes:", DG.graph["start_nodes"])
    print("DA node", DG.graph["DA"])
    print(temp)
    return temp[0], temp[1], DG.graph["start_nodes"]

def draw_networkx(G, name):
    nx.draw(G, with_labels = True)
    import matplotlib.pyplot as plt
    plt.savefig(name + ".png")
    plt.clf()

# in graph r500-dag, 48 is domain adminres_list.index(min(res_list)