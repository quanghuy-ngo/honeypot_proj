from utility import f, evaluate, is_start, remove_start, is_blockable
from networkx.algorithms.shortest_paths.generic import shortest_path
from networkx.exception import NetworkXNoPath


def trivialFPT(CG, best_trivial, temp, budget=None):
    if budget is None:
        budget = CG.graph["budget"]
    if budget == 0:
        return evaluate(CG)

    start = None
    # print("START_NODES: ", CG.graph["start_nodes"])
    for v in CG.graph["start_nodes"]:
        if is_start(CG, v):
            start = v
            break
    if start is None:
        return 0

    try:
        path = shortest_path(CG, start, CG.graph["DA"])
        cost = f(len(path) - 1, CG)
    except NetworkXNoPath:
        path = None
        cost = 0

    res = []
    CG_copy = CG.copy()

    #Two options:
    temp_dict = {}
    #1. ignore

    temp_node = []
    temp_node.append((start, 0))
    remove_start(CG_copy, start)
    res.append(cost + trivialFPT(CG_copy, best_trivial, temp_node, budget))
    temp_dict[res[0]] = temp_node
    print("RES_TRIVIAL: ", res)

    #2. block at least one edge

    if path is not None:
        for i in range(len(path) - 1):
            temp_edge = []
            if is_blockable(CG, path[i], path[i + 1]):
                CG_copy = CG.copy()
                temp_edge.append((path[i], path[i + 1]))
                CG_copy.remove_edge(path[i], path[i + 1])
                res.append(trivialFPT(CG_copy, best_trivial, temp_edge, budget - 1))
                temp_dict[res[len(res) - 1]] = temp_edge
    temp.extend(temp_dict[min(res)])
    return min(res)

def trivialFPT_helper(CG, budget = None):
    best_trivial = {}
    temp = []

    result = trivialFPT(CG, best_trivial, temp, budget = None)

    return result, temp


