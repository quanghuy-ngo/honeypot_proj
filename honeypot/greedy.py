from utility import evaluate, is_blockable


# def greedy_cf():
#     _, spent = greedy(DG, DG.graph["budget"], [])
#     CG = DG.copy()
#     for u, v in spent:
#         CG.remove_edge(u, v)
#     split_nodes = DG.graph["split_nodes"]
#     print(split_nodes)
#     cf = []
#     for u in split_nodes:
#         correct_dist_u = correct_dist(CG, u)
#         if correct_dist_u is None:
#             cf.append(-1)
#         else:
#             future = sorted(list(DG.successors(u)))
#             for i, v in enumerate(future):
#                 correct_dist_v = correct_dist(CG, v)
#                 if correct_dist_v is not None:
#                     if correct_dist_u == 1 + correct_dist_v:
#                         cf.append(i)
#                         break
#     assert len(cf) == len(split_nodes)
#     return tuple(cf)

# best_greedy = []
def greedy(CG, best_greedy, budget=None):
    if budget is None:
        budget = CG.graph["budget"]
    if budget == 0:
        num = evaluate(CG)
        # print("Last num: ", num)
        return num
    # print("GRAPHHHHHH: ", CG)
    res = []
    CG_dict = {}
    for u, v in CG.edges():
        # print("Edges: ", u, v)
        if is_blockable(CG, u, v):
            # print("Edges blockable: ", u, v)
            CG_copy = CG.copy()
            CG_copy.remove_edge(u, v)
            CG_dict[(u, v)] = CG_copy
            # print("Greedy run: ", CG_dict[(u, v)])
            # print("Greedy run: ", CG_copy)
            res.append((evaluate(CG_copy), (u, v)))
            # print("Result after evaluation: ", res[len(res) - 1])
    if len(res) == 0:
        num = evaluate(CG)
        # print("Last num: ", num)
        return num
    best_res, best_CG_key = min(res)
    # print("Best: ", best_CG_key)
    best_greedy.append(best_CG_key)
    return greedy(CG_dict[best_CG_key], best_greedy, budget - 1)

def greedy_v2(CG, budget = None):
    best_greedy = []
    result = greedy(CG, best_greedy, budget=None)
    print("GREEDY Details: ", best_greedy)
    return result, best_greedy
