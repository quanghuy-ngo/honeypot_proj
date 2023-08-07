from utility import remove_dead_nodes, is_blockable, set_blockable, is_start, f
from itertools import product
from functools import lru_cache
from networkx.algorithms.shortest_paths.generic import shortest_path
from networkx.exception import NetworkXNoPath


def get_cf_max():
    cf_max = []
    for v in split_nodes:
        cf_max.append(DG.out_degree(v) - 1)
    return tuple(cf_max)


def earliest_blockable_edge(CG, path):
    res = []
    for i in range(len(path) - 1):
        if not CG.has_edge(path[i], path[i + 1]):
            return None
        if is_blockable(CG, path[i], path[i + 1]):
            res.append((path[i], path[i + 1]))
    return res


@lru_cache(maxsize=None)
def path_to_split_node_or_DA(u, v):
    if v in split_nodes or v == DG.graph["DA"]:
        return [u, v]
    else:
        future = list(DG.successors(v))
        assert len(future) == 1
        return [u] + path_to_split_node_or_DA(v, future[0])


def set_realised_distances(CG, cf, cf_dict):
    for u in split_nodes:
        if cf_dict[u] is not None:
            path = path_to_split_node_or_DA(u, cf_dict[u])
            for i in range(len(path) - 1):
                set_blockable(CG, path[i], path[i + 1], False)

    CG.nodes[CG.graph["DA"]]["dist"] = 0
    penalty = 0
    known_continue = True
    while known_continue:
        known_continue = False
        for u in split_nodes:
            if cf_dict[u] is not None and "dist" not in CG.nodes[u]:
                path = path_to_split_node_or_DA(u, cf_dict[u])
                if "dist" in CG.nodes[path[-1]]:
                    CG.nodes[u]["dist"] = CG.nodes[path[-1]]["dist"]
                    CG.nodes[u]["dist"] += len(path) - 1
                    known_continue = True
    for u in split_nodes:
        if cf_dict[u] is not None and "dist" not in CG.nodes[u]:
            penalty += 1
    return penalty


def block_dest(CG, u, v, cf_dict):
    spent = []
    penalty = 0
    if "dist" in CG.nodes[u]:
        achieved = CG.nodes[u]["dist"]
    else:
        achieved = 1000000
    path = path_to_split_node_or_DA(u, v)
    if "dist" in CG.nodes[path[-1]]:
        current_dist = len(path) - 1 + CG.nodes[path[-1]]["dist"]
        if current_dist < achieved:
            to_block = earliest_blockable_edge(CG, path)
            if to_block is None:
                pass
            elif len(to_block) == 0:
                penalty += 1
            else:
                CG.remove_edge(to_block[-1][0], to_block[-1][1])
                spent.append((to_block[-1][0], to_block[-1][1]))
    return spent, penalty


def build_cf_dict(cf):
    cf_dict_raw = dict(zip(split_nodes, cf))
    cf_dict = {}

    for u in split_nodes:
        if cf_dict_raw[u] > DG.out_degree(u) - 1:
            assert False

    for u in split_nodes:
        if cf_dict_raw[u] < 0:
            cf_dict[u] = None
        else:
            cf_dict[u] = sorted(list(DG.successors(u)))[cf_dict_raw[u]]
    return cf_dict


def split_node_not_picked_edges(u, cf_dict):
    all_dests = sorted(list(DG.successors(u)))
    if cf_dict[u] is not None:
        all_dests.remove(cf_dict[u])
    return all_dests


@lru_cache(maxsize=None)
def classification(cf):
    CG = DG.copy()
    cf_dict = build_cf_dict(cf)
    penalty = set_realised_distances(CG, cf, cf_dict)
    spent = []
    for u in split_nodes:
        for v in split_node_not_picked_edges(u, cf_dict):
            spent_new, penalty_new = block_dest(CG, u, v, cf_dict)
            spent.extend(spent_new)
            penalty = penalty + penalty_new

    # make it a tree
    for u in split_nodes:
        for v in split_node_not_picked_edges(u, cf_dict):
            if CG.has_edge(u, v):
                CG.remove_edge(u, v)
        remove_dead_nodes(CG)
    deficit = max(len(spent) - DG.graph["budget"], 0)
    res, frontier_edges = frontier_tree(CG, spent)
    # print("RESULT FRONTIER: ", res)
    return (res, penalty, deficit, frontier_edges.copy())


def cf_setup(CG):
    global DG, split_nodes
    DG = CG
    split_nodes = DG.graph["split_nodes"]
    classification.cache_clear()
    cached_evaluate.cache_clear()
    path_to_split_node_or_DA.cache_clear()


def all_classifications(CG):
    best_edges = []
    cf_setup(CG)
    cfs = product(*[range(-1, DG.out_degree(v)) for v in split_nodes])
    print(type(cfs))
    # print("CFSSSSS: ", cfs[0])
    counter = 0
    min_so_far = 1
    for cf in cfs:
        temp = classification(cf)
        print("CHECK CLASS: ", temp)
        # res = classification(cf)[0]
        res = temp[0]
        if res < min_so_far:
            min_so_far = res
            # print()
            best_edges = temp[3].copy()
            print(min_so_far, cf)
        counter += 1
        if counter == 1000:
            print("all classification giveup")
            break
    return min_so_far, best_edges


@lru_cache(maxsize=None)
def cached_evaluate(spent):
    res = 0
    DG_copy = DG.copy()
    for e in spent:
        DG_copy.remove_edge(e[0], e[1])
    for v in DG.nodes():
        if is_start(DG, v):
            try:
                path = shortest_path(DG_copy, v, DG.graph["DA"])
                res += f(len(path) - 1, DG)
            except NetworkXNoPath:
                pass
    return res


def frontier_tree(T, spent):
    if DG.graph["budget"] <= len(spent):
        print("Go here")
        return cached_evaluate(tuple(sorted(spent[: DG.graph["budget"]]))), list(spent)

    budget = DG.graph["budget"] - len(spent)
    for v in T.nodes():
        assert T.out_degree(v) == 1 or v == DG.graph["DA"]

    # no need to copy tree, modify in place, tree is thrown away anyway
    frontier_tree_go(T, DG.graph["DA"], True)
    evaluate_tree_go(T, DG.graph["DA"], 0)

    candidates = []
    for u, v in T.edges():
        if is_blockable(T, u, v):
            candidates.append((T.nodes[u]["value"], (u, v)))
    spent = list(spent) + [x[1] for x in sorted(candidates)[-budget:]]
    result_edges = spent
    # print("FRONTIER: ", spent)
    return cached_evaluate(tuple(sorted(spent))), list(result_edges)


def frontier_tree_go(T, current_root, is_frontier):
    for v in T.predecessors(current_root):
        if T[v][current_root]["blockable"]:
            if is_frontier:
                frontier_tree_go(T, v, False)
            else:
                set_blockable(T, v, current_root, False)
                frontier_tree_go(T, v, False)
        else:
            frontier_tree_go(T, v, is_frontier)


def evaluate_tree_go(T, current_root, dist):
    res = 0
    if is_start(T, current_root):
        res += f(dist, T)
    for v in T.predecessors(current_root):
        res += evaluate_tree_go(T, v, dist + 1)
    T.nodes[current_root]["value"] = res
    return res
