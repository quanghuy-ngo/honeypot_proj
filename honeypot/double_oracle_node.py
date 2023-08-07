from pulp import LpVariable, LpMaximize, LpProblem, LpStatus, value, lpSum, GUROBI_CMD
import networkx as nx

def ip_block_paths(G, paths, nodes):
    prob = LpProblem("IP", LpMaximize)
    variables = {}
    node_vars = {}
    path_count = 1

    # set up varaibles
    for path in paths:
        variables[tuple(path)] = LpVariable(f"path_{path_count}", cat="Binary")
        path_count += 1
    for n in nodes.keys():
        if nodes[n] > 1:
            variables[n]= LpVariable(f"node_{len(node_vars) + 1}", cat="Binary")
            node_vars[n] = variables[n]




    prob += lpSum([variables[tuple(path)] for path in paths]), "Maximise blocked paths"

    for path in paths:
        vars = get_vars_on_path(path, variables)
        if not len(vars):
            for i in range(1, len(path) - 1):
                n = path[i]

                if G.nodes[n]['blockable']:
                    variables[n] = LpVariable(f"node_{len(node_vars) + 1}", cat="Binary")
                    node_vars[n] = variables[n]
                    vars.append(variables[n])
                    break
        prob += variables[tuple(path)] <= lpSum(vars)
        prob += variables[tuple(path)] * len(vars) >= lpSum(vars)

    prob += lpSum(list(node_vars.values())) <= G.graph['budget']
    prob.solve(GUROBI_CMD(msg=False))
    assert LpStatus[prob.status] == "Optimal"

    res = value(prob.objective)
    strategy = set()
    
    for n in node_vars.keys():
        var = node_vars[n]
        if var.varValue > 0:
            strategy.add(n)
            assert G.nodes[n]['blockable']
    assert len(strategy) <= G.graph['budget']
    return res, strategy

def get_vars_on_path(path, variables):
    vars = []
    for i in range(len(path) - 1):
        e = (path[i], path[i+1])
        if e in variables:
            vars.append(variables[e])
    return vars

def double_oracle(G):
    add_supersource(G)
    paths = []
    nodes_removed = {}
    strategy = []
    while True:
        if nx.has_path(G, 'super_source', G.graph['DA']):
            path = nx.shortest_path(G, 'super_source', G.graph['DA'], 'weight', 'dijkstra')
            path_len = path_weight(G, path)
            if len(paths) > 0 and path_len < paths[-1][1]:
                temp_paths = []
                for p in paths:
                    if p[1] <= path_len:
                        temp_paths.append(p)
                    else:
                        break
                paths = temp_paths
            paths.append((tuple(path), path_len))

            for n in nodes_removed.keys():
                G.add_node(n)
                G.nodes[n]["blockable"] = True
                for u, v in nodes_removed[n]:
                    G.add_edge(u, v)
            path_nodes = {}
            for path, len_temp in paths:
                for i in range(len(path)):
                    if G.nodes[path[i]]['blockable']:
                        if (path[i], path[i+1]) not in path_nodes:
                            path_nodes[path[i]] = 0
                        path_nodes[path[i]] += 1
            # print(path_nodes)
            blocked_count, strategy = ip_block_paths(G, [path[0] for path in paths], path_nodes)
            # print(strategy)
            if blocked_count < len(paths):
                return strategy, path_len
            else:
                nodes_removed.clear()
                edge_remove = []
                for n in strategy:
                    assert G.nodes[n]['blockable']
                    edges = G.edges(n)
                    nodes_removed[n] = edges
                    edge_remove += edges
                G.remove_edges_from(edge_remove)
        else:
            return strategy, int(1e10) # infinite length

def add_supersource(G):
    G.add_node('super_source')
    G.nodes['super_source']['blockable'] = False
    for src in G.graph['start_nodes']:
        G.add_edge('super_source', src, weight=0.0, blockable=False)

def path_weight(G, path):
    res = 0
    for i in range(len(path) - 1):
        res += 1
    return res

# def main():
#     G = nx.DiGraph()
#     for u, v in [(1, 2), (2, 3), (3, 5), (1, 4), (4, 5) ]:
#         G.add_edge(u, v, weight=1.0)

#     G.graph['start_nodes'] = [1]
#     G.graph['DA'] = 5
#     G.graph['budget'] = 1
#     for n in G.nodes():
#         if n == G.graph['DA'] or n in G.graph['start_nodes']:
#             G.nodes[n]["blockable"] = False
#         else:
#             G.nodes[n]["blockable"] = True
#     res_len = double_oracle(G)
#     print(f'res: {res_len}')

# if __name__ == '__main__':
#     main()