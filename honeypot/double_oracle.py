from pulp import LpVariable, LpMaximize, LpProblem, LpStatus, value, lpSum, GUROBI_CMD
import networkx as nx

def ip_block_paths(G, paths, edges):
    prob = LpProblem("IP", LpMaximize)
    variables = {}
    edge_vars = {}
    path_count = 1

    # set up varaibles
    print(edges)
    for path in paths:
        variables[tuple(path)] = LpVariable(f"path_{path_count}", cat="Binary")
        path_count += 1
    for edge in edges.keys():
        if edges[edge] > 1:

            variables[edge]= LpVariable(f"edge_{len(edge_vars) + 1}", cat="Binary")
            edge_vars[edge] = variables[edge]

    prob += lpSum([variables[tuple(path)] for path in paths]), "Maximise blocked paths"

    for path in paths:
        vars = get_vars_on_path(path, variables)
        if not len(vars):
            for i in range(1, len(path) - 1):
                u = path[i]
                v = path[i+1]
                if G[u][v]['blockable']:
                    variables[(u, v)] = LpVariable(f"edge_{len(edge_vars) + 1}", cat="Binary")
                    edge_vars[(u, v)] = variables[(u, v)]
                    vars.append(variables[(u, v)])
                    break
        prob += variables[tuple(path)] <= lpSum(vars)
        prob += variables[tuple(path)] * len(vars) >= lpSum(vars)

    prob += lpSum(list(edge_vars.values())) <= G.graph['budget']
    prob.solve(GUROBI_CMD(msg=False))
    assert LpStatus[prob.status] == "Optimal"

    res = value(prob.objective)
    strategy = set()
    
    for edge in edge_vars.keys():
        var = edge_vars[edge]
        if var.varValue > 0:
            strategy.add(edge)
            assert G[edge[0]][edge[1]]['blockable']
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
    edges_removed = {}
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
            for u, v in edges_removed.keys():
                G.add_edge(u, v, **edges_removed[(u, v)])
            path_edges = {}
            for path, len_temp in paths:
                for i in range(len(path) - 1):
                    if G[path[i]][path[i+1]]['blockable']:
                        if (path[i], path[i+1]) not in path_edges:
                            path_edges[(path[i], path[i+1])] = 0
                        path_edges[(path[i], path[i+1])] += 1
            blocked_count, strategy = ip_block_paths(G, [path[0] for path in paths], path_edges)
            print(strategy)
            if blocked_count < len(paths):
                return path_len
            else:
                edges_removed.clear()
                for u, v in strategy:
                    assert G[u][v]['blockable']
                    edges_removed[(u, v)] = G[u][v]
                    G.remove_edge(u, v)
        else:
            return int(1e10) # infinite length

def add_supersource(G):
    G.add_node('super_source')
    for src in G.graph['start_nodes']:
        G.add_edge('super_source', src, weight=0.0, blockable=False)

def path_weight(G, path):
    res = 0
    for i in range(len(path) - 1):
        res += G[path[i]][path[i + 1]]['weight']
    return res

def main():
    G = nx.DiGraph()
    for u, v in [(1, 2), (2, 3), (1, 3)]:
        G.add_edge(u, v, weight=1.0, blockable=True)
    G.graph['start_nodes'] = [1]
    G.graph['DA'] = 3
    G.graph['budget'] = 1
    res_len = double_oracle(G)
    print(f'res: {res_len}')

if __name__ == '__main__':
    main()