from pulp import *
import networkx as nx
from itertools import combinations
from utility import is_start, is_blockable, report, is_node_blockable, evaluate_flow, evaluate_flow_competent
from setupgraph import random_setup, flow_graph_setup, get_spath_graph
from networkx.readwrite.gpickle import write_gpickle, read_gpickle
import networkx as nx
from argparse import ArgumentParser


def mip_flow(graph_condensed):
    path_all = graph_condensed.graph["path_all"]
    path_number_dict = graph_condensed.graph["path_number_dict"]
    starting_nodes = graph_condensed.graph["starting_nodes"]
    budget = graph_condensed.graph["budget"]
    max_node = 0
    for i in graph_condensed.nodes():
        if i > max_node:
            max_node = i
    # mapping down the 
    # in path matrix, if 1, node of index exist in path
    path_matrix = []
    for i in range(len(path_all)):
        temp = [0 for j in range(max_node+1)]
        for j in path_all[i]:
            if is_node_blockable(graph_condensed, j):
                temp[j] = 1
        path_matrix.append(temp)
    
    # create a dicitonary of path
    path_matrix_dict = dict()
    for i in range(len(path_matrix)):
        path_matrix_dict[i] = path_matrix[i]
    


    path_score = []
    for i in range(len(path_matrix_dict)):
        score = 1/(path_number_dict[path_all[i][0]])*1/(len(starting_nodes))
        path_score.append(score)
    y = path_score

    print("Adding Constraint")
    # Create the 'prob' variable to contain the problem data
    prob = LpProblem("The Honeypot Allocation Problem", LpMinimize)

    x = LpVariable.dicts("path", path_matrix_dict.keys(), 0, 1, cat="Integer")

    x_rev = LpVariable.dicts("path_rev", path_matrix_dict.keys(), 0, 1, cat="Integer")

    xb = LpVariable.dicts("Node", graph_condensed.nodes(), 0, 1, cat="Integer")


    prob += (
        lpSum([x[p]* y[p] for p in path_matrix_dict.keys()]),
        "Sum_of_Transporting",)

    prob += (
        lpSum([xb[n] for n in graph_condensed.nodes()]) <= budget,
        "Budget Constraint",
    )
    for i in range(len(path_matrix)):
        prob += (
            lpSum(xb[n]*path_matrix[i][n] for n in graph_condensed.nodes()) - (1 - x[i])*(budget + 1) + 0.001 <= 0,
            ""
        )
    for i in range(len(path_matrix)):
        prob += (
            lpSum(xb[n]*path_matrix[i][n] for n in graph_condensed.nodes()) - (1 - x[i]) >= 0,
            ""
        )

    # for i in range(len(path_matrix)):
    #     prob += (
    #         x[i] + x_rev[i] == 1,
    #         ""
    #     )

    print("Solving LP")
    prob.writeLP("netflowsimple.lp")

    # The problem is solved using PuLP's choice of Solver
    prob.solve()
    # The status of the solution is printed to the screen
    print("Status:", LpStatus[prob.status])

    # Each of the variables is printed with it's resolved optimum value
    node_blocked = []
    for v in prob.variables():
        # print(v.name, "=", v.varValue)
        if "Node" in v.name:
            if v.varValue >= 0.5:
                node_blocked.append(int(v.name[5:]))
    
    true_score = evaluate_flow(graph_condensed, node_blocked)
    print(node_blocked)

    # The optimised objective function value is printed to the screen
    print("Score from the Solver Function = ", value(prob.objective))
    print("True Score: ", true_score)

    return true_score, node_blocked

def mip_flow_2(graph_condensed):
    path_all = graph_condensed.graph["path_all"]
    path_number_dict = graph_condensed.graph["path_number_dict"]
    starting_nodes = graph_condensed.graph["starting_nodes"]
    budget = graph_condensed.graph["budget"]
    DA = graph_condensed.graph["DA"]
    blockable_node = []
    for i in graph_condensed.nodes():
        if graph_condensed.nodes[i]["blockable"]:
            blockable_node.append(i)
    
    temp = []
    for n in blockable_node:
        if n not in starting_nodes:
            temp.append(n)
    blockable_node = temp
    
    # Linear Programming to find the clean path
    
    prob1 = LpProblem("Finding Total of Clean Path SubProblem", LpMaximize)
    
    theta = LpVariable.dicts("# of total path to DA", graph_condensed.nodes(), lowBound = 0, cat="Integer")
    
    _variable = theta[DA]
    _variable.setInitialValue(1)
    _variable.fixValue()
    
    prob1 += (
        lpSum([theta[n] for n in starting_nodes]),
        "Chance attacker get to DA node from given starting nodes",)

    for i in graph_condensed.nodes():
        
        if i == DA: continue
        inbound_node = []
        for u, v in graph_condensed.out_edges(i):
            inbound_node.append(v)
        
        # print(i, inbound_node)
        prob1 += (
                theta[i] - lpSum(theta[n] for n in inbound_node) == 0,
            ""
        )      
    
    prob1.writeLP("netflowsimple.lp")
    # The problem is solved using PuLP's choice of Solver
    prob1.solve()
    # The status of the solution is printed to the screen
    
    theta = dict()
    for v in prob1.variables():
        node = int(v.name.split("_")[-1])
        theta[node] = v.varValue
    
    
    prob2 = LpProblem("The Honeypot Allocation Problem", LpMinimize)

    alpha = LpVariable.dicts("# of clean path to DA", graph_condensed.nodes(), lowBound = 0, cat="Integer")

    z = LpVariable.dicts("temp variable", graph_condensed.nodes(), lowBound = 0, cat="Integer") 
    
    b = LpVariable.dicts("Budget for nodes", graph_condensed.nodes(), 0, 1, cat="Integer")



    # Theta and Alpha of DA node is a fixed value of 1
    
    
    _variable = alpha[DA]
    _variable.setInitialValue(1)
    _variable.fixValue()
    
    
    _variable = z[DA]
    _variable.setInitialValue(1)
    _variable.fixValue()
    
    prob2 += (
        lpSum([(z[n]) * (1/(int(theta[n])*len(starting_nodes))) for n in starting_nodes]),
        "Chance attacker get to DA node from given starting nodes",)

    prob2 += (
        lpSum([b[n] for n in graph_condensed.nodes()]) <= budget,
        "Budget Constraint",
    )
    
    for i in graph_condensed.nodes():
        

        if i == DA: continue
        inbound_node = []
        for u, v in graph_condensed.out_edges(i):
            inbound_node.append(v)
        
        prob2 += (
                alpha[i] - lpSum(z[n] for n in inbound_node) == 0,
            ""
        )
        
        

        if i in blockable_node:

            prob2 += (
                z[i] <= alpha[i],
                "",
            )
            prob2 += (
                z[i] <= theta[i]* (1-b[i]),
                "",
            )
            prob2 += (
                z[i] >= alpha[i] + theta[i]*(-b[i]),
                "",
            )
            prob2 += (
                z[i] >= 0,
                "",
            )
        else:
            prob2 += (
                z[i] == alpha[i],
                "",
            )

    prob2.writeLP("netflowsimple.lp")

    # The problem is solved using PuLP's choice of Solver
    prob2.solve()
    # The status of the solution is printed to the screen
    # print("Status:", LpStatus[prob2.status])
    # Each of the variables is printed with it's resolved optimum value
    node_blocked = []
    for v in prob2.variables():
        # print(v.name, "=", v.varValue)
        if "Budget" in v.name:
            if v.varValue >= 0.5:
                node_blocked.append(int(v.name.split("_")[-1]))
    
    true_score = evaluate_flow(graph_condensed, node_blocked)
    # print(node_blocked)
    # The optimised objective function value is printed to the screen
    print("Score from the Solver Function = ", value(prob2.objective))
    print("True Score: ", true_score)

    return true_score, node_blocked

def mip_dygraph(list_graph):
    graph_sample_number = len(list_graph)
    budget = list_graph[0].graph["budget"]

    mulgraph_strarting_nodes = []
    mulgraph_DA = []
    mulgraph_blockable_node = []
    # Linear Programming to find the clean path
    
    prob1 = LpProblem("Finding Total of Clean Path SubProblem", LpMaximize)
    theta = dict()
    
    #reindexing nodes number in graphs
    
    mip_to_origin_list = [] # convert from mip indexing to networkx graph (for running mip on multiple graph)
    origin_to_mip_list = []
    multi_graph_nodes = []
    node_union = set()
    index_from = 0
    for i in range(len(list_graph)):
        mip_to_origin = dict()
        origin_to_mip = dict()
        starting_nodes = []
        blockable_nodes = []

        for n in list_graph[i].nodes():
            node_union.add(n)
            mip_to_origin[n + index_from] = n 
            origin_to_mip[n] = n+index_from
            multi_graph_nodes.append(n + index_from)
            if n in list_graph[i].graph["starting_nodes"]:
                starting_nodes.append(n+index_from)
            if n == list_graph[i].graph["DA"]:
                mulgraph_DA.append(n+index_from)
            if list_graph[i].nodes[n]["blockable"]:
                blockable_nodes.append(n+index_from)
        
        origin_to_mip_list.append(origin_to_mip)
        mip_to_origin_list.append(mip_to_origin)
        mulgraph_strarting_nodes.append(starting_nodes)
        mulgraph_blockable_node.append(blockable_nodes)
        #print(list_graph[i].nodes())
        index_from = max(list_graph[i].nodes()) + index_from
    # print(graph_reindex_list[1])

    theta = LpVariable.dicts("Total path to DA", multi_graph_nodes, lowBound = 0, cat="Integer")
    
    
    for DA_node in mulgraph_DA:
        _variable = theta[DA_node]
        _variable.setInitialValue(1)
        _variable.fixValue()
    
    prob1 += (
        lpSum([theta[n] for starting_nodes in mulgraph_strarting_nodes for n in starting_nodes]),
        "Chance attacker get to DA node from given starting nodes",)

    for t in range(graph_sample_number):
        graph = list_graph[t]
        for n in graph.nodes(): 
            i =  origin_to_mip_list[t][n]
            if i in mulgraph_DA: continue
            inbound_node = []
            for u, v in graph.out_edges(n):
                inbound_node.append(origin_to_mip_list[t][v])
            
            # print(i, inbound_node)
            prob1 += (
                    theta[i] - lpSum(theta[n] for n in inbound_node) == 0,
                ""
            )      
    
    prob1.writeLP("netflowsimple.lp")
    # The problem is solved using PuLP's choice of Solver
    prob1.solve(GUROBI_CMD(msg=0))
    # The status of the solution is printed to the screen
    
    theta = dict()
    for v in prob1.variables():
        node = int(v.name.split("_")[-1])
        theta[node] = v.varValue
    
    
    prob2 = LpProblem("The Honeypot Allocation Problem", LpMinimize)


    alpha = LpVariable.dicts("# of clean path to DA", multi_graph_nodes, lowBound = 0, cat="Integer")

    z = LpVariable.dicts("temp variable", multi_graph_nodes, lowBound = 0, cat="Integer") 
    
    b = LpVariable.dicts("Budget for nodes", node_union, 0, 1, cat="Integer")

    beta = LpVariable.dicts("temp variable for b", multi_graph_nodes, 0, 1, cat="Integer") 


    # Theta and Alpha of DA node is a fixed value of 1
    
    for DA_node in mulgraph_DA:
        _variable = alpha[DA_node]
        _variable.setInitialValue(1)
        _variable.fixValue()
        _variable = z[DA_node]
        _variable.setInitialValue(1)
        _variable.fixValue()

    prob2 += (
        lpSum([(z[n]) * (1/(int(theta[n])*len(starting_nodes))) for starting_nodes in mulgraph_strarting_nodes for n in starting_nodes]),
        "Chance attacker get to DA node from given starting nodes",)

    
    prob2 += (
        lpSum([b[n] for n in node_union]) <= budget,
        "Budget Constraint",
    )
    
    for t in range(graph_sample_number):
        graph = list_graph[t]
        for n in graph.nodes():
            i =  origin_to_mip_list[t][n]
            if i in mulgraph_DA: continue         
            inbound_node = []
            for u, v in graph.out_edges(n):
                inbound_node.append(origin_to_mip_list[t][v])
            
            prob2 += (
                    alpha[i] - lpSum(z[n] for n in inbound_node) == 0,
                ""
            )
            
            prob2 += (
                    b[n] - beta[i] == 0,
                ""
            )
            
            if i in mulgraph_blockable_node[t]:
                prob2 += (
                    z[i] <= alpha[i],
                    "",
                )
                prob2 += (
                    z[i] <= theta[i]* (1-beta[i]),
                    "",
                )
                
                prob2 += (
                    z[i] >= alpha[i] + theta[i]*(-beta[i]),
                    "",
                )
                prob2 += (
                    z[i] >= 0,
                    "",
                )
            else:
                prob2 += (
                    z[i] == alpha[i],
                    "",
                )

    prob2.writeLP("netflowsimple.lp")

    # The problem is solved using PuLP's choice of Solver
    prob2.solve(GUROBI_CMD(msg=0)) 
    # The status of the solution is printed to the screen
    # print("Status:", LpStatus[prob2.status])
    # Each of the variables is printed with it's resolved optimum value
    node_blocked = []
    for v in prob2.variables():
        # print(v.name, "=", v.varValue)
        if "Budget" in v.name:
            if v.varValue >= 0.5:
                node_blocked.append(int(v.name.split("_")[-1]))
    
    score_list = []
    for i in range(graph_sample_number):
        score = evaluate_flow(list_graph[i], node_blocked)
        print("Score for Graph: ", i, ": ", score)
        score_list.append(score)
    average_score = sum(score_list) / len(score_list)

    # print(node_blocked)
    # The optimised objective function value is printed to the screen
    print("Score from the Solver Function = ", value(prob2.objective))
    print("Average Score: ", average_score)

    return average_score, node_blocked
    
def mip_dygraph_weight(list_graph, weight_list):

    graph_sample_number = len(list_graph)
    budget = list_graph[0].graph["budget"]
    mulgraph_strarting_nodes = []
    mulgraph_DA = []
    mulgraph_blockable_node = []
    # Linear Programming to find the clean path
    
    prob1 = LpProblem("Finding Total of Clean Path SubProblem", LpMaximize)
    theta = dict()
    
    #reindexing nodes number in graphs
    
    graph_reindex_list = [] # convert from mip indexing to networkx graph (for running mip on multiple graph)
    origin_to_mip_list = []
    multi_graph_nodes = []
    node_union = set()
    index_from = 0
    for i in range(len(list_graph)):
        reindex_dict = dict()
        starting_nodes = []
        blockable_nodes = []
        origin_to_mip = dict()
        for n in list_graph[i].nodes():
            node_union.add(n)
            reindex_dict[n + index_from] = n 
            origin_to_mip[n] = n+index_from
            multi_graph_nodes.append(n + index_from)
            if n in list_graph[i].graph["starting_nodes"]:
                starting_nodes.append(n+index_from)
            if n == list_graph[i].graph["DA"]:
                mulgraph_DA.append(n+index_from)
            if list_graph[i].nodes[n]["blockable"]:
                blockable_nodes.append(n+index_from)
        
        origin_to_mip_list.append(origin_to_mip)
        graph_reindex_list.append(reindex_dict)
        mulgraph_strarting_nodes.append(starting_nodes)
        mulgraph_blockable_node.append(blockable_nodes)
        index_from = max(list_graph[i].nodes()) + index_from
    # print(graph_reindex_list[1])

    theta = LpVariable.dicts("Total path to DA", multi_graph_nodes, lowBound = 0, cat="Integer")
    
    
    for DA_node in mulgraph_DA:
        _variable = theta[DA_node]
        _variable.setInitialValue(1)
        _variable.fixValue()
    
    prob1 += (
        lpSum([theta[n] for starting_nodes in mulgraph_strarting_nodes for n in starting_nodes]),
        "Chance attacker get to DA node from given starting nodes",)

    for t in range(graph_sample_number):
        graph = list_graph[t]
        for n in graph.nodes(): 
            i =  origin_to_mip_list[t][n]
            if i in mulgraph_DA: continue
            inbound_node = []
            for u, v in graph.out_edges(n):
                inbound_node.append(origin_to_mip_list[t][v])
            
            # print(i, inbound_node)
            prob1 += (
                    theta[i] - lpSum(theta[n] for n in inbound_node) == 0,
                ""
            )      
    
    prob1.writeLP("netflowsimple.lp")
    # The problem is solved using PuLP's choice of Solver
    prob1.solve(GUROBI_CMD(msg=0))
    # The status of the solution is printed to the screen
    
    theta = dict()
    for v in prob1.variables():
        node = int(v.name.split("_")[-1])
        theta[node] = v.varValue
    
    
    prob2 = LpProblem("The Honeypot Allocation Problem", LpMinimize)


    alpha = LpVariable.dicts("# of clean path to DA", multi_graph_nodes, lowBound = 0, cat="Integer")

    z = LpVariable.dicts("temp variable", multi_graph_nodes, lowBound = 0, cat="Integer") 
    
    b = LpVariable.dicts("Budget for nodes", node_union, 0, 1, cat="Integer")

    beta = LpVariable.dicts("temp variable for b", multi_graph_nodes, 0, 1, cat="Integer") 


    # Theta and Alpha of DA node is a fixed value of 1
    
    for DA_node in mulgraph_DA:
        _variable = alpha[DA_node]
        _variable.setInitialValue(1)
        _variable.fixValue()
        _variable = z[DA_node]
        _variable.setInitialValue(1)
        _variable.fixValue()
        
    

    # print(theta)
    # print(starting_nodes)
    # print(len(theta))
    
    # dsadsa
    #print(theta)
    prob2 += (
        lpSum([weight_list[i]*(z[n]) * (1/(int(theta[n])*len(mulgraph_strarting_nodes[i]))) for i in range(len(mulgraph_strarting_nodes)) for n in mulgraph_strarting_nodes[i]]),
        "Chance attacker get to DA node from given starting nodes",)

    
    prob2 += (
        lpSum([b[n] for n in node_union]) <= budget,
        "Budget Constraint",
    )
    
    for t in range(graph_sample_number):
        graph = list_graph[t]
        for n in graph.nodes():
            i =  origin_to_mip_list[t][n]
            if i in mulgraph_DA: continue         
            inbound_node = []
            for u, v in graph.out_edges(n):
                inbound_node.append(origin_to_mip_list[t][v])
            
            prob2 += (
                    alpha[i] - lpSum(z[n] for n in inbound_node) == 0,
                ""
            )
            
            prob2 += (
                    b[n] - beta[i] == 0,
                ""
            )
            
            if i in mulgraph_blockable_node[t]:
                prob2 += (
                    z[i] <= alpha[i],
                    "",
                )
                prob2 += (
                    z[i] <= theta[i]* (1-beta[i]),
                    "",
                )
                
                prob2 += (
                    z[i] >= alpha[i] + theta[i]*(-beta[i]),
                    "",
                )
                prob2 += (
                    z[i] >= 0,
                    "",
                )
            else:
                prob2 += (
                    z[i] == alpha[i],
                    "",
                )

    prob2.writeLP("netflowsimple.lp")

    # The problem is solved using PuLP's choice of Solver
    prob2.solve(GUROBI_CMD(msg=0)) 
    # The status of the solution is printed to the screen
    # print("Status:", LpStatus[prob2.status])
    # Each of the variables is printed with it's resolved optimum value
    node_blocked = []
    for v in prob2.variables():
        # print(v.name, "=", v.varValue)
        if "Budget" in v.name:
            if v.varValue >= 0.5:
                node_blocked.append(int(v.name.split("_")[-1]))
    
    score_list = []
    for i in range(graph_sample_number):
        score = evaluate_flow(list_graph[i], node_blocked)
        print("Score for Graph: ", i, ": ", score)
        score_list.append(score)
    average_score = sum(score_list) / len(score_list)

    # print(node_blocked)
    # The optimised objective function value is printed to the screen
    print("Score from the Solver Function = ", value(prob2.objective))
    print("Average Score: ", average_score)

    return average_score, node_blocked 

def mip_flow_competent(graph_original):
    starting_nodes = graph_original.graph["starting_nodes"]
    print(len(starting_nodes))
    budget = graph_original.graph["budget"]
    print(budget)
    DA = graph_original.graph["DA"]
    blockable_node = []
    for i in graph_original.nodes():
        if graph_original.nodes[i]["blockable"]:
            if i not in starting_nodes:
                blockable_node.append(i)

    # Linear Programming to find the clean path
    
    prob1 = LpProblem("Competent attacker problem", LpMinimize)

    R = LpVariable.dicts("DA Reachability variable", graph_original.nodes(), 0, 1, cat="Integer")
    
    b = LpVariable.dicts("Budget for nodes", graph_original.nodes(), 0, 1, cat="Integer")

    
    _variable = R[DA]
    _variable.setInitialValue(1)
    _variable.fixValue()

    prob1 += (
        lpSum([R[n] for n in starting_nodes]),
        "Chance attacker get to DA node from given starting nodes",)
        
    prob1 += (
        lpSum([b[n] for n in graph_original.nodes()]) <= budget,
        "Budget Constraint",
    )

    for u, v in graph_original.edges():

        if u not in blockable_node:                 
            prob1 += (
                    R[u] >= R[v],
                    "",)
        else:
            prob1 += (
                    R[u] >= R[v] - b[u],
                    "",)
            
    
    prob1.writeLP("netflowsimple.lp")
    # The problem is solved using PuLP's choice of Solver
    prob1.solve(GUROBI_CMD(msg=0))
    # prob1.solve()
    # The status of the solution is printed to the screen
    
    theta = dict()
    for v in prob1.variables():
        node = int(v.name.split("_")[-1])
        theta[node] = v.varValue
    # The status of the solution is printed to the screen
    # print("Status:", LpStatus[prob2.status])
    # Each of the variables is printed with it's resolved optimum value
    node_blocked = []

    for v in prob1.variables():
        # print(v.name, "=", v.varValue)
        if "Budget" in v.name:
            if v.varValue >= 0.5:
                node_blocked.append(int(v.name.split("_")[-1]))

    true_score = evaluate_flow_competent(graph_original, node_blocked)
    print(node_blocked)
    # The optimised objective function value is printed to the screen
    print("Score from the Solver Function = ", value(prob1.objective))
    print("True Score: ", true_score)

    return true_score, node_blocked

def mip_flow_phiattack(graph_original, phi=1):
    #path_all = graph_condensed.graph["path_all"]
    #path_number_dict = graph_condensed.graph["path_number_dict"]
    starting_nodes = graph_original.graph["starting_nodes"]
    budget = graph_original.graph["budget"]
    DA = graph_original.graph["DA"]
    blockable_node = []
    for i in graph_original.nodes():
        if graph_original.nodes[i]["blockable"]:
            blockable_node.append(i)
    temp = []
    for n in blockable_node:
        if n not in starting_nodes:
            temp.append(n)
    blockable_node = temp

    print(len(blockable_node))
    # Linear Programming to find the clean path
    
    prob1 = LpProblem("Finding Total of Clean Path SubProblem", LpMaximize)
    
    theta = LpVariable.dicts("# of total path to DA", graph_original.graph["shortest_nodes"], lowBound = 0, cat="Integer")
    
    _variable = theta[DA]
    _variable.setInitialValue(1)
    _variable.fixValue()

    prob1 += (
        lpSum([theta[n] for n in starting_nodes]),
        "Chance attacker get to DA node from given starting nodes",)

    for i in graph_original.graph["shortest_nodes"]:
        
        if i == DA: continue
        inbound_node = []
        for u, v in graph_original.out_edges(i):
            if graph_original[u][v]["shortest_edges"] == True:
                inbound_node.append(v)
        
        # print(i, inbound_node)
        prob1 += (
                theta[i] - lpSum(theta[n] for n in inbound_node) <= 0,
            ""
        )      
    
    
    prob1.writeLP("netflowsimple_1.lp")

    # The problem is solved using PuLP's choice of Solver
    # prob1.solve(CPLEX_CMD())
    prob1.solve(PULP_CBC_CMD(msg=0))
    # The status of the solution is printed to the screen
    
    theta = dict()
    for v in prob1.variables():
        node = int(v.name.split("_")[-1])
        # print(v.name, "=", v.varValue)
        theta[node] = v.varValue
    # print(theta)

        
    prob2 = LpProblem("Mixed Attacker Problem", LpMinimize)

    alpha = LpVariable.dicts("# of clean path to DA", graph_original.graph["shortest_nodes"], lowBound = 0, cat="Integer")

    z = LpVariable.dicts("temp variable", graph_original.graph["shortest_nodes"], lowBound = 0, cat="Integer") 
    
    b = LpVariable.dicts("Budget for nodes", graph_original.nodes(), 0, 1, cat="Integer")
    
    R = LpVariable.dicts("DA Reachability variable", graph_original.nodes(), 0, 1, cat="Integer")


    # Theta and Alpha of DA node is a fixed value of 1
    
    
    _variable = alpha[DA]
    _variable.setInitialValue(1)
    _variable.fixValue()
    
    _variable = z[DA]
    _variable.setInitialValue(1)
    _variable.fixValue()

    _variable = R[DA]
    _variable.setInitialValue(1)
    _variable.fixValue()
    
    prob2 += (
        lpSum([(1-phi)*(z[n]) * (1/(int(theta[n])*len(starting_nodes))) + phi*R[n] for n in starting_nodes]),
        "Mixed attacker problem",)
    # prob2 += (
    #     lpSum([phi*R[n] for n in starting_nodes]),
    #     "Chance attacker get to DA node from given starting nodes",)




    prob2 += (
        lpSum([b[n] for n in graph_original.nodes()]) <= budget,
        "Budget Constraint",
    )
    
    for i in graph_original.graph["shortest_nodes"]:
        

        if i == DA: continue
        inbound_node = []
        for u, v in graph_original.out_edges(i):
            if graph_original[u][v]["shortest_edges"] == True:
                inbound_node.append(v)
        
        prob2 += (
                alpha[i] - lpSum(z[n] for n in inbound_node) == 0,
            ""
        )
        
        

        if i in blockable_node:

            prob2 += (
                z[i] <= alpha[i],
                "",
            )
            prob2 += (
                z[i] <= theta[i]* (1-b[i]),
                "",
            )
            prob2 += (
                z[i] >= alpha[i] + theta[i]*(-b[i]),
                "",
            )
            prob2 += (
                z[i] >= 0,
                "",
            )
        else:
            prob2 += (
                z[i] == alpha[i],
                "",
            )

    for u, v in graph_original.edges():

        if u not in blockable_node:                 
            prob2 += (
                    R[u] >= R[v],
                    "",)
        else:
            prob2 += (
                    R[u] >= R[v] - b[u],
                    "",)

    prob2.writeLP("netflowsimple.lp")

    # The problem is solved using PuLP's choice of Solver
    prob2.solve(GUROBI_CMD(msg=0))
    # The status of the solution is printed to the screen
    print("Status:", LpStatus[prob2.status])

    # Each of the variables is printed with it's resolved optimum value
    node_blocked = []
    for v in prob2.variables():
        # print(v.name, "=", v.varValue)
        # print(v.varValue)
        if "Budget" in v.name:
            if v.varValue >= 0.5:
                node_blocked.append(int(v.name.split("_")[-1]))
    

    # print(node_blocked)
    # The optimised objective function value is printed to the screen
    print("Score from the Solver Function = ", value(prob2.objective))
    # print("True Score: ", true_score)

    return node_blocked

def mip_dygraph_mixed_attack(list_graph, phi=1):
    graph_sample_number = len(list_graph)
    budget = list_graph[0].graph["budget"]

    mulgraph_strarting_nodes = []
    mulgraph_DA = []
    mulgraph_blockable_node = []
    # Linear Programming to find the clean path
    
    prob1 = LpProblem("Finding Total of Clean Path SubProblem", LpMaximize)
    theta = dict()
    
    #reindexing nodes number in graphs
    list_graph_relabel = []
    mip_to_origin_list = []
    origin_to_mip_list = []
    multi_graph_nodes = []
    node_union = set()
    index_from = 0
    shortest_nodes_list = []
    for i in range(len(list_graph)):
        mip_to_origin = dict()
        starting_nodes = []
        blockable_nodes = []
        origin_to_mip = dict()
        shortest_nodes = []
        for n in list_graph[i].nodes():
            node_union.add(n)
            mip_to_origin[n + index_from] = n 
            origin_to_mip[n] = n+index_from
            multi_graph_nodes.append(n + index_from)
            if n in list_graph[i].graph["starting_nodes"]:
                starting_nodes.append(n+index_from)
            if n == list_graph[i].graph["DA"]:
                mulgraph_DA.append(n+index_from)
            if list_graph[i].nodes[n]["blockable"]:
                blockable_nodes.append(n+index_from)
            if n in list_graph[i].graph["shortest_nodes"]:
                shortest_nodes_list.append(n+index_from)
                shortest_nodes.append(n+index_from)

        
        graph_relabel = nx.relabel_nodes(list_graph[i], origin_to_mip, copy=True)
        graph_relabel.graph["shortest_nodes"] = shortest_nodes
        graph_relabel.graph["starting_nodes"] = starting_nodes
        list_graph_relabel.append(graph_relabel)



        # shortest_nodes_list.append(shortest_nodes)
        origin_to_mip_list.append(origin_to_mip)
        mip_to_origin_list.append(mip_to_origin)
        mulgraph_strarting_nodes.append(starting_nodes)
        mulgraph_blockable_node.append(blockable_nodes)
        index_from = max(list_graph[i].nodes()) + index_from
    




    theta = LpVariable.dicts("Total path to DA", shortest_nodes_list, lowBound = 0, cat="Integer")
    
    
    for DA_node in mulgraph_DA:
        _variable = theta[DA_node]
        _variable.setInitialValue(1)
        _variable.fixValue()
    
    prob1 += (
        lpSum([theta[n] for starting_nodes in mulgraph_strarting_nodes for n in starting_nodes]),
        "Chance attacker get to DA node from given starting nodes",)

    for t in range(graph_sample_number):
        graph = list_graph_relabel[t]
        for i in graph.graph["shortest_nodes"]: 
            # i =  origin_to_mip_list[t][n]
            if i in mulgraph_DA: continue
            inbound_node = []
            for u, v in graph.out_edges(i):
                if graph[u][v]["shortest_edges"] == True:
                    inbound_node.append(v)
            
            # print(t, inbound_node)
            prob1 += (
                    theta[i] - lpSum(theta[n] for n in inbound_node) == 0,
                ""
            )      
    
    prob1.writeLP("netflowsimple.lp")
    # print(GUROBI_CMD.getOption())
    # dadas
    # The problem is solved using PuLP's choice of Solver
    prob1.solve(GUROBI_CMD(msg=0))
    # The status of the solution is printed to the screen
    
    theta = dict()
    for v in prob1.variables():
        # print(v.name, "=", v.varValue)
        node = int(v.name.split("_")[-1])
        theta[node] = v.varValue
    

    
    
    prob2 = LpProblem("The Honeypot Allocation Problem", LpMinimize)


    alpha = LpVariable.dicts("# of clean path to DA", shortest_nodes_list, lowBound = 0, cat="Integer")

    z = LpVariable.dicts("temp variable", shortest_nodes_list, lowBound = 0, cat="Integer") 
    
    b = LpVariable.dicts("Budget for nodes", node_union, 0, 1, cat="Integer")

    beta = LpVariable.dicts("temp variable for b", multi_graph_nodes, 0, 1, cat="Integer") 

    R = LpVariable.dicts("DA Reachability variable", multi_graph_nodes, 0, 1, cat="Integer")


    # Theta and Alpha of DA node is a fixed value of 1
    
    for DA_node in mulgraph_DA:
        _variable = alpha[DA_node]
        _variable.setInitialValue(1)
        _variable.fixValue()
        _variable = z[DA_node]
        _variable.setInitialValue(1)
        _variable.fixValue()
        _variable = R[DA_node]
        _variable.setInitialValue(1)
        _variable.fixValue()

    prob2 += (
        lpSum([(1 - phi) * (z[n]) * (1/(int(theta[n]))) + phi*R[n]  for starting_nodes in mulgraph_strarting_nodes for n in starting_nodes]),
        "Chance attacker get to DA node from given starting nodes",)

    
    prob2 += (
        lpSum([b[n] for n in node_union]) <= budget,
        "Budget Constraint",
    )
    
    for t in range(graph_sample_number):
        graph = list_graph_relabel[t]
        for i in graph.graph["shortest_nodes"]:
            if i in mulgraph_DA: continue         
            inbound_node = []
            for u, v in graph.out_edges(i):
                if graph[u][v]["shortest_edges"] == True:
                    inbound_node.append(v)
            
            prob2 += (
                    alpha[i] - lpSum(z[n] for n in inbound_node) == 0,
                ""
            )
            
            prob2 += (
                    b[mip_to_origin_list[t][i]] - beta[i] == 0,
                ""
            )
            
            if i in mulgraph_blockable_node[t]:
                prob2 += (
                    z[i] <= alpha[i],
                    "",
                )
                prob2 += (
                    z[i] <= theta[i]* (1-beta[i]),
                    "",
                )
                
                prob2 += (
                    z[i] >= alpha[i] + theta[i]*(-beta[i]),
                    "",
                )
                prob2 += (
                    z[i] >= 0,
                    "",
                )
            else:
                prob2 += (
                    z[i] == alpha[i],
                    "",
                )

        for u, v in graph.edges():

            if u not in mulgraph_blockable_node[t]:                 
                prob2 += (
                        R[u] >= R[v],
                        "",)
            else:
                prob2 += (
                        R[u] >= R[v] - beta[u],
                        "",)

    prob2.writeLP("netflowsimple.lp")

    # The problem is solved using PuLP's choice of Solver
    prob2.solve(GUROBI_CMD(msg=0) )
    # The status of the solution is printed to the screen
    # print("Status:", LpStatus[prob2.status])
    # Each of the variables is printed with it's resolved optimum value
    node_blocked = []
    for v in prob2.variables():
        # print(v.name, "=", v.varValue)
        if "Budget" in v.name:
            if v.varValue >= 0.5:
                node_blocked.append(int(v.name.split("_")[-1]))
    
    print("Score from the Solver Function = ", value(prob2.objective))

    return node_blocked