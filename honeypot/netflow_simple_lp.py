from importlib.resources import path
from pulp import *
from re import L
from xxlimited import new
import networkx as nx
from itertools import combinations
from patch import topological_generations
from utility import is_start, is_blockable, report, is_node_blockable, evaluate_flow, evaluate_flow_2, evaluate_flow_competent
from functools import lru_cache
from setupgraph import random_setup, flow_graph_setup, get_spath_graph
from networkx.readwrite.gpickle import write_gpickle, read_gpickle
import networkx as nx
from argparse import ArgumentParser


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
#     "blockable_p": 0.7,   # blockable_p = 1 means that all are blockable
#     "double_edge_p": 0, # if double_edge = 0, no double edge appear
#     # "cross_path_p": 0,
#     "multi_block_p": 1,
# }




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
    
    print(len(blockable_node))
    
    
    
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
    prob1.solve(PULP_CBC_CMD(msg=0))
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
    print(type(DA))
    
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
    prob2.solve(GUROBI_CMD(msg=0))
    # The status of the solution is printed to the screen
    print("Status:", LpStatus[prob2.status])

    # Each of the variables is printed with it's resolved optimum value
    node_blocked = []
    for v in prob2.variables():
        # print(v.name, "=", v.varValue)
        if "Budget" in v.name:
            if v.varValue >= 0.5:
                node_blocked.append(int(v.name.split("_")[-1]))
    


    return node_blocked

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



    return node_blocked



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
        lpSum([(1-phi)*(z[n]) * (1/(int(theta[n]))) + phi*R[n] for n in starting_nodes]),
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
    
    # flat_score = evaluate_flow(graph_original, node_blocked)
    # competent_score = evaluate_flow_competent(graph_original, node_blocked)
    # true_score = (flat_score+competent_score)/2
    # print("Flat Score: ", flat_score)
    # print("Competent Score: ",competent_score)
    # # print(node_blocked)
    # # The optimised objective function value is printed to the screen
    # print("Score from the Solver Function = ", value(prob2.objective))
    # print("True Score: ", true_score)

    return node_blocked
    
def mip_flow_phiattack_wm(graph_original, weight=0.5, p = 1):
    #path_all = graph_condensed.graph["path_all"]
    #path_number_dict = graph_condensed.graph["path_number_dict"]
    strat_flat = mip_flow_phiattack(graph_original, 0)
    opt_flat = evaluate_flow(graph_original, strat_flat)
    strat_competent = mip_flow_phiattack(graph_original, 1)
    opt_competent = evaluate_flow_competent(graph_original, strat_competent)

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
    

    # prob2 += (
    #     lpSum([phi*R[n] for n in starting_nodes]),
    #     "Chance attacker get to DA node from given starting nodes",)


    obj_flat = lpSum([(z[n]) * (1/(int(theta[n]))) for n in starting_nodes])
    obj_competent = lpSum([R[n] for n in starting_nodes])
    obj = weight*(obj_flat-(opt_flat-0.0000001)) + (1-weight)*((obj_competent-(opt_competent-0.000001)))
    prob2 += (obj,
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
    
    # flat_score = evaluate_flow(graph_original, node_blocked)
    # competent_score = evaluate_flow_competent(graph_original, node_blocked)
    # true_score = (flat_score+competent_score)/2
    # print("Flat Score: ", flat_score)
    # print("Competent Score: ",competent_score)
    # # print(node_blocked)
    # # The optimised objective function value is printed to the screen
    # print("Score from the Solver Function = ", value(prob2.objective))
    # print("True Score: ", true_score)

    return node_blocked
