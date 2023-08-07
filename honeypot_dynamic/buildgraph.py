import json
import io
import networkx as nx
from networkx.algorithms.cycles import simple_cycles
from itertools import islice
from networkx.readwrite.gpickle import write_gpickle
from utility import remove_dead_nodes, display, report
from collections import Counter

# graphs = [

#     (
#         "r500",
#         "/Users/huyngo/Desktop/Research/honeypot_dynamic/data/AAAI_graph/records500.json",
#         [48, 55, 54] + [1182, 1333, 1431, 1507],
#         [530],
#     ),

#     (
#         "r2000",
#         "/Users/huyngo/Desktop/Research/honeypot_dynamic/data/AAAI_graph/records2000.json",
#         [
#             14166,
#             14172,
#             14173,
#         ]
#         + [18729, 19276, 19374, 19892],
#         [17255, 17138, 14501, 17761],
#     ),
#         (
#         "r4000",
#         "/Users/huyngo/Desktop/Research/honeypot_dynamic/data/AAAI_graph/records4000.json",
#         [
#             8407,
#             8414,
#             8413,
#         ]
#         + [18986, 19698, 19708, 20080],
#         []
#     ),
# ]


graphs = [

    #     (
    #     "adsim025",
    #     "/Users/huyngo/Desktop/Research/honeypot_dynamic/data/adsimulator_graph/graph_025.json"
    # ),
    #         (
    #     "adsim05",
    #     "/Users/huyngo/Desktop/Research/honeypot_dynamic/data/adsimulator_graph/graph_05.json"
    # ),
                # (
    #     "adsimcompadapt",
    #     "/Users/huyngo/Desktop/Research/honeypot_dynamic/data/adsimulator_graph/adsimcompadapt.json"
    # ),
    (        "adsimcompadapt_1",
        "/Users/huyngo/Desktop/Research/honeypot_dynamic/data/adsimulator_graph/adsimcompadapt_1.json"
    ),
    # (        "adsimcompadapt_2",
    #     "/Users/huyngo/Desktop/Research/honeypot_dynamic/data/adsimulator_graph/adsimcompadapt_2.json"
    # ),
    # ( "adsimcompadapt_3",
    #     "/Users/huyngo/Desktop/Research/honeypot_dynamic/data/adsimulator_graph/adsimcompadapt_3.json"
    # ),
    # (        "adsimx05_1",
    #     "/Users/huyngo/Desktop/Research/honeypot_dynamic/data/adsimulator_graph/adsimx05_1.json"
    # ),
    #         (
    #     "adsimx05",
    #     "/Users/huyngo/Desktop/Research/honeypot_dynamic/data/adsimulator_graph/adsimx05.json"
    # ),
    #             (
    #     "adsimx10",
    #     "/Users/huyngo/Desktop/Research/honeypot_dynamic/data/adsimulator_graph/adsimx10.json"
    # ),
]

# graphs = [

#     (
#         "n1528e5864",
#         "/home/quanghuyngo/Desktop/CFR/Code/honeypot/data/adsimulator_graph/n1528e5864.json"
#     ),
#     (
#         "n2238e13156",
#         "/home/quanghuyngo/Desktop/CFR/Code/honeypot/data/adsimulator_graph/n2238e13156.json"
#     ),
# ]    

    # (
    #     "r2000",
    #     "/home/andrewngo/Desktop/Research-Express/routes/AAAI/records2000.json",
    #     [
    #         14166,
    #         14172,
    #         14173,
    #     ]
    #     + [18729, 19276, 19374, 19892],
    #     [17255, 17138, 14501, 17761],
    # ),
# ]


def buildgraph_DB_creator(graph):
    for name, json_location, DA_set, delete_snowball in graphs:
        DA = DA_set[0]
        DG = nx.DiGraph(DA=DA)
        edges = json.load(io.open(json_location, "r", encoding="utf-8-sig"))

        access = []
        for e in edges:
            access_type = e["p"]["segments"][0]["relationship"]["type"]
            # if access_type in [
            #     "AdminTo",
            #     "MemberOf",
            #     "HasSession",
            # ]: 

            access.append(access_type)
            start = e["p"]["start"]["identity"]
            end = e["p"]["end"]["identity"]
            source_label = e["p"]["start"]["labels"][-1]
            destination_label = e["p"]["end"]["labels"][-1]
            DG.add_node(start)
            DG.add_node(end)
            DG.nodes[start]["label"] = source_label
            DG.nodes[end]["label"] = destination_label
            if start in DA_set:
                start = DA
            if end in DA_set:
                end = DA
            if start == end:
                continue
            if start != DA and not DG.has_edge(start, end):
                DG.add_edge(start, end)
                DG.edges[start, end]["label"] = access_type

                    
            
        print(set(access))   
        print("Raw graph")
        report(DG, True)
        print("dasdassda")
        print("Number of nodes: ", len(DG.nodes()))
        print("Number of edges", len(DG.edges()))
        # remove_dead_nodes(DG)

        print(name)
        print("delete_snowball", len(delete_snowball))
        print(
            "delete_snowball percentage",
            len(delete_snowball) / DG.number_of_nodes(),
        )
        
        CG = DG.copy()
        # CG.remove_nodes_from(delete_snowball)
        # remove_dead_nodes(CG)
        # node_in_edge = set(node_in_edge)
        node_label = dict()
        user_node = 0
        comp_node = 0
        for n in DG.nodes():  
            label = DG.nodes[n]["label"]
            node_label[n] = label
            if label == "User":
                user_node += 1
            elif label == "Computer":
                comp_node += 1
        print("Number of User: ", user_node)
        print("Number of Comp: ", comp_node)
        number_of_DA_user = 0
        da_user = []
        for i in DA_set:
            print(node_label[i])
            if node_label[i] == "User":
                number_of_DA_user += 1
                da_user.append(i)
        print(number_of_DA_user)


        print(len(da_user))
        DG.graph["DA_user"] = DA_set
        DG.graph["DA_set"] = DA_set


        all_cycles = list(islice(simple_cycles(CG), 10000))
        # print(all_cycles)
        cycle_counter = Counter()
        for c in all_cycles:
            cycle_counter.update(c)
        # for p in all_cycles[:100]:
        #     print(p)
        # for v, ccount in cycle_counter.most_common(100):
        #     print(v, ccount, CG.out_degree(v))

        # asddsasd
        # assert len(all_cycles) == 0
        


        print("DG", name)
        # report(DG)
        # display(DG, delete_snowball)
        # print("CG", name)
        # report(CG)
        # display(CG)

        write_gpickle(DG, f"//Users/huyngo/Desktop/Research/honeypot_dynamic/processed_graph/{name}_all.gpickle")



def read_json_neo4j_3(graphs):
    for name, input_path in graphs:

        edges = []
        nodes = []
        for line in open(input_path, 'r'):
            json_line = json.loads(line)
            if json_line["type"] == "node":
                nodes.append(json_line)
            elif json_line["type"] == "relationship":
                edges.append(json_line)

        # print(edges)
        data = dict()
        data["nodes"] = nodes
        data["edges"] = edges
        nodes = list(data["nodes"])
    
        # print(nodes)
        # dasda
        DA_set = []
        #Find domain admin nodeids and definde DA set
        
        
        # labels = set()
        # for i in range(len(nodes)):
        #     if tuple(nodes[i]["labels"]) not in labels:
        #         labels.add(tuple(nodes[i]["labels"]))
        for i in range(len(nodes)):
            if nodes[i] == None:
                continue
            if "admincount" in nodes[i]["properties"]:
                if nodes[i]["properties"]["admincount"] == True:
                    DA_set.append(int(nodes[i]["id"]))
            # if nodes[i]["labels"] == ["Base", "Group"]:
            #     # "DOMAIN" in nodes[i]["properties"]["name"] or "ADMIN" in nodes[i]["properties"]["name"]:
            #     if "ADMIN" in nodes[i]["properties"]["name"]:
            #         DA_set.append(int(nodes[i]["id"]))
            # elif nodes[i]["labels"] == ["Base","Domain"]:
            #     DA_set.append(int(nodes[i]["id"]))
            # if "User" in nodes[i]["labels"]:
            #     if "ADMIN" in nodes[i]["properties"]["name"]:
            #         DA_set.append(int(nodes[i]["id"]))
        DA = DA_set[0]
        DG = nx.DiGraph(DA=DA)
        print("DOMAIN ADMINS set: ", DA_set)
        edges = list(data["edges"])
        
        node_in_edge = []
        access = []
        dict_edge = dict()
        dict_edge["AdminTo"] = 0
        dict_edge["MemberOf"] = 0
        dict_edge["HasSession"] = 0 
        for e in edges:
            access_type = e["label"]
            access.append(access_type)
            # if access_type in [
            #     "AdminTo",
            #     "MemberOf",
            #     "HasSession",
            # ]:
            start = int(e["start"]["id"])
            end = int(e["end"]["id"])

            if start in DA_set:
                start = DA
            if end in DA_set:
                end = DA
            if start == end:
                continue
            
            # if start not in DA_set and not DG.has_edge(start, end):
            if start != DA and not DG.has_edge(start, end):
                node_in_edge.append(start)
                node_in_edge.append(end)
                DG.add_edge(start, end)
                DG.edges[start, end]["label"] = access_type
            # if end == DA:
                #     dict_edge[access_type] = dict_edge[access_type] + 1
        # print(dict_edge)
        print(nodes[0])
        print(DA)
        print("dasdassda")
        print("Number of nodes: ", len(DG.nodes()))
        print("Number of edges", len(DG.edges()))
        
        # print(set(access))
        node_in_edge = set(node_in_edge)
        node_label = dict()
        user_node = 0
        comp_node = 0
        for n in nodes:
            id = int(n["id"])      
            label = n["labels"][-1]
            node_label[id] = label
            if label == "User":
                user_node += 1
            elif label == "Computer":
                comp_node += 1
        print("Number of User:", user_node)
        print("NUmber of Computer: ", comp_node)
        number_of_DA_user = 0
        da_user = []
        for i in DA_set:
            # print(node_label[i])
            if node_label[i] == "User":
                number_of_DA_user += 1
                da_user.append(i)
        # print(number_of_DA_user)
        for n in node_in_edge:
            DG.nodes[n]["label"] = node_label[n]
        print(len(da_user))
        DG.graph["DA_user"] = da_user
        DG.graph["DA_set"] = DA_set
        
        
        print(len(DG.edges()))
        print(len(DG.nodes()))
        # print(DG.edges())
        print("Raw graph")
        # count = 0
        # for u, v in DG.edges():
        #     if DG.edges[u, v]["label"] == "HasSession":
        #         count += 1
        # print(count)
        # remove_dead_nodes(DG)
        # count = 0
        # for u, v in DG.edges():
        #     if DG.edges[u, v]["label"] == "HasSession":
        #         count += 1
        # print(count)
        
        
        
        
        
        # dadsasdda

        CG = DG.copy()

        # all_cycles = list(islice(simple_cycles(CG), 10000))
        # # print(all_cycles)
        # cycle_counter = Counter()
        # for c in all_cycles:
        #     cycle_counter.update(c)
        # for p in all_cycles[:100]:
        #     print(p)
        # for v, ccount in cycle_counter.most_common(100):
        #     print(v, ccount, CG.out_degree(v))

        print("DG", name)
        # report(DG)
        write_gpickle(DG, f"//Users/huyngo/Desktop/Research/honeypot_dynamic/processed_graph/{name}_all.gpickle")



if __name__ == "__main__":
    data = read_json_neo4j_3(graphs)
    
    # data = buildgraph_DB_creator(graphs)
    print(data)