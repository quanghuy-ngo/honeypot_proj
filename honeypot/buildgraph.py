import json
import io
import networkx as nx
from networkx.algorithms.cycles import simple_cycles
from itertools import islice
from networkx.readwrite.gpickle import write_gpickle
from utility import remove_dead_nodes, display, report
from collections import Counter

graphs = [

    (
        "r500",
        "/Users/huyngo/Desktop/Research/honeypot/data/AAAI_graph/records500.json",
        [48, 55, 54] + [1182, 1333, 1431, 1507],
        [530],
    ),

    (
        "r2000",
        "/Users/huyngo/Desktop/Research/honeypot/data/AAAI_graph/records2000.json",
        [
            14166,
            14172,
            14173,
        ]
        + [18729, 19276, 19374, 19892],
        [17255, 17138, 14501, 17761],
    ),
        (
        "r4000",
        "/Users/huyngo/Desktop/Research/honeypot/data/AAAI_graph/records4000.json",
        [
            8407,
            8414,
            8413,
        ]
        + [18986, 19698, 19708, 20080],
        []
    ),
]
# graphs = [
#     (
#         "adsimx10",
#         "/Users/huyngo/Desktop/Research/honeypot/data/adsimulator_graph/adsimx10.json"
#     ),
#     (
#         "adsimx05",
#         "/Users/huyngo/Desktop/Research/honeypot/data/adsimulator_graph/adsimx05.json"
#     ),
# ]

# graphs = [
#     # (
#     #     "adsim025",
#     #     "/Users/huyngo/Desktop/Research/honeypot/data/adsimulator_graph/graph_025.json"
#     # ),
#     # (
#     #     "adsim05",
#     #     "/Users/huyngo/Desktop/Research/honeypot/data/adsimulator_graph/graph_05.json"
#     # ),
#     (
#         "adsim10",
#         "/Users/huyngo/Desktop/Research/honeypot/data/adsimulator_graph/graph_10.json"
#     ),
#     # (
#     #     "adsim25",
#     #     "/Users/huyngo/Desktop/Research/honeypot/data/adsimulator_graph/graph_25.json"
#     # ),
#     # (
#     #     "adsim50",
#     #     "/Users/huyngo/Desktop/Research/honeypot/data/adsimulator_graph/graph_50.json"
#     # ),
#     (
#         "adsim100",
#         "/Users/huyngo/Desktop/Research/honeypot/data/adsimulator_graph/graph_100.json"
#     ),
# ]
 
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
            if access_type in [
                "AdminTo",
                "MemberOf",
                "HasSession",
            ]: 
                access.append(access_type)
                start = e["p"]["start"]["identity"]
                end = e["p"]["end"]["identity"]
                if start in DA_set:
                    start = DA
                if end in DA_set:
                    end = DA
                if start == end:
                    continue
                if start != DA and not DG.has_edge(start, end):
                    DG.add_edge(start, end)
                    DG.edges[start, end]["label"] = access_type
                    source_label = e["p"]["start"]["labels"][-1]
                    destination_label = e["p"]["end"]["labels"][-1]
                    DG.nodes[start]["label"] = source_label
                    DG.nodes[end]["label"] = destination_label
                
        user_nodes = []
        computer_nodes = []
        for n in DG.nodes():
            if DG.nodes[n]["label"] == "User":
                user_nodes.append(n)
            elif DG.nodes[n]["label"] == "Computer":
                computer_nodes.append(n)


        print("Number of Edges: ", len(DG.edges()))
        print("Number of Nodes: ",len(DG.nodes()))
        print("Niumber of Computer: ", len(computer_nodes))
        print("NUmber of User: ", len(user_nodes))
        print(set(access))   
        print("Raw graph")
        report(DG, True)

        remove_dead_nodes(DG)

        print(name)
        print("delete_snowball", len(delete_snowball))
        print(
            "delete_snowball percentage",
            len(delete_snowball) / DG.number_of_nodes(),
        )

        CG = DG.copy()
        # CG.remove_nodes_from(delete_snowball)
        # remove_dead_nodes(CG)



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
        report(DG)
        # display(DG, delete_snowball)
        # print("CG", name)
        # report(CG)
        # display(CG)

        write_gpickle(DG, f"/Users/huyngo/Desktop/Research/honeypot/{name}.gpickle")



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

            # print(type(start))
            # print(end)
            if start in DA_set:
                start = DA
            if end in DA_set:
                end = DA
            if start == end:
                continue
            if start != DA and not DG.has_edge(start, end):
                node_in_edge.append(start)
                node_in_edge.append(end)
                DG.add_edge(start, end)
                DG.edges[start, end]["label"] = access_type
    
        
        print(set(access))
        node_in_edge = set(node_in_edge)
        node_label = dict()
        for n in nodes:
            id = int(n["id"])      
            label = n["labels"][-1]
            node_label[id] = label
        user_nodes = []
        computer_nodes = []
        for n in node_in_edge:
            DG.nodes[n]["label"] = node_label[n]
            # print(DG.nodes[n]["label"])
            if DG.nodes[n]["label"] == "User":
                user_nodes.append(n)
            elif DG.nodes[n]["label"] == "Computer":
                computer_nodes.append(n)

        
        
        
        print("Number of Edges: ", len(DG.edges()))
        print("Number of Nodes: ",len(DG.nodes()))
        print("Niumber of Computer: ", len(computer_nodes))
        print("NUmber of User: ", len(user_nodes))
        # print(DG.edges())
        print("Raw graph")

        # report(DG, True)

        remove_dead_nodes(DG)

        # print(name)
        # print("delete_snowball", len(delete_snowball))
        # print(
        #     "delete_snowball percentage",
        #     len(delete_snowball) / DG.number_of_nodes(),
        # )

        CG = DG.copy()
        # CG.remove_nodes_from(delete_snowball)
        # remove_dead_nodes(CG)



        all_cycles = list(islice(simple_cycles(CG), 10000))
        # print(all_cycles)
        cycle_counter = Counter()
        for c in all_cycles:
            cycle_counter.update(c)
        # for p in all_cycles[:100]:
        #     print(p)
        # for v, ccount in cycle_counter.most_common(100):
        #     print(v, ccount, CG.out_degree(v))

        print("DG", name)
        report(DG)
        write_gpickle(DG, f"/Users/huyngo/Desktop/Research/honeypot/{name}_alledges.gpickle")



if __name__ == "__main__":
    # data = read_json_neo4j_3(graphs)
    data = buildgraph_DB_creator(graphs)
    print(data)