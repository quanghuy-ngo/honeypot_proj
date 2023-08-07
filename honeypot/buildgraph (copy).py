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
        "/home/andrewngo/Desktop/Research-Express/routes/AAAI/records500.json",
        [48, 55, 54] + [1182, 1333, 1431, 1507],
        [530],
    ),

    (
        "r2000",
        "/home/andrewngo/Desktop/Research-Express/routes/AAAI/records2000.json",
        [
            14166,
            14172,
            14173,
        ]
        + [18729, 19276, 19374, 19892],
        [17255, 17138, 14501, 17761],
    ),
]


for name, json_location, DA_set, delete_snowball in graphs:
    DA = DA_set[0]
    DG = nx.DiGraph(DA=DA)
    edges = json.load(io.open(json_location, "r", encoding="utf-8-sig"))

    for e in edges:
        access_type = e["p"]["segments"][0]["relationship"]["type"]
        # if access_type in [
        #     "AdminTo",
        #     "MemberOf",
        #     "HasSession",
        # ]:
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
    print(all_cycles)
    cycle_counter = Counter()
    for c in all_cycles:
        cycle_counter.update(c)
    for p in all_cycles[:100]:
        print(p)
    for v, ccount in cycle_counter.most_common(100):
        print(v, ccount, CG.out_degree(v))

    # asddsasd
    # assert len(all_cycles) == 0
    


    print("DG", name)
    report(DG)
    # display(DG, delete_snowball)
    # print("CG", name)
    # report(CG)
    # display(CG)

    write_gpickle(DG, f"/home/andrewngo/Desktop/CFR/Code/AAAI/{name}_alledge.gpickle")
    # write_gpickle(CG, f"/home/andrewngo/Desktop/Research-Express/routes/AAAI/{name}-dag_alledge.gpickle")
# MATCH p = (a)-[r]->(b)
# RETURN p