
#kmean
# for graph in ["dyadsimx05_all"]:
#     for algo in ["mixed", "flat", "competent"]:
#         for graph_number in [5000]:
#             for cluster in [10, 50, 100]:
#                 for seed in range(10):
#                     print(
#                         f"sbatch --export=ALL,a=\"{graph}\",b=20,c=-1,d=-2,e=\"normal\",f={seed},g={int(graph_number/10)},h={graph_number},i=\"{algo}\",j={cluster} kmean_clustering.sh "
#                     )

# for graph in ["dyadsimx05_all"]:
#     for algo in ["mixed", "flat", "competent"]:
#         for graph_number in [1000]:
#             for cluster in [1, 500]:
#                 for seed in range(10):
#                     print(
#                         f"sbatch --export=ALL,a=\"{graph}\",b=20,c=-1,d=-2,e=\"normal\",f={seed},g={int(graph_number/10)},h={graph_number},i=\"{algo}\",j={cluster} kmean_clustering.sh "
#                     )


# for graph in ["dyadsimx05_all"]:
#     for algo in ["competent"]:
#         for graph_number in [1000]:
#             for cluster in [50, 100]:
#                 for seed in range(10):
#                     print(
#                         f"sbatch --export=ALL,a=\"{graph}\",b=20,c=-1,d=-2,e=\"binomial\",f={seed},g={int(graph_number/10)},h={graph_number},i=\"{algo}\",j={cluster} kmean_clustering.sh "
#                     )




#voting
# for graph in ["dyadsimx05_all"]:
#     for algo in ["mixed", "flat", "competent"]:
#         for graph_number in [5000]:
#             for batch in [1]:
#                 for seed in range(10):
#                     print(
#                         f"sbatch --export=ALL,a=\"{graph}\",b=20,c=-1,d=-2,e=\"normal\",f={seed},g={batch},h={batch},i={int(graph_number/10)},j={graph_number},k=\"{algo}\" dyMIP_sw_par.sh"
#                     )





# # single

# for graph in ["dyadsimx05_all"]:
#     for algo in ["mixed", "flat", "competent"]:
#         for graph_number in [5000]:
#             for batch in [1, 10, 50, 100]:
#                 for seed in range(10):
#                     print(
#                         f"sbatch --export=ALL,a=\"{graph}\",b=20,c=-1,d=-2,e=\"normal\",f={batch},g={graph_number},h=\"{algo}\",i={seed} dyMIP_single.sh"
#                     )


# for graph in ["dyadsimx05_all"]:
#     for algo in ["mixed", "flat", "competent"]:
#         for graph_number in [1000]:
#             for batch in [500]:
#                 for seed in range(10):
#                     print(
#                         f"sbatch --export=ALL,a=\"{graph}\",b=20,c=-1,d=-2,e=\"normal\",f={batch},g={graph_number},h=\"{algo}\",i={seed} dyMIP_single.sh"
#                     )




#monte_carlo for kmean
# for graph in ["dyadsimx05_all"]:
#     for algo in ["flat", "mixed", "competent"]:
#         for graph_number in [5000]:
#             for cluster in [10, 50, 100]:
#                 for seed in range(10):
#                     print(
#                         f"sbatch --export=ALL,a=\"kmean_{graph}_20_-1_-2.0_normal_{graph_number}_{algo}_{seed}_{cluster}\",b=\"normal\",c=10000,d=100000 monte_carlo_simulation_par_ice.sh"
#                     )




# for graph in ["dyadsimcompadapt_1_all"]:
#     for algo in ["flat", "mixed", "competent"]:
#         for graph_number in [1000]:
#             for cluster in [10, 50, 100]:
#                 for seed in range(10):
#                     print(
#                         f"sbatch --export=ALL,a=\"kmean_{graph}_5_-1_0.8_company_{graph_number}_{algo}_{seed}_{cluster}\",b=\"company\",c=800,d=8000,e={seed} monte_carlo_simulation_par_comp.sh"
#                     )


# # #monte_carlo for voting
# for graph in ["dyadsimcompadapt_1_all"]:
#     for algo in ["mixed", "flat", "competent"]:
#         for graph_number in [1000]:
#             for batch in [1, 10, 50, 100]:
#                 for seed in range(10):
#                     print(
#                         f"sbatch --export=ALL,a=\"sw_{graph}_10_-1_0.8_company_{batch}_{batch}_{graph_number}_{seed}_{algo}\",b=\"company\",c=800,d=8000 monte_carlo_simulation_par_ice.sh"
#                     )

# for graph in ["dyadsimcompadapt_1_all"]:
#     for algo in ["mixed", "flat", "competent"]:
#         for graph_number in [1000]:
#             for batch in [1, 10, 50, 100]:
#                 for seed in range(10):
#                     print(
#                         f"sbatch --export=ALL,a=\"sw_{graph}_10_-1_0.8_company_{batch}_{batch}_{graph_number}_{seed}_{algo}\",b=\"company\",c=800,d=8000,e={seed} monte_carlo_simulation_par_comp.sh"
#                     )

# # # # # #monte_carlo for single
# for graph in ["dyadsimx05_all"]:
#     for algo in ["mixed", "flat", "competent"]:
#         for graph_number in [5000]:
#             for batch in [1, 10, 50, 100]:
#                 for seed in range(10):
#                     print(
#                         f"sbatch --export=ALL,a=\"single_{graph}_20_-1_-2.0_normal_{batch}_{graph_number}_{seed}_{algo}\",b=\"normal\",c=10000,d=100000 monte_carlo_simulation_par_ice.sh"
#      )
# for graph in ["dyadsimcompadapt_1_all"]:
#     for algo in ["mixed", "flat", "competent"]:
#         for graph_number in [1000]:
#             for batch in [1, 10, 50, 100]:
#                 for seed in range(10):
#                     print(
#                         f"sbatch --export=ALL,a=\"single_{graph}_5_-1_0.8_company_{batch}_{graph_number}_{seed}_{algo}\",b=\"company\",c=800,d=8000,e={seed} monte_carlo_simulation_par_comp.sh"
#     )

#lb

# for graph in ["dyadsimcompadapt_1_all"]:
#     for algo in ["mixed", "flat", "competent"]:
#         for graph_number in [100000]:
#             for batch in [10]:
#                 for seed in range(1):
#                     print(
#                         f"sbatch --export=ALL,a=\"{graph}\",b=10,c=-1,d=0.8,e=\"normal\",f={seed},g={batch},h={batch},i={int(graph_number/5)},j={graph_number},k=\"{algo}\" dyMIP_sw_par_lb.sh"
#                     )





#static

# for graph in ["r2000", "r4000", "adsimx05", "adsimx10", "adsim025", "adsim05", "adsim10", "adsim100"]:
    # for algo in ["mixed_attack", "flat", "competent"]:
    #     for graph_number in [1000]:
    #         for batch in [10, 50, 100, 500]:
    #             for seed in range(10):
    #                 print(
    #                     f"sbatch --export=ALL,a=\"{graph}\",b=0,c=-1,d=-2,e=\"normal\",f={batch},g={graph_number},h=\"{algo}\",i={seed} dyMIP_single.sh"
    #                 )

# for graph in ["r2000", "r4000", "adsimx05", "adsimx10", "adsim025", "adsim05", "adsim10", "adsim100"]:
#     for budget in [10, 20]:
#         print(
#             f"sbatch --export=ALL,a=\"{graph}\",b={budget},c=50,d=\"DO\" competent_test.sh"
#         )







############################################################################
#Temporal test


# #kmean
# for graph in ["dyadsimcompadapt_1_all"]:
#     for algo in ["mixed"]:
#         for graph_number in [1000]:
#             for cluster in [10, 100]:
#                 for budget in [9]:
#                     for seed in range(10):
#                         print(
#                             f"sbatch --export=ALL,a=\"{graph}\",b={budget},c=-1,d=0.8,e=\"company\",f={seed},g={int(graph_number/10)},h={graph_number},i=\"{algo}\",j={cluster} kmean_clustering.sh "
#                         )
# # #voting
# for graph in ["dyadsimcompadapt_1_all"]:
#     for algo in ["mixed"]:
#         for graph_number in [1000]:
#             for batch in [10]:
#                 for budget in [6, 8, 10]:
#                     for seed in range(10):
#                         print(
#                             f"sbatch --export=ALL,a=\"{graph}\",b={budget},c=-1,d=0.8,e=\"company\",f={seed},g={batch},h={batch},i={int(graph_number/10)},j={graph_number},k=\"{algo}\" dyMIP_sw_par.sh"
#                         )
# # #single
# for graph in ["dyadsimcompadapt_1_all"]:
#     for algo in ["mixed"]:
#         for graph_number in [1000]:
#             for batch in [10]:
#                 for budget in [9]:
#                     for seed in range(10):
#                         print(
#                             f"sbatch --export=ALL,a=\"{graph}\",b={budget},c=-1,d=0.8,e=\"company\",f={batch},g={graph_number},h=\"{algo}\",i={seed} dyMIP_single.sh"
#                         )

# # # Monte carlo

# for graph in ["dyadsimcompadapt_1_all"]:
#     for algo in ["mixed"]:
#         for graph_number in [1000]:
#             for cluster in [10, 100]:
#                 for budget in [6]:
#                     for seed in range(10):
#                         print(
#                             f"sbatch --export=ALL,a=\"kmean_{graph}_{budget}_-1_0.8_company_{graph_number}_{algo}_{seed}_{cluster}\",b=\"company\",c=800,d=8000,e={seed} monte_carlo_simulation_par_comp.sh"
#                         )
# for graph in ["dyadsimcompadapt_1_all"]:
#     for algo in ["mixed"]:
#         for graph_number in [1000]:
#             for batch in [10]:
#                 for budget in [6]:
#                     for seed in range(10):
#                         print(
#                             f"sbatch --export=ALL,a=\"single_{graph}_{budget}_-1_0.8_company_{batch}_{graph_number}_{seed}_{algo}\",b=\"company\",c=800,d=8000,e={seed} monte_carlo_simulation_par_comp.sh"
#         )
# for graph in ["dyadsimcompadapt_1_all"]:
#     for algo in ["mixed"]:
#         for graph_number in [1000]:
#             for batch in [10]:
#                 for budget in [6]:
#                     for seed in range(10):
#                         print(
#                             f"sbatch --export=ALL,a=\"sw_{graph}_{budget}_-1_0.8_company_{batch}_{batch}_{graph_number}_{seed}_{algo}\",b=\"company\",c=800,d=8000,e={seed} monte_carlo_simulation_par_comp.sh"
#                         )


# retrain sw
for graph in ["dyadsimcompadapt_1_all"]:
    for algo in ["mixed"]:
        for graph_number in [1000]:
            for batch in [10]:
                for budget in [6, 8, 10]:
                    for seed in range(10):
                        print(
                            f"sbatch --export=ALL,a=\"{graph}\",b={budget},c=-1,d=0.8,e=\"retrain\",f={seed},g={batch},h={batch},i={int(graph_number/10)},j={graph_number},k=\"{algo}\" dyMIP_sw_par.sh"
                        )

# retrain montecarlo
# for graph in ["dyadsimcompadapt_1_all"]:
#     for algo in ["mixed"]:
#         for graph_number in [1000]:
#             for batch in [10]:
#                 for budget in [6, 8, 10]:
#                     for seed in range(10):
#                         print(
#                             f"sbatch --export=ALL,a=\"sw_{graph}_{budget}_-1_0.8_retrain_{batch}_{batch}_{graph_number}_{seed}_{algo}\",b=\"company\",c=800,d=8000,e={seed} monte_carlo_simulation_par_comp.sh"
#                         )




for graph in ["dyadsimcompadapt_1_all"]:
    for algo in ["mixed"]:
        for graph_number in [2000]:
            for batch in [10]:
                for budget in [6, 8, 10]:
                    for seed in range(10):
                        print(
                            f"sbatch --export=ALL,a=\"{graph}\",b={budget},c=-1,d=0.8,e=\"company\",f={seed},g={batch},h={batch},i={int(graph_number/5)},j={graph_number},k=\"{algo}\" dyMIP_sw_par_lb.sh"
                        )




# for graph in ["dyadsim100_alledges_3"]:
#     for algo in ["mixed"]:
#         for graph_number in [2000]:
#             for batch in [10]:
#                 for budget in [10, 15, 20]:
#                     for seed in range(10):
#                         print(
#                             f"sbatch --export=ALL,a=\"{graph}\",b={budget},c=100,d=0.8,e=\"factor\",f={seed},g={batch},h={batch},i={int(graph_number/5)},j={graph_number},k=\"{algo}\" dyMIP_sw_par_lb.sh"
#                         )


###############################################################################
#on adsimcomp normal


#kmean
# for graph in ["dyadsimcompadapt_1_all"]:
#     for algo in ["mixed", "flat", "competent"]:
#         for graph_number in [1000]:
#             for cluster in [10]:
#                 for seed in range(10):
#                     print(
#                         f"sbatch --export=ALL,a=\"{graph}\",b=10,c=-1,d=0.8,e=\"normal\",f={seed},g={int(graph_number/10)},h={graph_number},i=\"{algo}\",j={cluster} kmean_clustering.sh "
#                     )
# #voting
# for graph in ["dyadsimcompadapt_1_all"]:
#     for algo in ["mixed", "flat", "competent"]:
#         for graph_number in [1000]:
#             for batch in [10]:
#                 for seed in range(10):
#                     print(
#                         f"sbatch --export=ALL,a=\"{graph}\",b=10,c=-1,d=0.8,e=\"normal\",f={seed},g={batch},h={batch},i={int(graph_number/10)},j={graph_number},k=\"{algo}\" dyMIP_sw_par.sh"
#                     )

# #single 
# for graph in ["dyadsimcompadapt_1_all"]:
#     for algo in ["mixed", "flat", "competent"]:
#         for graph_number in [1000]:
#             for batch in [10]:
#                 for seed in range(10):
#                     print(
#                         f"sbatch --export=ALL,a=\"{graph}\",b=10,c=-1,d=0.8,e=\"normal\",f={batch},g={graph_number},h=\"{algo}\",i={seed} dyMIP_single.sh"
#                     )

#monte carlo adsimcomp normal

# for graph in ["dyadsimcompadapt_1_all"]:
#     for algo in ["mixed"]:
#         for graph_number in [1000]:
#             for cluster in [10, 100]:
#                 for budget in [8]:
#                     for seed in range(10):
#                         print(
#                             f"sbatch --export=ALL,a=\"kmean_{graph}_{budget}_-1_0.8_normal_{graph_number}_{algo}_{seed}_{cluster}\",b=\"normal\",c=800,d=8000,e={seed} monte_carlo_simulation_par_comp.sh"
#                         )

# for graph in ["dyadsimcompadapt_1_all"]:
#     for algo in ["mixed"]:
#         for graph_number in [1000]:
#             for cluster in [10, 50, 100]:
#                 for budget in [10]:
#                     for seed in range(10):
#                         print(
#                             f"sbatch --export=ALL,a=\"kmean_{graph}_{budget}_-1_0.8_normal_{graph_number}_{algo}_{seed}_{cluster}\",b=\"normal\",c=10000,d=100000 monte_carlo_simulation_par_ice.sh"
#                         )

# for graph in ["dyadsimcompadapt_1_all"]:
#     for algo in ["mixed"]:
#         for graph_number in [1000]:
#             for batch in [1, 10, 50, 100]:
#                 for budget in [10]:
#                     for seed in range(10):
#                         print(
#                             f"sbatch --export=ALL,a=\"single_{graph}_{budget}_-1_0.8_normal_{batch}_{graph_number}_{seed}_{algo}\",b=\"normal\",c=10000,d=100000 monte_carlo_simulation_par_ice.sh"
#         )
                        

# for graph in ["dyadsimcompadapt_1_all"]:
#     for algo in ["mixed"]:
#         for graph_number in [1000]:
#             for batch in [1, 10, 50, 100]:
#                 for budget in [10]:
#                     for seed in range(10):
#                         print(
#                             f"sbatch --export=ALL,a=\"sw_{graph}_10_-1_0.8_normal_{batch}_{batch}_{graph_number}_{seed}_{algo}\",b=\"normal\",c=10000,d=100000 monte_carlo_simulation_par_ice.sh"
#                         )




# for graph in ["dyadsim100_alledges_3"]:
#     for algo in ["mixed"]:
#         for graph_number in [1000]:
#             for cluster in [10]:
#                 for blockable_p in [1, 0.9, 0.8]:
#                     for budget in [10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
#                         for seed in range(1):
#                             print(
#                                 f"sbatch --export=ALL,a=\"{graph}\",b={budget},c=100,d={blockable_p},e=\"factor\",f={seed},g={int(graph_number/10)},h={graph_number},i=\"{algo}\",j={cluster} kmean_clustering.sh "
#                             )

# for graph in ["dyadsim100_alledges_3"]:
#     for algo in ["mixed"]:
#         for graph_number in [1000]:
#             for cluster in [10]:
#                 for blockable_p in [1, 0.9, 0.8]:
#                     for budget in [10, 15, 20]:
#                         for seed in range(10):
#                             print(
#                                 f"sbatch --export=ALL,a=\"{graph}\",b={budget},c=100,d={blockable_p},e=\"factor\",f={seed},g={int(graph_number/10)},h={graph_number},i=\"{algo}\",j={cluster} kmean_clustering.sh "
#                             )
# #voting
# for graph in ["dyadsim100_alledges_3"]:
#     for algo in ["mixed"]:
#         for graph_number in [1000]:
#             for batch in [10]:
#                 for blockable_p in [1, 0.9]:
#                     for budget in [10, 15, 20]:
#                         for seed in range(10):
#                             print(
#                                 f"sbatch --export=ALL,a=\"{graph}\",b={budget},c=500,d={blockable_p},e=\"factor\",f={seed},g={batch},h={batch},i={int(graph_number/10)},j={graph_number},k=\"{algo}\" dyMIP_sw_par.sh"
                            # )
# # #single
# for graph in ["dyadsim100_alledges_3"]:
#     for algo in ["mixed"]:
#         for graph_number in [1000]:
#             for batch in [10]:
#                 for blockable_p in [0.8, 0.9, 1]:
#                     for budget in [10, 15, 20]:
#                         for seed in range(10):
#                             print(
#                                 f"sbatch --export=ALL,a=\"{graph}\",b={budget},c=100,d={blockable_p},e=\"factor\",f={batch},g={graph_number},h=\"{algo}\",i={seed} dyMIP_single.sh"
#                             )

# for graph in ["dyadsim100_alledges_3"]:
#     for algo in ["mixed"]:
#         for graph_number in [1000]:
#             for batch in [10]:
#                 for blockable_p in [1, 0.9, 0.8]:
#                     for budget in [10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
#                         for seed in range(1):
#                             print(
#                                 f"sbatch --export=ALL,a=\"{graph}\",b={budget},c=,d={blockable_p},e=\"factor\",f={batch},g={graph_number},h=\"{algo}\",i={seed} dyMIP_single.sh"
#                             )


# for graph in ["dyadsim100_alledges_3"]:
#     for algo in ["mixed"]:
#         for graph_number in [1000]:
#             for batch in [10]:
#                 for blockable_p in [0.8, 0.9, 1.0]:
#                     for budget in [10, 15, 20]:
#                         for seed in range(10):
#                             print(
#                                 f"sbatch --export=ALL,a=\"single_{graph}_{budget}_100_{blockable_p}_factor_{batch}_{graph_number}_{seed}_{algo}\",b=\"factor\",c=200,d=2000,e={seed} monte_carlo_simulation_par_comp.sh"
#             )



# for graph in ["dyadsim100_alledges_3"]:
#     for algo in ["mixed"]:
#         for graph_number in [1000]:
#             for batch in [10]:
#                 for blockable_p in [0.9, 0.8, 1.0]:
#                     for budget in [10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
#                         for seed in range(1):
#                             print(
#                                 f"sbatch --export=ALL,a=\"single_{graph}_{budget}_100_{blockable_p}_factor_{batch}_{graph_number}_{seed}_{algo}\",b=\"factor\",c=200,d=2000,e={seed} monte_carlo_simulation_par_comp.sh"
#             )


#note:


# training:
#ex1: ["kmean", sw, single] budget = [10, 15, 20], blockable_p = [1, 0.9], seed 0->9, batch number = 10
#ex2: kmean budget = [10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100], blockable_p = [1, 0.9, 0.8], batch = 10, seed = 1


# for i in range(971447,971504 + 1):
#     print(
#         f"scancel {i}"
#         )