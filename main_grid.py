import random
from time import time

import matplotlib.pyplot as plt
import numpy as np

from simulator.simulator import Simulator

if __name__ == "__main__":
    total_start = time()

    percentage_are_anchors_list = [0.1, 0.2, 0.3, 0.4, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    total_algorithm_results = {
        # 'number_of_packets': {},
        'distance_error': {},
    }

    num_re_runs = 1
    num_time_instances = 100

    num_nodes = 50
    communication_radius = 50
    max_width = 500
    max_height = 500
    max_v = 10

    # Initiate
    for k in total_algorithm_results.keys():
        total_algorithm_results[k] = {}
        for algo in Simulator().algorithms:
            total_algorithm_results[k][type(algo)] = np.zeros((len(percentage_are_anchors_list),1))

    # Run Grid
    for i, percentage_are_anchors in enumerate(percentage_are_anchors_list):
        for re_runs in range(num_re_runs):
            start = time()
            simulator = Simulator()

            # random.seed((i+1) * 5)

            simulator_results, algorithm_results = simulator.run(num_time_instances=num_time_instances,
                                                                 num_nodes=num_nodes,
                                                                 num_anchors=num_nodes * percentage_are_anchors,
                                                                 stage_size=(max_width, max_height),
                                                                 max_v=max_v,
                                                                 communication_radius=communication_radius)

            for k in total_algorithm_results.keys():
                for algo in simulator.algorithms:
                    total_algorithm_results[k][type(algo)][i] += np.mean(algorithm_results[k][type(algo)]) / num_re_runs

            print("Simulation", i + 1, "/", len(percentage_are_anchors_list),"Took", time() - start, "seconds")

    print('=====')
    print({
        'num_nodes':num_nodes,
        'communication_radius': communication_radius,
        'max_width': max_width,
        'max_height': max_height,
        'max_v': max_v,

    })
    print(percentage_are_anchors_list)
    for k in total_algorithm_results.keys():
        for algo in Simulator().algorithms:
            print([x[0] for x in total_algorithm_results[k][type(algo)]])
            plt.plot(percentage_are_anchors_list, total_algorithm_results[k][type(algo)], ':')
        plt.legend(Simulator().algorithms)
        plt.show()

    print("Total Experiment Took", time() - total_start, "seconds")