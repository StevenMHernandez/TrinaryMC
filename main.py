from time import time

import matplotlib.pyplot as plt
import numpy as np

from simulator.simulator import Simulator
from trinary_mcl.main import TrinaryMCL

if __name__ == "__main__":
    start = time()
    simulator = Simulator()
    communication_radius = 100
    num_time_instances = 10
    simulator_results, algorithm_results = simulator.run(num_time_instances=num_time_instances,
                                                         num_nodes=100,
                                                         num_anchors=10,
                                                         stage_size=(1000, 1000),
                                                         max_v=10,
                                                         communication_radius=communication_radius)

    PLOT_NODE_POSITIONS = False
    PLOT_NUMBER_OF_NODES = True
    PLOT_COMPUTATION_TIME = True
    PLOT_COMMUNICATION = False
    PLOT_DISTANCE_ERROR = True
    PLOT_POSITION_ERROR = True

    if PLOT_NODE_POSITIONS:
        for points in simulator_results['node_positions']:
            plt.plot(list(map(lambda p: p.x, points)), list(map(lambda p: p.y, points)), '-')
        plt.show()

    if PLOT_NUMBER_OF_NODES:
        plt.plot(simulator_results['avg_number_of_first_hop_neighbors'])
        plt.plot(simulator_results['avg_number_of_second_hop_neighbors'])
        plt.legend(["1st hop neighbors", "2nd hop neighbors"])
        plt.show()

    if PLOT_COMPUTATION_TIME:
        for algorithms in simulator.algorithms:
            plt.plot(algorithm_results['prediction_time'][algorithms])
        plt.xlabel("Experiment Time (s)")
        plt.ylabel("Time to compute predictions for all nodes (s)")
        plt.legend([type(a) for a in simulator.algorithms])
        plt.show()

    if PLOT_COMMUNICATION:
        for algorithms in simulator.algorithms:
            plt.plot(algorithm_results['number_of_packets'][algorithms])
        plt.xlabel("Time (s)")
        plt.ylabel("Number of Unique Packet Transmissions")
        plt.legend([type(a) for a in simulator.algorithms])
        plt.show()

    if PLOT_POSITION_ERROR:
        print(algorithm_results['position_error'])
        for a in simulator.algorithms:
            if type(a) is not TrinaryMCL:
                print(a, "average accuracy:", sum(algorithm_results['position_error'][a]) / communication_radius / len(algorithm_results['position_error'][a]))
                plt.plot(range(1,num_time_instances+1), np.array(algorithm_results['position_error'][a]) / communication_radius, ':')
        plt.xlabel("Time (s)")
        plt.ylabel("Position Prediction Error")
        plt.legend([type(a) for a in simulator.algorithms if type(a) is not TrinaryMCL])
        plt.show()

    if PLOT_DISTANCE_ERROR:
        print(algorithm_results['distance_error'])
        for a in simulator.algorithms:
            print(a, "average accuracy:", sum(algorithm_results['distance_error'][a]) / communication_radius / len(algorithm_results['distance_error'][a]))
            plt.plot(range(1,num_time_instances+1), np.array(algorithm_results['distance_error'][a]) / communication_radius, ':')
        plt.xlabel("Time (s)")
        plt.ylabel("Relative Distance Error")
        plt.legend([type(a) for a in simulator.algorithms])
        plt.show()

    print("Took", time() - start, "seconds")
