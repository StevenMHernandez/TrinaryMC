import random
from time import time

import matplotlib.pyplot as plt
import numpy as np

from simulator.simulator import Simulator
from mcl_algorithms.trinary_mcl.main import TrinaryMCL

if __name__ == "__main__":
    start = time()
    simulator = Simulator()

    num_time_instances = 100
    percentage_are_anchors = 0.2

    num_nodes = 50
    communication_radius = 50
    max_width = 500
    max_height = 500
    max_v = 10

    # random.seed(0)

    simulator_results, algorithm_results = simulator.run(num_time_instances=num_time_instances,
                                                         num_nodes=num_nodes,
                                                         num_anchors=num_nodes * percentage_are_anchors,
                                                         stage_size=(max_width, max_height),
                                                         max_v=max_v,
                                                         communication_radius=communication_radius)

    PLOT_NODE_POSITIONS = False
    PLOT_NUMBER_OF_NODES = False
    PLOT_NUMBER_OF_NEIGHBORS = False
    PLOT_COMPUTATION_TIME = False
    PLOT_NUMBER_OF_SAMPLES = False
    PLOT_COMMUNICATION = True
    PLOT_DISTANCE_ERROR = False
    PLOT_POSITION_ERROR = False
    PLOT_CDF = False

    def cdf(X, reverse=False):
        if isinstance(X, list):
            X = np.array(X)
        Y = np.zeros(X.shape)
        for i in range(len(X)):
            if reverse:
                Y[i] = np.ma.count(X[X >= X[i]])
            else:
                Y[i] = np.ma.count(X[X <= X[i]])
        return Y / len(X)

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
        for algo in simulator.algorithms:
            plt.plot(algorithm_results['prediction_time'][algo.name()])
        plt.xlabel("Experiment Time (s)")
        plt.ylabel("Time to compute predictions for all nodes (s)")
        plt.legend([type(a) for a in simulator.algorithms])
        plt.show()

    if PLOT_NUMBER_OF_NEIGHBORS:
        X1 = sorted([len(n.one_hop_neighbors) for n in simulator.nodes])
        plt.plot(X1,cdf(X1))
        X2 = sorted([len(n.two_hop_neighbors) for n in simulator.nodes])
        plt.plot(X2, cdf(X2))
        plt.legend(["1-hop neighbors", "2-hop neighbors"])
        plt.xlabel("Number of k-hop neighbors")
        plt.ylabel("CDF")
        plt.show()

    if PLOT_NUMBER_OF_SAMPLES:
        for algo in simulator.algorithms:
            plt.plot(np.mean(algorithm_results['number_of_samples'][algo.name()], axis=1))
        plt.xlabel("Experiment Time (s)")
        plt.ylabel("Number of Samples at end of algorithm")
        plt.legend([type(a) for a in simulator.algorithms])
        plt.show()
        for algo in simulator.algorithms:
            X = algorithm_results['number_of_samples'][algo.name()]
            X = np.array(X)
            X = X.reshape([1, X.shape[0] * X.shape[1]]).tolist()[0]
            X = sorted(X,reverse=True)
            print(X)
            plt.plot(X, cdf(X,reverse=True))
        plt.xlabel("Number of Samples")
        plt.ylabel("CDF")
        plt.xlim([max(X), min(X)])
        plt.legend([type(a) for a in simulator.algorithms])
        plt.show()

    if PLOT_COMMUNICATION:
        for algo in simulator.algorithms:
            plt.plot(algorithm_results['number_of_packets'][algo.name()])
            print("Final Number of Packets", algo.name(), algorithm_results['number_of_packets'][algo.name()][-1])
        plt.xlabel("Time (s)")
        plt.ylabel("Number of Unique Packet Transmissions")
        plt.legend([type(a) for a in simulator.algorithms])
        plt.show()

    if PLOT_POSITION_ERROR:
        print(algorithm_results['position_error'])
        for algo in simulator.algorithms:
            if not isinstance(algo, TrinaryMCL):
                print(algo, "average accuracy:", sum(algorithm_results['position_error'][algo.name()]) / communication_radius / len(algorithm_results['position_error'][algo.name()]))
                plt.plot(range(1,num_time_instances+1), np.array(algorithm_results['position_error'][algo.name()]) / communication_radius, ':')
        plt.xlabel("Time (s)")
        plt.ylabel("Position Prediction Error")
        plt.legend([type(a) for a in simulator.algorithms if not isinstance(a, TrinaryMCL)])
        plt.show()

    if PLOT_DISTANCE_ERROR:
        print(algorithm_results['distance_error'])
        for algo in simulator.algorithms:
            if len(algorithm_results['distance_error'][algo.name()]):
                print(algo, "average accuracy:", sum(algorithm_results['distance_error'][algo.name()]) / communication_radius / len(algorithm_results['distance_error'][algo.name()]))
                plt.plot(range(1,num_time_instances+1), np.array(algorithm_results['distance_error'][algo.name()]) / communication_radius, ':')
        plt.xlabel("Time (s)")
        plt.ylabel("Relative Distance Error")
        plt.legend([type(a) for a in simulator.algorithms])
        plt.show()

    if PLOT_CDF:
        for algo_i,algo in enumerate(simulator.algorithms):
            X_prime = algorithm_results['distance_error_all'][algo.name()]
            l = sum([len(x) for x in X_prime])
            X = []
            for x in X_prime:
                X += x

            X = np.array(sorted(X)) / communication_radius
            # X = np.array(sorted(algorithm_results['distance_error_all'][algo.name()][-1])) / communication_radius
            plt.plot(X, cdf(X))
            print("%",algo.name())
            print("X_{} = [{}]".format(algo_i, ",".join([str(x) for x in X])))
            print("CDF_{} = [{}]".format(algo_i, ",".join([str(x) for x in cdf(X)])))
        plt.legend([a.name() for a in simulator.algorithms], loc='lower right')
        plt.xlabel("Error")
        plt.ylabel("CDF")
        plt.xlim([0,1])
        plt.show()


    print("Took", time() - start, "seconds")
