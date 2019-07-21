from time import time

import matplotlib.pyplot as plt

from simulator.simulator import Simulator

if __name__ == "__main__":
    start = time()
    simulator = Simulator()
    simulator_results, algorithm_results = simulator.run(num_time_instances=10,
                                                         num_nodes=100,
                                                         num_anchors=25,
                                                         stage_size=(1000, 1000),
                                                         max_v=10,
                                                         communication_radius=100)

    PLOT_NODE_POSITIONS = False
    PLOT_COMMUNICATION = False
    PLOT_ACCURACY = True

    if PLOT_NODE_POSITIONS:
        for points in simulator_results['node_positions']:
            plt.plot(list(map(lambda p: p.x, points)), list(map(lambda p: p.y, points)), '-')
        plt.show()

    if PLOT_COMMUNICATION:
        for algorithms in simulator.algorithms:
            plt.plot(algorithm_results['number_of_packets'][algorithms])
        plt.xlabel("Time (s)")
        plt.ylabel("Number of Unique Packet Transmissions")
        plt.legend([type(a) for a in simulator.algorithms])
        plt.show()

    if PLOT_ACCURACY:
        print(algorithm_results['accuracy'])
        for a in simulator.algorithms:
            print(a, "average accuracy:", sum(algorithm_results['accuracy'][a]) / len(algorithm_results['accuracy'][a]))
            plt.plot(algorithm_results['accuracy'][a])
        plt.xlabel("Time (s)")
        plt.ylabel("Prediction Accuracy")
        plt.legend([type(a) for a in simulator.algorithms])
        plt.show()

    print("Took", time() - start, "seconds")
