import math
import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

from simulator.node import Node
from simulator.point import Point
from simulator.simulator import Simulator
from mcl_algorithms.trinary_mcl.main import TrinaryMCL

STATE_INVISIBLE = 0
STATE_APPROACHING = 1
STATE_RETREATING = 2


class SimulatorSingleStepMDS(Simulator):
    algorithms = [
        TrinaryMCL(),
    ]

    def plot_nodes(self, communication_radius):
        plt.plot([p.currentP.x for p in self.nodes], [p.currentP.y for p in self.nodes], '*k')
        for n1 in self.nodes:
            for n2 in self.nodes:
                if n1.index != n2.index:
                    if n1.currentP.distance(n2.currentP) < communication_radius:
                        plt.plot([n1.currentP.x, n2.currentP.x], [n1.currentP.y, n2.currentP.y], ':k')
        plt.show()

    def run(self, num_time_instances=2, num_nodes=4, num_anchors=25, stage_size=(200, 200), max_v=50, communication_radius=50, should_plot=True):
        config = {
            'max_x': stage_size[0],
            'max_y': stage_size[1],
            'max_v': max_v,
            'communication_radius': communication_radius,
        }

        self.algorithms = [
            TrinaryMCL(),
        ]

        # Trinary Example
        self.nodes = []
        NUM_NODES_TO_RENDER = num_nodes
        # random.seed(NUM_NODES_TO_RENDER)
        for i in range(NUM_NODES_TO_RENDER):
            if i > 0:
                MAX_MOVEMENT = 2.0
                new_p = Point(random.random() * (config['communication_radius'] * MAX_MOVEMENT) - (config['communication_radius'] * MAX_MOVEMENT / 2),
                              random.random() * (config['communication_radius'] * MAX_MOVEMENT) - (config['communication_radius'] * MAX_MOVEMENT / 2))
                random_int = i - math.ceil((min(5,i)) * random.random())
                new_p.x += self.nodes[random_int].currentP.x
                new_p.y += self.nodes[random_int].currentP.y
                self.nodes.append(Node(i, is_anchor=False, config=config, currentP=new_p))
            else:
                self.nodes.append(Node(i, is_anchor=False, config=config, currentP=Point(0, 0)))

        # Build state matrix and neighbor lists
        # Time=1
        self.update_global_state_matrix(self.nodes, config['communication_radius'])
        self.update_one_hop_neighbors_lists(self.nodes)
        self.update_two_hop_neighbors_lists(self.nodes)

        # Move each node
        for n in self.nodes:  # type: Node
            n.move()

        if should_plot:
            self.plot_nodes(config['communication_radius'])

        # Build state matrix and neighbor lists
        # Time=2
        self.update_global_state_matrix(self.nodes, config['communication_radius'])
        self.update_one_hop_neighbors_lists(self.nodes)
        self.update_two_hop_neighbors_lists(self.nodes)

        # Make predictions for all nodes
        for algo in self.algorithms:
            # MDS calculation
            n_nodes = len(self.nodes)
            predicted_distances = np.ones((n_nodes,n_nodes)) * 100  # 50  # -1  # 100
            predicted_distances_count = np.ones((n_nodes,n_nodes))

            for n in self.nodes:
                n.one_hop_neighbor_predicted_distances[algo.name()] = {}

            algo.predict(config, self.nodes, self.current_global_state_matrix)

            for n in self.nodes:
                for n2 in n.one_hop_neighbors:
                    predicted_distances[n.index][n2.index] = n.currentP.distance(n2.currentP)
                    predicted_distances[n2.index][n.index] = n.currentP.distance(n2.currentP)
                    predicted_distances_count[n.index][n2.index] += 1
                    predicted_distances_count[n2.index][n.index] += 1
                for n2 in n.two_hop_neighbors:
                    predicted_distances[n.index][n2.index] = n.currentP.distance(n2.currentP)
                    predicted_distances[n2.index][n.index] = n.currentP.distance(n2.currentP)
                    predicted_distances_count[n.index][n2.index] += 1
                    predicted_distances_count[n2.index][n.index] += 1

            X_true = np.array([[p.currentP.x,p.currentP.y] for p in self.nodes])

            # Get average
            predicted_distances = predicted_distances / predicted_distances_count

            # MDS
            mds_model = MDS(n_components=2, max_iter=3000, eps=1e-19, dissimilarity="precomputed")
            mds_positions = mds_model.fit_transform(predicted_distances)
            # mds_positions = mds_model.fit(predicted_distances).embedding_

            # Rotate
            clf = PCA(n_components=2)
            X_true = clf.fit_transform(X_true)
            mds_positions = clf.fit_transform(mds_positions)

            # Rescale
            mds_positions *= np.sqrt((X_true ** 2).sum()) / np.sqrt((mds_positions ** 2).sum())
            # new_X_true = X_true * np.sqrt((mds_positions ** 2).sum()) / np.sqrt((X_true ** 2).sum())
            # new_mds_positions = mds_positions * np.sqrt((X_true ** 2).sum()) / np.sqrt((mds_positions ** 2).sum())

            # Scale
            for i in [0,1]:
                d_true = X_true[:,i].max() - X_true[:,i].min()
                d_mds = mds_positions[:,i].max() - mds_positions[:,i].min()
                mds_positions[:,i] *= (d_true / d_mds)

            # Rotate
            clf = PCA(n_components=2)
            X_true = clf.fit_transform(X_true)
            mds_positions = clf.fit_transform(mds_positions)

            for i in range(len(self.nodes)):
                n1 = self.nodes[i]
                for j in range(len(self.nodes)):
                    n2 = self.nodes[j]
                    if i != j:
                        print("not the same!")
                        if n1.currentP.distance(n2.currentP) < config['communication_radius']:
                            print("in distance!")
                            # plt.plot([n1.currentP.x, n2.currentP.x], [n1.currentP.y, n2.currentP.y], '--r')
                            # plt.plot([X_true[i,0], X_true[j,0]], [X_true[i,1], X_true[j,1]], '--r')
                            # plt.plot([mds_positions[i,0], mds_positions[j,0]], [mds_positions[i,1], mds_positions[j,1]], ':b')
            if should_plot:
                plt.plot(X_true[:,0], X_true[:,1], '^r')
                plt.plot(mds_positions[:,0], mds_positions[:,1], '*b')

                for i in range(n_nodes):
                    plt.text(X_true[i,0], X_true[i,1], str(i), color='r')
                    plt.text(mds_positions[i,0], mds_positions[i,1], str(i), color='b')

                plt.legend(["True positions", "MDS predictions"])

            print(X_true)
            print(mds_positions)

            err = 0
            for i in range(len(X_true)):
                err = math.sqrt((X_true[i, 0] - mds_positions[i, 0])**2 + (X_true[i, 1] - mds_positions[i, 1])**2)
            print("Error: ", err)

            if should_plot:
                plt.show()

            return {'error': err}