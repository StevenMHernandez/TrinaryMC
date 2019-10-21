import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

from simulator.node import Node
from simulator.point import Point
from simulator.simulator import Simulator
from trinary_mcl.main import TrinaryMCL

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

    def run(self, num_time_instances=2, num_nodes=4, num_anchors=25, stage_size=(200, 200), max_v=50, communication_radius=50):
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
        NUM_NODES_TO_RENDER = 6
        for i in range(NUM_NODES_TO_RENDER):
            self.nodes.append(Node(i, is_anchor=False, config=config, currentP=Point(i*15,round(i/2) * 25)))

        # self.nodes.append(Node(0, is_anchor=False, config=config, currentP=Point(0,0)))
        # self.nodes.append(Node(1, is_anchor=False, config=config, currentP=Point(0,45)))
        # self.nodes.append(Node(2, is_anchor=False, config=config, currentP=Point(40,40)))
        # self.nodes.append(Node(3, is_anchor=False, config=config, currentP=Point(32,30)))
        # self.nodes.append(Node(4, is_anchor=False, config=config, currentP=Point(40,60)))
        # self.nodes.append(Node(5, is_anchor=False, config=config, currentP=Point(0,-45)))
        # self.nodes.append(Node(6, is_anchor=False, config=config, currentP=Point(0,-75)))
        # self.nodes.append(Node(7, is_anchor=False, config=config, currentP=Point(-40,-50)))
        # self.nodes.append(Node(8, is_anchor=False, config=config, currentP=Point(10,-10)))
        # self.nodes.append(Node(9, is_anchor=False, config=config, currentP=Point(20,-20)))
        # self.plot_nodes(communication_radius)
        self.single_run(config, self.nodes)

    def single_run(self, config, nodes):
        self.nodes = nodes

        # Build state matrix and neighbor lists
        self.update_global_state_matrix(self.nodes, config['communication_radius'])
        self.update_one_hop_neighbors_lists(self.nodes)
        self.update_two_hop_neighbors_lists(self.nodes)

        # fig = plt.figure(figsize=(10, 10))
        # ax = [None] * len(self.algorithms)
        # for i, a in enumerate(self.algorithms):
        #     ax[i] = fig.add_subplot(2, 2, i + 1)
        #     ax[i].set_xlim((0, config['max_x']))
        #     ax[i].set_ylim((0, config['max_y']))
        #     ax[i].set_title(a)
        #     for n_i, n in enumerate(self.nodes):
        #         if n.index == 0:
        #             color = 'r'
        #         elif self.current_global_state_matrix[0][n_i] > 0:
        #             color = 'b'
        #         else:
        #             color = 'k'
        #         linestyle = '-' if n.is_anchor or n.index == 0 else '--'
        #         ax[i].add_artist(plt.Circle((n.currentP.x, n.currentP.y), config['communication_radius'], color=color, fill=False,
        #                                     linestyle=linestyle))

        # def plot_sample_set(plt, sample_set, style):
        #     x_s = [p.currentP.x for p in sample_set]
        #     y_s = [p.currentP.y for p in sample_set]
        #     plt.plot(x_s, y_s, style)

        # Make predictions for all nodes
        print(self.algorithms)
        for i, a in enumerate(self.algorithms):
            print("=====>")
            a.max_initial_sample_iterations = 4500

            # MDS calculation
            n_nodes = len(self.nodes)
            predicted_distances = np.ones((n_nodes,n_nodes)) * 0  # -1
            predicted_distances_count = np.ones((n_nodes,n_nodes))

            # Run monte carlo and plot results here.
            for n in self.nodes:
                # node = self.nodes[0]
                node = n
                # a.predict(config, self.nodes, self.current_global_state_matrix)
                sample_set = a.initialization_step(config, node)
                sample_set = a.sampling_step(config, node, sample_set)
                sample_set = a.filtering_step(config, node, sample_set, self.current_global_state_matrix)
                pred = a.predicting_step(sample_set)
                print(a)
                print(pred)

                for n2 in pred.keys():
                    predicted_distances[n.index][n2.index] = n.currentP.distance(n2.currentP)
                    predicted_distances[n2.index][n.index] = n.currentP.distance(n2.currentP)
                    predicted_distances_count[n.index][n2.index] += 1
                    predicted_distances_count[n2.index][n.index] += 1
                    print("value->",pred[n2])

            # X_true = np.array([[p.currentP.x for p in self.nodes], [p.currentP.y for p in self.nodes]])
            X_true = np.array([[p.currentP.x,p.currentP.y] for p in self.nodes])

            # Get average
            predicted_distances = predicted_distances / predicted_distances_count

            # MDS
            mds_model = MDS(n_components=2, max_iter=3000, eps=1e-9, dissimilarity="precomputed")
            # mds_positions = mds_model.fit_transform(predicted_distances)
            mds_positions = mds_model.fit(predicted_distances).embedding_

            # Rescale
            mds_positions *= np.sqrt((X_true ** 2).sum()) / np.sqrt((mds_positions ** 2).sum())

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
                            plt.plot([X_true[i,0], X_true[j,0]], [X_true[i,1], X_true[j,1]], '--r')
                            plt.plot([mds_positions[i,0], mds_positions[j,0]], [mds_positions[i,1], mds_positions[j,1]], ':b')
            plt.plot(X_true[:,0], X_true[:,1], '^r')
            plt.plot(mds_positions[:,0], mds_positions[:,1], '*b')
            plt.legend(["True positions", "MDS predictions"])


            print(X_true)
            print(mds_positions)

            plt.show()