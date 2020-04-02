import matplotlib.pyplot as plt

from simulator.node import Node
from simulator.point import Point
from simulator.simulator import Simulator
from mcl_algorithms.st_mcl.main import StMCL
from mcl_algorithms.va_mcl.main import VA_MCL
from mcl_algorithms.orbit_mcl import OrbitMCL
from mcl_algorithms.lcc_mcl.main import LCC_MCL

STATE_INVISIBLE = 0
STATE_APPROACHING = 1
STATE_RETREATING = 2


class SimulatorSingleStep(Simulator):
    algorithms = [
        StMCL(),
        VA_MCL(),
        OrbitMCL(),
        LCC_MCL(),
    ]

    def run(self, num_time_instances=2, num_nodes=4, num_anchors=25, stage_size=(200, 200), max_v=50, communication_radius=50):
        config = {
            'max_x': stage_size[0],
            'max_y': stage_size[1],
            'max_v': max_v,
            'communication_radius': communication_radius,
        }

        # VA Example
        self.nodes = []
        self.nodes.append(Node(0, is_anchor=False, config=config, currentP=Point(60, 100)))
        self.nodes.append(Node(1, is_anchor=True, config=config, currentP=Point(100, 100)))
        self.nodes.append(Node(2, is_anchor=True, config=config, currentP=Point(60, 60)))
        self.nodes.append(Node(3, is_anchor=False, config=config, currentP=Point(120, 130)))
        self.nodes.append(Node(4, is_anchor=True, config=config, currentP=Point(120, 70)))
        self.single_run(config, self.nodes)

        # Orbit Example
        self.nodes = []
        self.nodes.append(Node(0, is_anchor=False, config=config, currentP=Point(100, 100)))
        self.nodes.append(Node(1, is_anchor=True, config=config, currentP=Point(110, 100)))
        self.nodes.append(Node(2, is_anchor=True, config=config, currentP=Point(90, 100)))
        self.nodes.append(Node(3, is_anchor=True, config=config, currentP=Point(135, 140)))
        self.nodes.append(Node(4, is_anchor=True, config=config, currentP=Point(50, 87)))
        self.single_run(config, self.nodes)

        # LCC Example
        self.nodes = []
        self.nodes.append(Node(0, is_anchor=False, config=config, currentP=Point(100, 100)))
        self.nodes.append(Node(1, is_anchor=True, config=config, currentP=Point(140, 100)))
        self.nodes.append(Node(2, is_anchor=True, config=config, currentP=Point(140, 110)))
        self.nodes.append(Node(2, is_anchor=True, config=config, currentP=Point(130, 110)))
        self.nodes.append(Node(3, is_anchor=False, config=config, currentP=Point(140, 90)))
        self.nodes.append(Node(4, is_anchor=False, config=config, currentP=Point(65, 100)))
        self.nodes.append(Node(4, is_anchor=False, config=config, currentP=Point(100, 75)))
        self.nodes.append(Node(4, is_anchor=False, config=config, currentP=Point(150, 75)))
        self.single_run(config, self.nodes)

    def single_run(self, config, nodes):
        self.nodes = nodes

        # Give each non anchor node a perfectly predicted currentP (For LCC evaluation)
        for n in [n for n in self.nodes if not n.is_anchor]:  # type: Node
            for algo in self.algorithms:
                n.p_pred[algo] = n.currentP

        # Build state matrix and neighbor lists
        self.update_global_state_matrix(self.nodes, config['communication_radius'])
        self.update_one_hop_neighbors_lists(self.nodes)
        self.update_two_hop_neighbors_lists(self.nodes)

        fig = plt.figure(figsize=(10, 10))

        ax = [None] * len(self.algorithms)

        for i, a in enumerate(self.algorithms):
            ax[i] = fig.add_subplot(2, 2, i + 1)
            ax[i].set_xlim((0, config['max_x']))
            ax[i].set_ylim((0, config['max_y']))
            ax[i].set_title(a)
            for n_i, n in enumerate(self.nodes):
                if n.index == 0:
                    color = 'r'
                elif self.current_global_state_matrix[0][n_i] > 0:
                    color = 'b'
                else:
                    color = 'k'
                linestyle = '-' if n.is_anchor or n.index == 0 else '--'
                ax[i].add_artist(plt.Circle((n.currentP.x, n.currentP.y), config['communication_radius'], color=color, fill=False,
                                            linestyle=linestyle))

        def plot_sample_set(plt, sample_set, style):
            x_s = [p.x for p in sample_set]
            y_s = [p.y for p in sample_set]
            plt.plot(x_s, y_s, style)

        # Make predictions for all nodes
        for i, a in enumerate(self.algorithms):
            a.max_initial_sample_iterations = 1000

            # Run monte carlo and plot results here.
            node = self.nodes[0]
            sample_set = a.initialization_step(config, node)
            plot_sample_set(ax[i], sample_set, 'k*')
            # sample_set = a.sampling_step(config, node, sample_set)
            # # plot_sample_set(ax[i], sample_set, 'r*')
            # if len(sample_set) > a.sample_threshold:
            #     shuffle(sample_set)
            #     sample_set = sample_set[0:a.sample_threshold]
            #     # plot_sample_set(ax[i], sample_set, 'b*')
            sample_set = a.filtering_step(config, node, sample_set, self.current_global_state_matrix)
            plot_sample_set(ax[i], sample_set, 'g*')
            # return sample_set, self.predicting_step(sample_set)

        for i, a in enumerate(self.algorithms):
            for n in self.nodes:
                if n.index == 0:
                    color = 'r'
                elif self.current_global_state_matrix[0][n_i] > 0:
                    color = 'b'
                else:
                    color = 'k'
                ax[i].plot(n.currentP.x, n.currentP.y, color + "o")

        fig.show()