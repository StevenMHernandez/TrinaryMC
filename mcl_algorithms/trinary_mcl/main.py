import math
from random import uniform, random

import numpy as np

from simulator.node import Node
from simulator.point import Point
from mcl_algorithms.st_mcl.main import StMCL
from mcl_algorithms.trinary_mcl.sample_set import SampleSet

STATE_INVISIBLE = 0
STATE_APPROACHING = 1
STATE_RETREATING = 2


class TrinaryMCL(StMCL):
    def __init__(self, k_hop_neighbors=2):
        super(TrinaryMCL, self).__init__()
        self.max_initial_sample_iterations = 100
        self.k_hop_neighbors = k_hop_neighbors
        return

    def name(self):
        return "trinary(k-hop:" + str(self.k_hop_neighbors) + ")"

    def _generate_sub_sample(self, config, placed, to_be_placed: Node):
        max_x = math.inf
        max_y = math.inf
        min_x = -math.inf
        min_y = -math.inf
        for n, p in placed.items():  # type: Node, Point
            if to_be_placed in n.one_hop_neighbors:
                max_x = min(max_x, p.x + config['communication_radius'])
                max_y = min(max_y, p.y + config['communication_radius'])
                min_x = max(min_x, p.x - config['communication_radius'])
                min_y = max(min_y, p.y - config['communication_radius'])

        return Point(uniform(min_x, max_x), uniform(min_y, max_y))

    def _generate_sample(self, config, node):
        sample_set = SampleSet()

        sample_set.node_dict[node] = Point(0, 0)

        for n in node.one_hop_neighbors:
            placed = sample_set.node_dict
            sample_set.node_dict[n] = self._generate_sub_sample(config, placed, n)

        for n in node.two_hop_neighbors:
            placed = sample_set.node_dict
            sample_set.node_dict[n] = self._generate_sub_sample(config, placed, n)

        return sample_set

    def initialization_step(self, config, node):
        sample_set = []
        for i in range(self.max_initial_sample_iterations):
            sample_set.append(self._generate_sample(config, node))

        return sample_set

    def sampling_step(self, config, node, sample_set):
        sampled_sample_set = []

        for i in range(self.num_resample_iterations):
            for ss in sample_set:  # type: SampleSet
                sampled_ss = SampleSet()
                for n, p in ss.node_dict.items():
                    d = random() * config['max_v']
                    a = uniform(0, 2 * math.pi)
                    new_x = p.x + d * math.cos(a)
                    new_y = p.y + d * math.sin(a)
                    sampled_ss.node_dict[n] = Point(new_x, new_y)
                    sampled_ss.node_dict_previous[n] = p
                sampled_sample_set.append(sampled_ss)

        return sampled_sample_set

    def filtering_step(self, config, node, sample_set, current_global_state_matrix):
        filtered_sample_set = []
        for ss in sample_set:  # type: SampleSet
            is_valid = True
            for n1 in ss.node_dict.keys():
                for n2 in ss.node_dict.keys():
                    if n1 is not n2:
                        if ss.node_dict[n1].distance(ss.node_dict[n2]) > config['communication_radius']:
                            predicted_node_action = STATE_INVISIBLE
                        elif ss.node_dict[n1].distance(ss.node_dict[n2]) > ss.node_dict_previous[n1].distance(
                                ss.node_dict_previous[n2]):
                            predicted_node_action = STATE_RETREATING
                        else:
                            predicted_node_action = STATE_APPROACHING
                        actual_node_action = current_global_state_matrix[n1.index, n2.index]
                        if predicted_node_action != actual_node_action:
                            is_valid = False
            if is_valid:
                filtered_sample_set.append(ss)

        return filtered_sample_set

    def predicting_step(self, sample_set):
        all_predicted_distances = {}
        for s in sample_set:  # type: SampleSet
            for n in s.node_dict.keys():
                if n not in all_predicted_distances:
                    all_predicted_distances[n] = []
                all_predicted_distances[n].append(s.node_dict[n])

        avg_predicted_distances = {}

        for n in all_predicted_distances.keys():
            avg_predicted_distances[n] = np.mean([n.distance(Point(0, 0)) for n in all_predicted_distances[n]])

        return avg_predicted_distances

    def communication(self, nodes, previous_global_state_matrix, current_global_state_matrix):
        self.communication_share_trinary_connectivity_change_to_all_neighbors(nodes, previous_global_state_matrix,
                                                                              current_global_state_matrix, k_hop_neighbors=self.k_hop_neighbors)

    def predict(self, config, nodes, current_global_state_matrix):
        # Predict for all nodes
        for n1 in nodes:  # type: Node
            if n1 not in self.previous_sample_sets:
                self.previous_sample_sets[n1] = []
            self.previous_sample_sets[n1], n1.p_pred[self.name()] = self.monte_carlo(config, self.previous_sample_sets[n1], n1,
                                                                              current_global_state_matrix)

        # Use predicted point to determine predicted distance per neighbor
        for n1 in nodes:  # type: Node
            for n2 in n1.one_hop_neighbors:  # type: Node
                if n2 not in n1.p_pred[self.name()]:
                    n1.one_hop_neighbor_predicted_distances[self.name()][n2] = 0.0
                else:
                    n1.one_hop_neighbor_predicted_distances[self.name()][n2] = n1.p_pred[self.name()][n2]

        return np.mean(np.array([len(self.previous_sample_sets[n]) for n in nodes]))
