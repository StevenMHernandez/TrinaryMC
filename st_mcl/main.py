import math
from random import random, uniform
from random import shuffle

import numpy as np

from base_mcl_algorithm.base_mcl import BaseMCL
from simulator.node import Node
from simulator.point import Point


class StMCL(BaseMCL):

    def __init__(self):
        super(StMCL, self).__init__()

        self.sample_threshold = 100
        self.max_initial_sample_iterations = 300
        self.num_resample_iterations = 10
        self.previous_sample_sets = {}

    def monte_carlo(self, config, sample_set, node, current_global_state_matrix):
        if len(sample_set) == 0:
            sample_set = self.initialization_step(config, node)
        sample_set = self.sampling_step(config, node, sample_set)
        if len(sample_set) > self.sample_threshold:
            shuffle(sample_set)
            sample_set = sample_set[0:self.sample_threshold]
        sample_set = self.filtering_step(config, node, sample_set, current_global_state_matrix)
        return sample_set, self.predicting_step(sample_set)

    def _generate_sample(self, config, node):
        max_x = math.inf
        max_y = math.inf
        min_x = -math.inf
        min_y = -math.inf
        for a in [n for n in node.one_hop_neighbors if n.is_anchor]:  # type: Node
            max_x = min(max_x, a.currentP.x + config['communication_radius'])
            max_y = min(max_y, a.currentP.y + config['communication_radius'])
            min_x = max(min_x, a.currentP.x - config['communication_radius'])
            min_y = max(min_y, a.currentP.y - config['communication_radius'])

        return Point(uniform(min_x, max_x), uniform(min_y, max_y))

    def initialization_step(self, config, node):
        sample_set = []
        for i in range(self.max_initial_sample_iterations):
            sample = self._generate_sample(config, node)
            if not math.isnan(sample.x):
                sample_set.append(sample)

        return sample_set

    def sampling_step(self, config, node, sample_set):
        sampled_sample_set = []

        for i in range(self.num_resample_iterations):
            for p in sample_set:
                d = random() * config['max_v']
                a = uniform(0, 2 * math.pi)
                new_x = p.x + d * math.cos(a)
                new_y = p.y + d * math.sin(a)
                sampled_sample_set.append(Point(new_x, new_y))

        return sampled_sample_set

    def filtering_step(self, config, node, sample_set, current_global_state_matrix):
        filtered_sample_set = []
        for p in sample_set:
            is_valid = True
            for a in [n for n in node.one_hop_neighbors if n.is_anchor]:  # type: Node
                if a.currentP.distance(p) > config['communication_radius']:
                    is_valid = False

            for a in [n for n in node.two_hop_neighbors if n.is_anchor]:  # type: Node
                if a.currentP.distance(p) <= config['communication_radius']:
                    is_valid = False
            if is_valid:
                filtered_sample_set.append(p)

        return filtered_sample_set

    def predicting_step(self, sample_set):
        if len(sample_set) == 0:
            return Point(0,0)

        x_pred = sum([p.x for p in sample_set]) / len(sample_set)
        y_pred = sum([p.y for p in sample_set]) / len(sample_set)

        return Point(x_pred, y_pred)

    def communication(self, nodes, previous_global_state_matrix, current_global_state_matrix):
        self.communication_share_gps_to_all_1_and_2_hop_neighbors(nodes, previous_global_state_matrix, current_global_state_matrix)

    def predict(self, config, nodes, current_global_state_matrix):
        # Predict point location for all nodes
        for n1 in nodes:  # type: Node
            if n1 not in self.previous_sample_sets:
                self.previous_sample_sets[n1] = []
            self.previous_sample_sets[n1], n1.p_pred[self] = self.monte_carlo(config, self.previous_sample_sets[n1], n1, current_global_state_matrix)

        # Use predicted point to determine predicted distance per neighbor
        for n1 in nodes:  # type: Node
            for n2 in n1.one_hop_neighbors:  # type: Node
                p_pred = n1.p_pred[self]
                if n2.is_anchor:
                    n1.one_hop_neighbor_predicted_distances[self][n2] = n2.currentP.distance(p_pred)
                elif self in n2.p_pred:
                    n1.one_hop_neighbor_predicted_distances[self][n2] = n2.p_pred[self].distance(p_pred)

        return np.mean(np.array([len(self.previous_sample_sets[n]) for n in nodes]))
