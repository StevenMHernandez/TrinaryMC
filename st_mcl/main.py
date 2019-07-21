import math
from random import random, uniform
from random import shuffle

from base_mcl_algorithm.base_mcl import BaseMCL
from simulator.node import Node
from simulator.point import Point


class StMCL(BaseMCL):
    sample_threshold = 100
    max_sample_iterations = 100
    num_resample_iterations = 10
    previous_sample_sets = {}

    def __init__(self):
        super(StMCL, self).__init__()

    @staticmethod
    def monte_carlo(config, sample_set, node):
        if len(sample_set) == 0:
            sample_set = StMCL.initialization_step(config, node)
        sample_set = StMCL.sampling_step(config, node, sample_set)
        sample_set = StMCL.filtering_step(config, node, sample_set)
        if len(sample_set) > StMCL.sample_threshold:
            shuffle(sample_set)
            sample_set = sample_set[0:StMCL.sample_threshold]
        return sample_set, StMCL.predicting_step(sample_set)

    @staticmethod
    def _generate_sample(config, node):
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

    @staticmethod
    def initialization_step(config, node):
        sample_set = []
        for i in range(StMCL.max_sample_iterations):
            sample_set.append(StMCL._generate_sample(config, node))

        return sample_set

    @staticmethod
    def sampling_step(config, node, sample_set):
        sampled_sample_set = []

        for i in range(StMCL.num_resample_iterations):
            for p in sample_set:
                d = random() * config['max_v']
                a = uniform(0, 2 * math.pi)
                new_x = p.x + d * math.cos(a)
                new_y = p.y + d * math.sin(a)
                sampled_sample_set.append(Point(new_x, new_y))

        return sampled_sample_set

    @staticmethod
    def filtering_step(config, node, sample_set):
        filtered_sample_set = []
        for p in sample_set:
            is_valid = True
            for a in [n for n in node.one_hop_neighbors if n.is_anchor]:  # type: Node
                if a.currentP.distance(p) > config['communication_radius']:
                    is_valid = False
            if is_valid:
                filtered_sample_set.append(p)

        return filtered_sample_set

    @staticmethod
    def predicting_step(sample_set):
        if len(sample_set) == 0:
            return Point(0,0)

        x_pred = sum([p.x for p in sample_set]) / len(sample_set)
        y_pred = sum([p.y for p in sample_set]) / len(sample_set)

        return Point(x_pred, y_pred)

    def communication(self, nodes, previous_global_state_matrix, current_global_state_matrix):
        self.communication_share_gps_to_all_1_and_2_hop_neighbors(nodes, previous_global_state_matrix, current_global_state_matrix)

    def predict(self, config, nodes):
        for n1 in nodes:  # type: Node
            if len([n for n in n1.one_hop_neighbors if n.is_anchor]) > 0:
                if n1 not in self.previous_sample_sets:
                    self.previous_sample_sets[n1] = []
                sample_set, p_pred = self.monte_carlo(config, self.previous_sample_sets[n1], n1)
                self.previous_sample_sets[n1] = sample_set

                for n2 in n1.one_hop_neighbors:  # type: Node
                    if n2.is_anchor:
                        n1.one_hop_neighbor_predicted_distances[self][n2] = n2.currentP.distance(p_pred)
