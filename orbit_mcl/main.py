from random import shuffle

import numpy as np

from simulator.point import Point
from st_mcl.main import StMCL


class OrbitMCL(StMCL):
    POINT_DISTANCE_THRESHOLD = 25.0

    def __init__(self):
        super(OrbitMCL, self).__init__()
        return

    def filtering_step(self, config, node, sample_set, current_global_state_matrix):
        sample_set = super(OrbitMCL, self).filtering_step(config, node, sample_set, current_global_state_matrix)

        connectivity_graph = np.zeros((len(sample_set), len(sample_set)))
        for i, s1 in enumerate(sample_set):  # type: Point
            for j, s2 in enumerate(sample_set):  # type: Point
                if s1 is not s2:
                    connectivity_graph[i, j] = s1.distance(s2) <= OrbitMCL.POINT_DISTANCE_THRESHOLD

        samples_per_region = []
        samples_removed = []

        largest_set_length = 0

        for s_i in range(len(sample_set)):
            if s_i not in samples_removed:
                samples_removed.append(s_i)
                s = sample_set[s_i]
                new_region_samples = [s]
                for s_j in range(len(sample_set)):
                    if s_j not in samples_removed and connectivity_graph[s_i, s_j]:
                        samples_removed.append(s_j)
                        new_region_samples.append(sample_set[s_j])
                samples_per_region.append(new_region_samples)
                largest_set_length = max(largest_set_length, len(new_region_samples))

        samples_per_region = [s for s in samples_per_region if len(s) > largest_set_length / 3]

        def flatten(l):
            flat_l = []
            for x in l:
                flat_l += x
            return flat_l

        new_sample_set = flatten(samples_per_region)

        return new_sample_set

    def communication(self, nodes, previous_global_state_matrix, current_global_state_matrix):
        self.communication_share_binary_connectivity_change_to_all_neighbors(nodes, previous_global_state_matrix, current_global_state_matrix)
        self.communication_share_gps_to_all_1_and_2_hop_neighbors(nodes, previous_global_state_matrix, current_global_state_matrix)
