import math
from random import uniform

from simulator.node import Node
from simulator.point import Point
from st_mcl.main import StMCL


class LCC_MCL(StMCL):
    def __init__(self):
        super(LCC_MCL, self).__init__()

    def _generate_sample(self, config, node):
        max_x = math.inf
        max_y = math.inf
        min_x = -math.inf
        min_y = -math.inf
        for n1 in node.one_hop_neighbors:  # type: Node
            if n1.is_anchor:
                max_x = min(max_x, n1.currentP.x + config['communication_radius'])
                max_y = min(max_y, n1.currentP.y + config['communication_radius'])
                min_x = max(min_x, n1.currentP.x - config['communication_radius'])
                min_y = max(min_y, n1.currentP.y - config['communication_radius'])
            elif self in n1.p_pred and self._are_lcc_close(node, n1):
                max_x = min(max_x, n1.p_pred[self].x + config['communication_radius'])
                max_y = min(max_y, n1.p_pred[self].y + config['communication_radius'])
                min_x = max(min_x, n1.p_pred[self].x - config['communication_radius'])
                min_y = max(min_y, n1.p_pred[self].y - config['communication_radius'])

        return Point(uniform(min_x, max_x), uniform(min_y, max_y))

    def filtering_step(self, config, node, sample_set, current_global_state_matrix):
        filtered_sample_set = []
        for p in sample_set:
            is_valid = True
            for n1 in node.one_hop_neighbors:  # type: Node
                if n1.currentP.distance(p) > config['communication_radius']:
                    if n1.is_anchor or (self in n1.p_pred and self._are_lcc_close(node, n1)):
                        is_valid = False
            for n1 in node.two_hop_neighbors:  # type: Node
                if n1.currentP.distance(p) <= config['communication_radius']:
                    if n1.is_anchor or (self in n1.p_pred and self._are_lcc_close(node, n1)):
                        is_valid = False
            if is_valid:
                filtered_sample_set.append(p)

        return filtered_sample_set

    def communication(self, nodes, previous_global_state_matrix, current_global_state_matrix):
        self.communication_share_binary_connectivity_change_to_all_neighbors(nodes, previous_global_state_matrix, current_global_state_matrix)

        # Share GPS only to interested Nodes only
        for n1 in nodes:
            for n2 in n1.one_hop_neighbors:
                if self._are_lcc_close(n1, n2):
                    self.add_packet_communication('share_gps_change_to_first_hop_neighbors')
                    for n3 in n2.one_hop_neighbors:
                        if self._are_lcc_close(n2, n3):
                            self.add_packet_communication('share_gps_change_to_second_hop_neighbors')

    def _are_lcc_close(self, n1, n2):
        num_intersection = []
        one_hop_neighbors_set = set(n1.one_hop_neighbors)
        for n3 in n1.one_hop_neighbors:  # type: Node
            num_intersection.append(len(one_hop_neighbors_set.intersection(n3.one_hop_neighbors)))

        average_num_intersecting = sum(num_intersection) / len(num_intersection)

        return len(one_hop_neighbors_set.intersection(n2.one_hop_neighbors)) > average_num_intersecting
