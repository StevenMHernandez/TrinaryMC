from base_mcl_algorithm.base_mcl import BaseMCL
from simulator.node import Node


class LCC_MCL(BaseMCL):
    def __init__(self):
        super(LCC_MCL, self).__init__()

    def initialization_step(self):
        pass

    def sampling_step(self):
        pass

    def filtering_step(self):
        pass

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

    def predict(self, node):
        pass

    def _are_lcc_close(self, n1, n2):
        num_intersection = []
        one_hop_neighbors_set = set(n1.one_hop_neighbors)
        for n3 in n1.one_hop_neighbors:  # type: Node
            num_intersection.append(len(one_hop_neighbors_set.intersection(n3.one_hop_neighbors)))

        average_num_intersecting = sum(num_intersection) / len(num_intersection)

        return len(one_hop_neighbors_set.intersection(n2.one_hop_neighbors)) > average_num_intersecting
