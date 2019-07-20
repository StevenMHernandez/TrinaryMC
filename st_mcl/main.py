from base_mcl_algorithm.base_mcl import BaseMCL
from simulator.node import Node


class StMCL(BaseMCL):
    def __init__(self):
        super(StMCL, self).__init__()

    def initialization_step(self):
        pass

    def sampling_step(self):
        pass

    def filtering_step(self):
        pass

    def communication(self, nodes, previous_global_state_matrix, current_global_state_matrix):
        # Share GPS from anchor to all 1-hop and 2-hop neighbors
        for n1 in [n for n in nodes if n.is_anchor]:
            for n2 in n1.one_hop_neighbors:
                self.add_packet_communication('share_gps_change_to_first_hop_neighbors')
                for _ in n2.one_hop_neighbors:
                    self.add_packet_communication('share_gps_change_to_second_hop_neighbors')

    def predict(self, node):
        pass