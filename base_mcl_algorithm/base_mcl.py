class BaseMCL(object):
    def __init__(self):
        self.packets_dict = {}

    def add_packet_communication(self, packet_type, packet_count=1):
        if packet_type not in self.packets_dict:
            self.packets_dict[packet_type] = 0

        self.packets_dict[packet_type] += packet_count

    def get_total_number_of_packets(self):
        return sum(self.packets_dict.values())

    def communication(self, nodes, previous_global_state_matrix, current_global_state_matrix):
        pass

    def communication_share_binary_connectivity_change_to_all_neighbors(self, nodes, previous_global_state_matrix, current_global_state_matrix):
        one_hop_neighbors_which_received_update = []
        for i, n1 in enumerate(nodes):
            state_changed_between_any_neighbor = False

            for j, n2 in enumerate(nodes):
                if previous_global_state_matrix is None or \
                        previous_global_state_matrix[i, j] > 0 != current_global_state_matrix[i, j] > 0:
                    state_changed_between_any_neighbor = True

            if state_changed_between_any_neighbor:
                for n2 in n1.one_hop_neighbors:
                    self.add_packet_communication('share_state_change_to_first_hop')
                    one_hop_neighbors_which_received_update.append(n2)
        for _ in set(one_hop_neighbors_which_received_update):
            self.add_packet_communication('share_state_change_second_hop')

    def communication_share_trinary_connectivity_change_to_all_neighbors(self, nodes, previous_global_state_matrix, current_global_state_matrix):
        one_hop_neighbors_which_received_update = []
        for i, n1 in enumerate(nodes):
            state_changed_between_any_neighbor = False

            for j, n2 in enumerate(nodes):
                if previous_global_state_matrix is None or \
                        previous_global_state_matrix[i, j] != current_global_state_matrix[i, j]:
                    state_changed_between_any_neighbor = True

            if state_changed_between_any_neighbor:
                for n2 in n1.one_hop_neighbors:
                    self.add_packet_communication('share_state_change_first_hop')
                    one_hop_neighbors_which_received_update.append(n2)
        for _ in set(one_hop_neighbors_which_received_update):
            self.add_packet_communication('share_state_change_second_hop')

    def communication_share_gps_to_all_1_and_2_hop_neighbors(self, nodes, previous_global_state_matrix, current_global_state_matrix):
        for n1 in [n for n in nodes if n.is_anchor]:
            for n2 in n1.one_hop_neighbors:
                self.add_packet_communication('share_gps_change_to_first_hop_neighbors')
                for _ in n2.one_hop_neighbors:
                    self.add_packet_communication('share_gps_change_to_second_hop_neighbors')

    def predict(self, config, node, current_global_state_matrix):
        pass
