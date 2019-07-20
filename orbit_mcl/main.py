from base_mcl_algorithm.base_mcl import BaseMCL

class OrbitMCL(BaseMCL):
    def __init__(self):
        super(OrbitMCL, self).__init__()
        return

    def initialization_step(self):
        pass

    def sampling_step(self):
        pass

    def filtering_step(self):
        pass

    def communication(self, nodes, previous_global_state_matrix, current_global_state_matrix):
        self.communication_share_binary_connectivity_change_to_all_neighbors(nodes, previous_global_state_matrix, current_global_state_matrix)
        self.communication_share_gps_to_all_1_and_2_hop_neighbors(nodes, previous_global_state_matrix, current_global_state_matrix)


    def predict(self, node):
        pass
