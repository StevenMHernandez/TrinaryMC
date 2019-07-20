from base_mcl_algorithm.base_mcl import BaseMCL


class TrinaryMCL(BaseMCL):
    def __init__(self):
        super(TrinaryMCL, self).__init__()
        return

    def initialization_step(self):
        pass

    def sampling_step(self):
        pass

    def filtering_step(self):
        pass

    def communication(self, nodes, previous_global_state_matrix, current_global_state_matrix):
        self.communication_share_trinary_connectivity_change_to_all_neighbors(nodes, previous_global_state_matrix, current_global_state_matrix)

    def predict(self, node):
        pass

    def _trinary_state_changed(self, n1, n2, previous_global_state_matrix, current_global_state_matrix):
        return previous_global_state_matrix is None or \
               previous_global_state_matrix[n1.index, n2.index] != current_global_state_matrix[n1.index, n2.index]
