from mcl_algorithms.trinary_mcl.main import TrinaryMCL
from mcl_algorithms.trinary_mcl.sample_set import SampleSet

STATE_INVISIBLE = 0
STATE_VISIBLE = 1


class BinaryMCL(TrinaryMCL):
    def __init__(self):
        super(BinaryMCL, self).__init__()
        return

    def name(self):
        return "binary"

    def filtering_step(self, config, node, sample_set, current_global_state_matrix):
        filtered_sample_set = []
        for ss in sample_set:  # type: SampleSet
            is_valid = True
            for n1 in ss.node_dict.keys():
                for n2 in ss.node_dict.keys():
                    if n1 is not n2:
                        if ss.node_dict[n1].distance(ss.node_dict[n2]) > config['communication_radius']:
                            predicted_node_action = STATE_INVISIBLE
                        else:
                            predicted_node_action = STATE_VISIBLE
                        actual_node_action = STATE_VISIBLE if current_global_state_matrix[n1.index, n2.index] > 0 else STATE_INVISIBLE
                        if predicted_node_action != actual_node_action:
                            is_valid = False
            if is_valid:
                filtered_sample_set.append(ss)

        return filtered_sample_set

    def communication(self, nodes, previous_global_state_matrix, current_global_state_matrix):
        self.communication_share_binary_connectivity_change_to_all_neighbors(nodes, previous_global_state_matrix, current_global_state_matrix)
