from binary_mcl.main import BinaryMCL


class BinaryNoMemMCL(BinaryMCL):
    def __init__(self):
        super(BinaryNoMemMCL, self).__init__()
        return

    def name(self):
        return "binary_no_mem"

    def monte_carlo(self, config, sample_set, node, current_global_state_matrix):
        sample_set = self.initialization_step(config, node)
        sample_set = self.filtering_step(config, node, sample_set, current_global_state_matrix)
        return sample_set, self.predicting_step(sample_set)
