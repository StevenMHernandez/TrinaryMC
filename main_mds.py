from time import time

from simulator.simulator_single_step_mds import SimulatorSingleStepMDS

if __name__ == "__main__":
    start = time()

    num_nodes = 11
    max_v = 10

    errors = []
    for i in range(50):
        packet = SimulatorSingleStepMDS().run(max_v=max_v, num_nodes=num_nodes, should_plot=False)
        errors.append(str(packet['error']))

    print(errors)
    print("Took", time() - start, "seconds")
    print("num_nodes: ", num_nodes)
    print("max_v: ", max_v)
    print("final errors: ", "\t".join(errors))

