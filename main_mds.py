from time import time

from simulator.simulator_single_step_mds import SimulatorSingleStepMDS

if __name__ == "__main__":
    start = time()

    SimulatorSingleStepMDS().run()

    print("Took", time() - start, "seconds")
