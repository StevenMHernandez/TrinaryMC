from time import time

from simulator.simulator_single_step import SimulatorSingleStep

if __name__ == "__main__":
    start = time()

    SimulatorSingleStep().run()

    print("Took", time() - start, "seconds")
