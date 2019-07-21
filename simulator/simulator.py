import numpy as np

from base_mcl_algorithm.base_mcl import BaseMCL
from simulator.node import Node
from trinary_mcl.main import TrinaryMCL
from st_mcl.main import StMCL
from va_mcl.main import VA_MCL
from orbit_mcl.main import OrbitMCL
from lcc_mcl.main import LCC_MCL

STATE_INVISIBLE = 0
STATE_APPROACHING = 1
STATE_RETREATING = 2


class Simulator:
    algorithms = [
        TrinaryMCL(),
        StMCL(),
        VA_MCL(),
        OrbitMCL(),
        LCC_MCL(),
    ]

    nodes = []

    def __init__(self):
        self.current_global_state_matrix = None
        self.previous_global_state_matrix = None

    def update_global_state_matrix(self, nodes, communication_radius):
        if self.current_global_state_matrix is not None:
            self.previous_global_state_matrix = self.current_global_state_matrix
        self.current_global_state_matrix = np.ndarray((len(nodes), len(nodes)))

        for i, n1 in enumerate(nodes):
            for j, n2 in enumerate(nodes):
                if n1 == n2:
                    self.current_global_state_matrix[i, j] = STATE_INVISIBLE
                elif n1.distance(n2) > communication_radius:
                    self.current_global_state_matrix[i, j] = STATE_INVISIBLE
                elif n1.distance(n2) > n1.distance_previously(n2):
                    self.current_global_state_matrix[i, j] = STATE_RETREATING
                else:
                    self.current_global_state_matrix[i, j] = STATE_APPROACHING

    def update_one_hop_neighbors_lists(self, nodes):
        for i, n1 in enumerate(nodes):
            n1.one_hop_neighbors = []
            for j, n2 in enumerate(nodes):
                if self.current_global_state_matrix[i, j] > 0:
                    n1.one_hop_neighbors.append(n2)

    def update_two_hop_neighbors_lists(self, nodes):
        for i, n1 in enumerate(nodes):
            n1.two_hop_neighbors = []

            for n2 in n1.one_hop_neighbors:  # type: Node
                for n3 in n2.one_hop_neighbors:
                    if n3 not in n1.one_hop_neighbors and n3 not in n1.two_hop_neighbors:
                        n1.two_hop_neighbors.append(n2)

    def run(self, num_time_instances=100, num_nodes=100, num_anchors=25, stage_size=(500, 500), max_v=50,
            communication_radius=50):
        algorithm_results = {
            'number_of_packets': {},
            'accuracy': {},
        }
        simulator_results = {
            'node_positions': []
        }

        config = {
            'max_v': max_v,
            'communication_radius': communication_radius,
        }

        # Initialize Results Dictionaries
        for a in self.algorithms:
            for k in algorithm_results.keys():
                algorithm_results[k][a] = []
        for i in range(num_nodes):
            simulator_results['node_positions'].append([])

        # Initialize Nodes
        for i in range(num_nodes):
            self.nodes.append(Node(i, i < num_anchors, stage_size[0], stage_size[1], max_v, communication_radius))

        for t in range(num_time_instances):
            # Move each node
            for n in self.nodes:  # type: Node
                n.move()

            # Build state matrix and neighbor lists
            self.update_global_state_matrix(self.nodes, communication_radius)
            self.update_one_hop_neighbors_lists(self.nodes)
            self.update_two_hop_neighbors_lists(self.nodes)

            # Log node positions
            for i in range(len(self.nodes)):
                simulator_results['node_positions'][i].append(self.nodes[i].currentP)
            # Simulate Communication
            for a in self.algorithms:  # type: BaseMCL
                a.communication(self.nodes, self.previous_global_state_matrix, self.current_global_state_matrix)
                algorithm_results['number_of_packets'][a].append(a.get_total_number_of_packets())

            # Make predictions for all nodes
            for a in self.algorithms:
                for n in self.nodes:
                    n.one_hop_neighbor_predicted_distances[a] = {}
                a.predict(config, self.nodes)

            # Evaluate Predictions
            for a in self.algorithms:
                total_distance_error = 0.0
                count = 0
                for i, n1 in enumerate(self.nodes):  # type: Node
                    if not n1.is_anchor:
                        for j, n2 in enumerate(self.nodes): # type: Node
                            if self.current_global_state_matrix[i,j] > 0:
                                distance_predicted = n1.one_hop_neighbor_predicted_distances[a][n2] if n2 in n1.one_hop_neighbor_predicted_distances[a] else 0.0
                                distance_actual = n1.distance(n2)
                                total_distance_error += abs(distance_actual - distance_predicted)
                                count += 1
                algorithm_results['accuracy'][a].append(total_distance_error / count if count > 0 else 0.0)

        return simulator_results, algorithm_results

