import math
import random
import time

import numpy as np

from base_mcl_algorithm.base_mcl import BaseMCL
from binary_mcl.main import BinaryMCL
from binary_no_mem_mcl.main import BinaryNoMemMCL
from orbit_mcl.main import OrbitMCL
from simulator.node import Node
from simulator.point import Point
from st_mcl.main import StMCL
from trinary_mcl.main import TrinaryMCL
from trinary_mcl.sample_set import SampleSet
from va_mcl.main import VA_MCL
from lcc_mcl.main import LCC_MCL

STATE_INVISIBLE = 0
STATE_APPROACHING = 1
STATE_RETREATING = 2


class Simulator:
    algorithms = [
        BinaryNoMemMCL(),
        BinaryMCL(),
        TrinaryMCL(),
        StMCL(),
        VA_MCL(),
        OrbitMCL(),  # From experiments, orbit works best when there are different sized communication radii
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

        for i, n1 in enumerate(nodes):  # type: int, Node
            for j, n2 in enumerate(nodes):  # type: int, Node
                if n1 == n2:
                    self.current_global_state_matrix[i, j] = STATE_INVISIBLE
                elif n1.distance(n2) > communication_radius:
                    self.current_global_state_matrix[i, j] = STATE_INVISIBLE
                elif n1.previousP is not None and n1.distance(n2) > n1.distance_previously(n2):
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
                    if n3 is not n1 and n3 not in n1.one_hop_neighbors and n3 not in n1.two_hop_neighbors:
                        n1.two_hop_neighbors.append(n3)

    def run(self, num_time_instances=100, num_nodes=100, num_anchors=25, stage_size=(500, 500), max_v=50,
            communication_radius=50):
        algorithm_results = {
            'number_of_packets': {},
            'distance_error': {},
            'position_error': {},
            'prediction_time': {},
            'number_of_samples': {},
        }
        simulator_results = {
            'node_positions': [],
            'avg_number_of_first_hop_neighbors': [],
            'avg_number_of_second_hop_neighbors': [],
        }

        config = {
            'max_x': stage_size[0],
            'max_y': stage_size[1],
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
        for i in range(num_nodes):  # type: int
            self.nodes.append(Node(i, i < num_anchors, config=config))

        for t in range(num_time_instances):

            # Move each node
            for n in self.nodes:  # type: Node
                n.move()

            # Build state matrix and neighbor lists
            self.update_global_state_matrix(self.nodes, communication_radius)
            self.update_one_hop_neighbors_lists(self.nodes)
            self.update_two_hop_neighbors_lists(self.nodes)

            simulator_results['avg_number_of_first_hop_neighbors'].append(sum([len(n.one_hop_neighbors) for n in self.nodes]) / len(self.nodes))
            simulator_results['avg_number_of_second_hop_neighbors'].append(sum([len(n.two_hop_neighbors) for n in self.nodes]) / len(self.nodes))

            # Log node positions
            for i in range(len(self.nodes)):
                simulator_results['node_positions'][i].append(self.nodes[i].currentP)
            # Simulate Communication
            for a in self.algorithms:  # type: BaseMCL
                a.communication(self.nodes, self.previous_global_state_matrix, self.current_global_state_matrix)
                algorithm_results['number_of_packets'][a].append(a.get_total_number_of_packets())

            # Make predictions for all nodes
            for a in self.algorithms:
                start_time = time.time()
                for n in self.nodes:
                    n.one_hop_neighbor_predicted_distances[a] = {}
                random.seed(t)
                algorithm_results['number_of_samples'][a].append(a.predict(config, self.nodes, self.current_global_state_matrix))
                end_time = time.time()
                algorithm_results['prediction_time'][a].append(end_time - start_time)

            # Evaluate Distance Error (Trinary)
            for a in self.algorithms:
                total_distance_error = 0.0
                count = 0
                for i, n1 in enumerate(self.nodes):  # type: Node
                    # Anchors never add to the distance error
                    if not n1.is_anchor:
                        # for n2 in n1.one_hop_neighbors + n1.two_hop_neighbors:
                        for j, n2 in enumerate(self.nodes):  # type: Node
                            if self.current_global_state_matrix[i,j] > 0:
                                distance_predicted = n1.one_hop_neighbor_predicted_distances[a][n2] if n2 in n1.one_hop_neighbor_predicted_distances[a] else 0.0
                                distance_actual = n1.distance(n2)
                                if math.isnan(distance_predicted):
                                    distance_predicted = 0.0
                                total_distance_error += abs(distance_actual - distance_predicted)
                                count += 1


                algorithm_results['distance_error'][a].append(total_distance_error / count if count > 0 else 0.0)

            # Evaluate Position Error (non-Trinary)
            for a in self.algorithms:
                if not isinstance(a, TrinaryMCL):
                    total_error = 0.0
                    for i, n1 in enumerate(self.nodes):  # type: Node
                        if not math.isnan(n1.p_pred[a].x):
                            total_error += n1.currentP.distance(n1.p_pred[a])
                    algorithm_results['position_error'][a].append(total_error)

        return simulator_results, algorithm_results

