import math
from random import uniform, shuffle

from simulator.node import Node
from simulator.point import Point
from st_mcl.main import StMCL


class VA_MCL(StMCL):
    def __init__(self):
        super(VA_MCL, self).__init__()

    def _generate_sample(self, config, node):
        max_x = math.inf
        max_y = math.inf
        min_x = -math.inf
        min_y = -math.inf

        anchors = [n for n in node.one_hop_neighbors if n.is_anchor]
        virtual_anchors = []
        for a1 in anchors:  # type: Node
            for a2 in anchors:  # type: Node
                if a1 is not a2:
                    va_x = ((a1.currentP.x - a2.currentP.x) / 2) + a1.currentP.x
                    va_y = ((a1.currentP.y - a2.currentP.y) / 2) + a1.currentP.y
                    virtual_anchors.append(Point(va_x, va_y))
                    
        if len(virtual_anchors) == 0:
            return super(VA_MCL, self)._generate_sample(config, node)

        for a in virtual_anchors:  # type: Point
            max_x = min(max_x, a.x + config['communication_radius'])
            max_y = min(max_y, a.y + config['communication_radius'])
            min_x = max(min_x, a.x - config['communication_radius'])
            min_y = max(min_y, a.y - config['communication_radius'])

        return Point(uniform(min_x, max_x), uniform(min_y, max_y))

    def communication(self, nodes, previous_global_state_matrix, current_global_state_matrix):
        self.communication_share_binary_connectivity_change_to_all_neighbors(nodes, previous_global_state_matrix, current_global_state_matrix)
        self.communication_share_gps_to_all_1_and_2_hop_neighbors(nodes, previous_global_state_matrix, current_global_state_matrix)
