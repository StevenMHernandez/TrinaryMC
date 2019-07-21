from __future__ import annotations

from random import random

from simulator.point import Point


class Node:
    def __init__(self, index, is_anchor, max_x, max_y, max_v, communication_radius):
        self.index = index
        self.is_anchor = is_anchor
        self.max_x = max_x
        self.max_y = max_y
        self.max_v = max_v
        self.communication_radius = communication_radius
        self.currentP = Point(random() * max_x, random() * max_y)
        self.previousP = None
        self.destination = None
        self.one_hop_neighbors = []
        self.two_hop_neighbors = []
        self.one_hop_neighbor_predicted_distances = {}
        self.p_pred = {}

    def distance(self, n2: Node):
        return self.currentP.distance(n2.currentP)

    def distance_previously(self, n2: Node):
        return self.previousP.distance(n2.previousP)

    def move(self):
        if self.currentP is not None:
            self.previousP = Point(self.currentP.x, self.currentP.y)

        if self.destination is None:
            self.destination = Point(random() * self.max_x, random() * self.max_y)

        distance = self.currentP.distance(self.destination)

        distance_x = self.destination.x - self.currentP.x
        distance_y = self.destination.y - self.currentP.y
        v = random() * (self.max_v - 1) + 1

        if v < distance:
            x = self.currentP.x + (v * distance_x / distance)
            y = self.currentP.y + (v * distance_y / distance)
            self.currentP = Point(x,y)
        else:
            self.currentP = Point(self.destination.x,self.destination.y)
            self.destination = None

    def is_one_hop_neighbor(self, n2):
        return n2 in self.one_hop_neighbors
