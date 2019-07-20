from math import sqrt


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, p2):
        return sqrt((p2.x - self.x)**2 + (p2.y - self.y)**2)
