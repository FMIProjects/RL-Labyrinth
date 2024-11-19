
from math import sqrt

def euclidean_distance(point1, point2):
    return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def manhattan_distance(point1,point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])