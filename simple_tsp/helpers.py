#!/usr/bin/env python

"""
    simple_tsp/helpers.py
"""

from numba import njit

@njit(cache=True)
def route2cost(route, dist):
    cost = dist[route[-1], route[0]]
    for i in range(len(route) - 1):
        cost += dist[route[i], route[i + 1]]
    
    return cost