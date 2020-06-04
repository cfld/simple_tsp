#!/usr/bin/env python

"""
    simple_tsp/helpers.py
"""

import numpy as np
from numba import njit

@njit(cache=True)
def route2cost(route, dist):
    cost = dist[route[-1], route[0]]
    for i in range(len(route) - 1):
        cost += dist[route[i], route[i + 1]]
    
    return cost

@njit(cache=True)
def numba_set_seed(seed):
    _ = np.random.seed(seed)

def set_seeds(seed):
    _ = np.random.seed(seed)
    _ = numba_set_seed(seed + 111)