#!/usr/bin/env python

"""
    simple_tsp/helpers.py
"""

import numpy as np
from numba import njit

# --
# Random seeds

@njit(cache=True)
def numba_set_seed(seed):
    _ = np.random.seed(seed)

def set_seeds(seed):
    _ = np.random.seed(seed)
    _ = numba_set_seed(seed + 111)

# --
# Compute cost

@njit(cache=True)
def route2cost(route, dist):
    cost = dist[route[-1], route[0]]
    for i in range(len(route) - 1):
        cost += dist[route[i], route[i + 1]]
    
    return cost

# --

@njit(cache=True)
def suc2cost(node2suc, dist, n_vehicles):
    cost = 0
    counter = 0
    for depot in range(n_vehicles):
        n = depot
        while True:
            cost += dist[n, node2suc[n]]
            n = node2suc[n]
            if n == depot: break
            counter += 1
            if counter > node2suc.shape[0]: raise Exception('!! loop detected')
    
    return cost

@njit(cache=True)
def walk_suc(depot, node2suc):
    route = []
    node = depot
    while True:
        route.append(node)
        node = node2suc[node]
        if node == depot: break
    
    return np.array(route)


def walk_sucs(node2suc, n_nodes, n_vehicles, verbose=False):
    if verbose: print('-' * 50)
    routes = []
    counter = 0
    for depot in range(n_vehicles):
        route = []
        node = depot
        while True:
            if verbose: print(node)
            
            route.append(node)
            counter += 1
            
            node = node2suc[node]
            if node == depot: break
            
            if counter > len(node2suc) + 2: raise Exception('!! loop detected')
        
        if verbose: print('-' * 10)
        routes.append(route)
    
    if counter != n_nodes: 
        raise Exception('!! invalid route')
        
    return routes