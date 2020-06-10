#!/usr/bin/env python

"""
    simple_tsp/perturb.py
"""

import numpy as np
from numba import njit

# @njit(cache=True)
def double_bridge_kick(route):
    """ Random double-bridge move """
    
    n_nodes = len(route)
    
    cut   = 1 + np.random.choice(n_nodes - 1, 4, replace=False) 
    cut   = np.sort(cut)
    
    zero  = route[:cut[0]]
    one   = route[cut[0]:cut[1]]
    two   = route[cut[1]:cut[2]]
    three = route[cut[2]:cut[3]]
    four  = route[cut[3]:]
    
    return np.hstack((zero, three, two, one, four))


# @njit(cache=True)
def double_bridge_kick_targeted(route, dist, node2suc, n_vehicles):
    """ Random double-bridge move, sampling weighted by edge length """
    
    # compute edge lengths
    lengths = np.zeros(len(node2suc), dtype=np.int64)
    offset = 0
    for depot in range(n_vehicles):
        n = depot
        while True:
            lengths[offset] = dist[n, node2suc[n]]
            offset += 1
            
            n = node2suc[n]
            if n == depot: break
    
    # do bridge kick
    n_nodes = len(route)
    
    cut   = 1 + np.random.choice(n_nodes - 1, 4, replace=False, p=lengths[1:] / lengths[1:].sum()) 
    cut   = np.sort(cut)
    
    zero  = route[:cut[0]]
    one   = route[cut[0]:cut[1]]
    two   = route[cut[1]:cut[2]]
    three = route[cut[2]:cut[3]]
    four  = route[cut[3]:]
    
    return np.hstack((zero, three, two, one, four))

def double_bridge_kick_weighted(route, weights):
    # do bridge kick
    n_nodes = len(route)
    
    cut   = 1 + np.random.choice(n_nodes - 1, 4, replace=False, p=weights[1:] / weights[1:].sum()) 
    cut   = np.sort(cut)
    
    zero  = route[:cut[0]]
    one   = route[cut[0]:cut[1]]
    two   = route[cut[1]:cut[2]]
    three = route[cut[2]:cut[3]]
    four  = route[cut[3]:]
    
    return np.hstack((zero, three, two, one, four))
    