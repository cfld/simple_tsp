#!/usr/bin/env python

"""
    simple_tsp/perturb.py
"""

import numpy as np

def kick(route):
    """ Random double-bridge move """
    
    route   = route[:-1]
    n_nodes = len(route)
    
    cut   = np.random.choice(n_nodes, 4, replace=False) 
    cut   = np.sort(cut)
    
    zero  = route[:cut[0]]
    one   = route[cut[0]:cut[1]]
    two   = route[cut[1]:cut[2]]
    three = route[cut[2]:cut[3]]
    four  = route[cut[3]:]
    
    new_route = np.hstack([zero, three, two, one, four])
    return np.hstack([new_route, [0]])