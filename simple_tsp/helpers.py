#!/usr/bin/env python

"""
    simple_tsp/helpers.py
"""

def route2cost(route, dist):
    assert route[0] == 0
    assert route[-1] == 0
    
    return dist[(route[:-1], route[1:])].sum()
