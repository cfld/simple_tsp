#!/usr/bin/env python

"""
    simple_tsp/helpers.py
"""

def route2cost(route, dist):
    return dist[(route[:-1], route[1:])].sum() + dist[route[-1], route[0]]
