#!/usr/bin/env python

"""
    main.py
"""

import os
import json
import argparse
import numpy as np
from time import time
from numba import njit

from simple_tsp.prep import load_problem, load_solution, get_distance_matrix
from simple_tsp.prep import knn_candidates, random_init
from simple_tsp.helpers import route2cost, set_seeds

from simple_tsp.cam import cam

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath',       type=str, default='data/cvrp/INSTANCES/Uchoa/X-n101-k25.vrp')
    parser.add_argument('--n-cands',      type=int, default=10)
    parser.add_argument('--n-kick-iters', type=int, default=100)
    parser.add_argument('--max-depth',    type=int, default=3)
    parser.add_argument('--seed',         type=int, default=123)
    return parser.parse_args()

args = parse_args()

_ = set_seeds(args.seed)

# --
# Load problem

prob = load_problem(args.inpath)

if prob['TYPE'] == 'CVRP':
    n_vehicles = prob['VEHICLES']
    
    cap        = prob['CAPACITY']
    
    demand = np.array(list(prob['DEMAND_SECTION'].values()))
    demand = np.hstack([
        np.repeat(-1, n_vehicles),
        demand[1:]
    ])
else:
    n_vehicles = 1

print('n_vehicles', n_vehicles)

dist = get_distance_matrix(prob, n_vehicles=n_vehicles)
near = knn_candidates(dist, args.n_cands)

n_nodes = dist.shape[0]

# --
# Initialize route

route = random_init(n_nodes, random_state=101)

# --

@njit(cache=True)
def edge_exists(u, v, n):
    return (
        (u == v + 1) or 
        (v == u + 1) or 
        (u == 0 and v == n - 1) or
        (v == 0 and u == n - 1)
    )

@njit
def cam(p00, p01, near, dist, rs, n_nodes, max_depth, dlb, demand, cap, counter):    
    n00  = rs.pos2node[p00]
    n01  = rs.pos2node[p01]
    
    fwd0 = p01 > p00
    if fwd0:
        r0 = rs.pos2route[p00] # edge belongs to same route as earlier node
    else:
        r0 = rs.pos2route[p01]
    
    for p10 in rs.node2pos[near[n01]]:
        if edge_exists(p01, p10, n_nodes): continue
        
        n10 = rs.pos2node[p10]
        
        for fwd1 in [True, False]:
            
            if fwd1:
                p11 = p10 + 1
                r1  = rs.pos2route[p10]
            else:
                p11 = p10 - 1
                r1  = rs.pos2route[p11]
            
            if r0 == r1: continue   # Edges can't be in same route
            
            n11 = rs.pos2node[p11]
            if n00 == n11: continue # skip for now -- though this is actually valid move (cat two routes)
            
            n_nodes = len(set([n00, n01, n10, n11])) # check 4 distinct nodes
            assert n_nodes == 4
            
            sav = (
                + dist[n00, n01]
                + dist[n10, n11]
                - dist[n01, n10]
                - dist[n11, n00]
            )
                        
            if sav > 0:
                # execute move
                
        