#!/usr/bin/env python

"""
    main.py
"""

import os
import json
import argparse
import numpy as np
from time import time

import numba
from numba import njit, prange

from simple_tsp.lk import lk_solve
from simple_tsp.prep import load_problem, load_solution
from simple_tsp.prep import knn_candidates, random_init
from simple_tsp.helpers import route2cost
from simple_tsp.perturb import double_bridge_kick

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath',       type=str, default='data/tsplib/ch150.tsp')
    parser.add_argument('--n-cands',      type=int, default=10)
    parser.add_argument('--n-kick-iters', type=int, default=100)
    parser.add_argument('--depth',        type=int, default=4)
    parser.add_argument('--seed',         type=int, default=123)
    parser.add_argument('--n-jobs',       type=int, default=10)
    return parser.parse_args()

args = parse_args()

_ = np.random.seed(args.seed)
numba.set_num_threads(args.n_jobs)

print(numba.threading_layer())

# --
# Load problem

dist, n_nodes = load_problem(args.inpath)
near          = knn_candidates(dist, args.n_cands)

# --
# Initialize route

route     = random_init(n_nodes)
best_cost = route2cost(route, dist)

# --
# Optimal route

opt_tour_path = args.inpath.replace('.tsp', '.opt.tour')
if os.path.exists(opt_tour_path):
    opt_route = load_solution(opt_tour_path)
    opt_cost  = route2cost(opt_route, dist)
    gap       = best_cost / opt_cost
else:
    opt_cost = None
    gap      = None

# --
# Run

@njit(parallel=True)
def multi_lk_solve(dist, near, route, depth, n_jobs):
    
    best_cost = route2cost(route, dist)
    costs     = np.zeros(n_jobs, dtype=np.int64) + best_cost
    
    best_route = route
    routes     = np.zeros((n_jobs, len(route)), dtype=np.int64)
    for idx in range(n_jobs):
        routes[idx] = route
    
    # for _ in range(10):
    #     for idx in prange(n_jobs):
    #         new_route = lk_solve(dist, near, double_bridge_kick(best_route), depth, lk_neibs)
    #         new_cost  = route2cost(new_route, dist)
            
    #         if new_cost < best_cost:
    #             costs[idx]  = new_cost
    #             routes[idx] = new_route
            
    #         print(costs)
        
    #     best_idx   = np.argmin(costs)
    #     best_cost  = costs[best_idx]
    #     best_route = routes[best_idx]
    #     costs.fill(best_cost)
    
    for idx in prange(10):
        lk_solve(dist=dist, near=near, route=route, depth=depth)
    
    best_idx = np.argmin(costs)
    return routes[best_idx], costs[best_idx]

t = time()
multi_lk_solve(
    dist, 
    near, 
    route, 
    depth  = args.depth, 
    n_jobs = args.n_jobs
)
print(time() - t)

t = time()
multi_lk_solve(
    dist, 
    near, 
    route, 
    depth  = args.depth, 
    n_jobs = args.n_jobs
)
print(time() - t)

# print('final_cost', cost)