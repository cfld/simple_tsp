#!/usr/bin/env python

"""
    main.py
"""

import os
import json
import argparse
import numpy as np
from time import time

from simple_tsp.lk import lk_solve, compute_penalty
from simple_tsp.prep import load_problem, load_solution, get_distance_matrix
from simple_tsp.prep import knn_candidates, random_init
from simple_tsp.helpers import route2cost, set_seeds
from simple_tsp.perturb import double_bridge_kick

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

# <<
route = random_init(n_nodes, random_state=101)
# --
# route = [r[:-1] for r in route]
# route = np.hstack(route)
# route[route == route.max()] = -1
# route[route != -1] += (route == -1).sum()
# route[route == -1] = np.arange((route == -1).sum())
# >>


best_cost = route2cost(route, dist)
best_pen  = compute_penalty(route, demand, cap, n_nodes)
print('init_pen', best_pen)

# --
# Optimal route

opt_cost = None
if prob['TYPE'] == 'TSP':
    opt_tour_path = args.inpath.replace('.tsp', '.opt.tour')
    if os.path.exists(opt_tour_path):
        opt_route = load_solution(opt_tour_path)
        opt_cost  = route2cost(opt_route, dist)

# --
# Run

new_route = lk_solve(dist, near, route, n_nodes, demand, cap, max_depth=2)


# t = time()
# for kick_iter in range(args.n_kick_iters):
    
#     new_route = lk_solve(dist, near, route, n_nodes, demand, cap, max_depth=args.max_depth)
#     assert (np.sort(new_route) == np.sort(route)).all()
    
#     cost = route2cost(new_route, dist)
#     pen  = compute_penalty(new_route, demand, cap, n_nodes)
    
#     if (pen, cost) <= (best_pen, best_cost):
#         best_route = new_route.copy()
#         best_cost  = cost
#         best_pen   = pen
    
#     route = double_bridge_kick(best_route)
    
#     print(json.dumps({
#         'kick_iter' : kick_iter,
#         'cost'      : int(cost), 
#         'pen'       : int(pen),
#         'best_cost' : int(best_cost), 
#         'best_pen'  : int(best_pen), 
#         'opt_cost'  : int(opt_cost) if opt_cost is not None else -1,
#         'gap'       : float(best_cost / opt_cost) - 1 if opt_cost is not None else -1,
#         'elapsed'   : time() - t
#     }))
