#!/usr/bin/env python

"""
    run_cam.py
    
    # !! Should run LK on individual routes
    # !! Perturbations should target routes w/ penalties
"""

import os
import json
import argparse
import numpy as np
from time import time
from scipy.spatial.distance import squareform, pdist

from simple_tsp.prep import load_problem
from simple_tsp.helpers import set_seeds
from simple_tsp.perturb import double_bridge_kick

# from simple_tsp.cam import do_camk
from simple_tsp.ce_cam import do_camce
from simple_tsp.cam_helpers import knn_candidates, routes2cost, walk_routes
from simple_tsp.cam_init import random_pos2node, init_routes

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath',       type=str, default='data/cvrp/INSTANCES/Belgium/L1.vrp')
    parser.add_argument('--n-cands',      type=int, default=5)
    parser.add_argument('--n-iters',      type=int, default=1000)
    parser.add_argument('--max-depth',    type=int, default=4)
    parser.add_argument('--seed',         type=int, default=123)
    return parser.parse_args()

args = parse_args()
_ = set_seeds(args.seed)

# --
# Load problem

prob = load_problem(args.inpath)

# >>
best_route = np.load('/Users/bjohnson/Desktop/routes.npy')
# <<

n_vehicles = 203 # prob['VEHICLES']    
cap        = prob['CAPACITY']

demand = np.array(list(prob['DEMAND_SECTION'].values()))
demand = np.hstack([
    np.repeat(0, n_vehicles),
    demand[1:]
])

n_nodes = demand.shape[0]

# --
# Distance

xy = np.row_stack(list(prob['NODE_COORD_SECTION'].values()))
xy = np.row_stack([
    np.repeat(xy[0].reshape(1, -1), n_vehicles, axis=0),
    xy[1:]
])

dist = squareform(pdist(xy)).astype(np.int32)
near = knn_candidates(dist, n_cands=args.n_cands, n_vehicles=n_vehicles)

# --
# Run

cap__data   = demand
cap__maxval = cap

tt = 0
_ = np.random.seed(123)
total = 0

# best_pen  = np.inf
# best_cost = np.inf

# Init
# best_route = random_pos2node(n_vehicles, n_nodes)
pos2node = best_route.copy()

from simple_tsp.cam_helpers import routes2cost
node2pre, node2suc, node2route, node2depot, _ = init_routes(pos2node, n_vehicles, n_nodes)
best_cost = routes2cost(dist, node2suc, n_vehicles)
best_pen  = 0

print(best_cost, best_pen)

for it in range(1):
    
    # Perturb
    node2pre, node2suc, node2route, node2depot, _ = init_routes(pos2node, n_vehicles, n_nodes)
    
    # Optimize
    t = time()
    new_cost, new_pen = do_camce(
        dist,
        near, 
        
        node2pre,
        node2suc,
        node2route,
        node2depot, 
        
        n_nodes, 
        n_vehicles, 
        
        # max_depth=args.max_depth,
        
        # @CONSTRAINT -- params
        cap__data=cap__data,
        cap__maxval=cap__maxval, 
        # <<
    )
    tt += time() - t
    print('done move')
    
    # Record
    if (new_pen, new_cost) < (best_pen, best_cost):
        best_route = np.hstack(walk_routes(n_vehicles, node2suc))
        assert len(best_route) == n_nodes
        best_cost  = new_cost
        best_pen   = new_pen
    
    total += new_cost
    print(it, new_cost, new_pen, best_cost, best_pen, tt)

    # Perturb
    pos2node = double_bridge_kick(best_route)
    pos2node[pos2node < n_vehicles] = np.arange(n_vehicles)


print(best_route)