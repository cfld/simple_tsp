#!/usr/bin/env python

"""
    main.py
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

from simple_tsp import cam

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath',       type=str, default='data/cvrp/INSTANCES/Uchoa/X-n101-k25.vrp')
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

n_vehicles = prob['VEHICLES']    
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
near = cam.knn_candidates(dist, n_cands=args.n_cands, n_vehicles=n_vehicles)

# --
# Run

node2pen = demand

tt = 0
_ = np.random.seed(123)
total = 0

best_pen  = np.inf
best_cost = np.inf

best_route = cam.random_pos2node(n_vehicles, n_nodes)
for it in range(args.n_iters):
    pos2node = double_bridge_kick(best_route)
    pos2node[pos2node < n_vehicles] = np.arange(n_vehicles)
    node2pre, node2suc, node2route, node2depot, _ = cam.init_routes(pos2node, n_vehicles, n_nodes)
    
    t = time()
    cam.do_camk(
        dist, 
        near, 
        node2pre,
        node2suc,
        node2route,
        node2depot,
        node2pen, 
        cap,
        n_nodes, 
        n_vehicles, 
        max_depth=args.max_depth
    )
    tt += time() - t
    
    new_cost = cam.route2cost(n_vehicles, node2suc, dist)
    new_pen  = cam.all_pen(n_vehicles, node2suc, node2pen, cap)
    
    if new_cost < best_cost and new_pen <= best_pen:
        best_route = np.hstack(cam.walk_routes(n_vehicles, node2suc))
        best_cost  = new_cost
        best_pen   = new_pen
    
    total += new_cost
    print(it, new_cost, new_pen, best_cost, best_pen, tt)
