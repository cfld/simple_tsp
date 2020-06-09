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

from simple_tsp.prep import load_problem, knn_candidates
from simple_tsp.helpers import set_seeds

from simple_tsp import cam

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
near = knn_candidates(dist, 20)

# --
# Run

tt = time()

# >>

# tt = 0
# _ = np.random.seed(123)
# total = 0
# for _ in range(50):
#     node2route, node2depot, node2suc, node2pre, pos2node = cam.init_routes(n_vehicles, n_nodes)
    
#     t = time()
#     cam.do_cam3(dist, near, node2pre, node2suc, node2route, node2depot, n_nodes)
#     new_cost = cam.route2cost(n_vehicles, node2suc, dist)
#     tt += time() - t
#     print('new_cost', new_cost, tt)

# print(tt)

print('-' * 50)

node2pen = demand

tt = 0
_ = np.random.seed(123)
total = 0
for _ in range(50):
    node2route, node2depot, node2suc, node2pre, pos2node = cam.init_routes(n_vehicles, n_nodes)
    
    t = time()
    cam.do_camk(dist, near, node2pre, node2suc, node2route, node2depot, node2pen, cap, n_nodes, n_vehicles, max_depth=4)
    
    new_cost = cam.route2cost(n_vehicles, node2suc, dist)
    new_pen  = cam.all_pen(n_vehicles, node2suc, node2pen, cap)
    
    tt += time() - t
    total += new_cost
    print(new_cost, new_pen, tt)

# print(tt, total / 50)
