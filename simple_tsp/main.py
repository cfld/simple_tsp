#!/usr/bin/env python

"""
    main.py
"""

import json
import argparse
import numpy as np
from time import time

from simple_tsp.lk import lk_solve
from simple_tsp.prep import load_problem, knn_candidates, random_init
from simple_tsp.helpers import route2cost
from simple_tsp.perturb import double_bridge_kick

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath',       type=str, default='data/tsplib/ch150.tsp')
    parser.add_argument('--n-cands',      type=int, default=10)
    parser.add_argument('--n-kick-iters', type=int, default=100)
    parser.add_argument('--depth',        type=int, default=4)
    parser.add_argument('--seed',         type=int, default=123)
    return parser.parse_args()

args = parse_args()

_ = np.random.seed(args.seed)

# --
# Load problem

dist, n_nodes = load_problem(args.inpath)
near          = knn_candidates(dist, args.n_cands)

# --
# Initialize route

route     = random_init(n_nodes)
best_cost = route2cost(route, dist)

# --
# Run

t = time()
for _ in range(args.n_kick_iters):
    
    new_route = lk_solve(dist, near, route, depth=args.depth, lk_neibs=args.n_cands)
    assert len(set(new_route)) == len(set(route))
    
    cost = route2cost(new_route, dist)
    
    if cost < best_cost:
        best_route = new_route.copy()
        best_cost  = cost
    
    route = double_bridge_kick(best_route)
    print(json.dumps({'cost' : int(cost), 'best_cost' : int(best_cost), 'elapsed' : time() - t}))
