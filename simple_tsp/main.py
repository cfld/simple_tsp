#!/usr/bin/env python

"""
    main.py
"""

import argparse
import numpy as np
from time import time

from simple_tsp.lk import lk_solve
from simple_tsp.prep import load_problem, knn_candidates, random_init
from simple_tsp.helpers import route2cost

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath',  type=str, default='data/tsplib/ch150.tsp')
    parser.add_argument('--n-cands', type=int, default=10)
    parser.add_argument('--depth',   type=int, default=4)
    parser.add_argument('--seed',    type=int, default=123)
    return parser.parse_args()

args = parse_args()

_ = np.random.seed(args.seed)

# --
# Init

dist, n_nodes = load_problem(args.inpath)
near          = knn_candidates(dist, args.n_cands)

route = random_init(n_nodes)

cost = route2cost(route, dist)
best_cost = cost

tt = time()
for _ in range(100):
    
    t = time()
    new_route = lk_solve(dist, near, route, cost, depth=args.depth, lk_neibs=args.n_cands)
    
    assert len(set(new_route)) == len(set(route))
    new_cost = dist[(new_route[:-1], new_route[1:])].sum()
    
    print(new_cost, best_cost, time() - t)
    
    if new_cost < best_cost:
        best_route = new_route.copy()
        best_cost  = new_cost
        print('*')
    
    t = time()
    route = kick(best_route)
    print(time() - t)

print({'cost' : best_cost, 'elapsed' : time() - tt})
