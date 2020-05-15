#!/usr/bin/env python

"""
    main.py
"""

from dask.distributed import Client, as_completed

import os
import json
import argparse
import numpy as np
from time import time, sleep

import numba
from numba import njit, prange

from simple_tsp.lk import lk_solve
from simple_tsp.prep import load_problem, load_solution
from simple_tsp.prep import knn_candidates, random_init
from simple_tsp.helpers import route2cost
from simple_tsp.perturb import double_bridge_kick

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath',       type=str, default='data/tsplib/kroC100.tsp')
    parser.add_argument('--n-cands',      type=int, default=10)
    parser.add_argument('--n-kick-iters', type=int, default=100)
    parser.add_argument('--depth',        type=int, default=5)
    parser.add_argument('--seed',         type=int, default=123)
    parser.add_argument('--n-jobs',       type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _ = np.random.seed(args.seed)

    client = Client(n_workers=args.n_jobs, threads_per_worker=1)
    
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
        gap       = (best_cost / opt_cost) - 1
    else:
        opt_cost = None
        gap      = None

    # --
    # Run

    _ = lk_solve(dist, near, route, args.depth)

    def _wrapper(dist, near, route, depth):
        from simple_tsp.lk import lk_solve
        
        best_route = route
        best_cost  = route2cost(route, dist)
        for _ in range(32):
            new_route = lk_solve(dist, near, double_bridge_kick(best_route), depth) # !! Not sure about random seeds here
            new_cost  = route2cost(new_route, dist)
            if new_cost < best_cost:
                return new_cost, new_route
        
    _ = _wrapper(dist, near, route, args.depth)
    
    # --
    
    dist_ = client.scatter(dist)
    near_ = client.scatter(near)
    
    q = as_completed(
        [client.submit(_wrapper, dist_, near_, double_bridge_kick(route), args.depth) for _ in range(2 * args.n_jobs)]
    )
    
    t = time()
    counter = 0
    for res in q:
        new_cost, new_route = res.result()
        if new_cost < best_cost:
            best_cost  = new_cost
            best_route = new_route
            gap        = (best_cost / opt_cost) - 1 if opt_cost else -1
            
            print('*', counter, new_cost, best_cost, gap, time() - t)
            
            if gap == 0: break
        
        if counter % 100 == 0:
            print('-', counter, new_cost, best_cost, gap, time() - t)
        
        q.add(client.submit(_wrapper, dist_, near_, double_bridge_kick(best_route), args.depth))
        counter += 1

