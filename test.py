#!/usr/bin/env python

"""
    run_cam.py
    
    # !! Should run LK on individual routes
    # !! Perturbations should target routes w/ penalties
    # !! Perturb with penalties, double bridge kick is too large, I think
        - How are penalties implemented in RoutingSolver?
    
    # !! With multiple nodes as depot, some neighbors are going to be all depot
    
    # !! Relax sequential requirement, sometimes
    # !! Size of perturbations?
    # !! Width-based perturbations
    # !! Anneal perturbations?
"""

import os
import sys
import argparse
import numpy as np
from time import time
from scipy.spatial.distance import squareform, pdist

from simple_tsp.helpers import set_seeds
from simple_tsp.perturb import double_bridge_kick, double_bridge_kick_targeted
from simple_tsp.prep import load_problem, random_init, route2lookups, knn_candidates

from simple_tsp.moves.rc import do_rc
from simple_tsp.moves.ce import do_ce
# from simple_tsp.moves.ropt import do_ropt
from simple_tsp.moves.lk import lk_solve

from simple_tsp.helpers import suc2cost, walk_routes, walk_route

# --
# Helpers

from numba import njit
# @njit(cache=True)
def dumb_lk(dist, node2suc, n_nodes, n_vehicles, route2stale, depth=4, n_cands=10):
    pos2node = walk_routes(node2suc, n_nodes, n_vehicles)
    node2pre, node2suc, node2route, node2depot, pos2route = route2lookups(pos2node, n_nodes=n_nodes, n_vehicles=n_vehicles)
    
    for depot in range(n_vehicles):
        if not route2stale[depot]: continue
        route2stale[depot] = False
        
        route   = walk_route(depot, node2suc)
        
        proute  = np.arange(len(route))
        proute  = np.hstack((proute, np.array([0])))
        
        subdist = dist[route][:,route]
        subnear = knn_candidates(subdist, n_cands=n_cands, n_vehicles=1)
        
        pos2node[pos2route == depot] = route[lk_solve(subdist, subnear, proute, depth=depth)[:-1]]
    
    return route2lookups(pos2node, n_nodes=n_nodes, n_vehicles=n_vehicles)

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath',       type=str, default='data/cvrp/INSTANCES/Belgium/L1.vrp')
    parser.add_argument('--n-cands',      type=int, default=10)
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
n_vehicles = 203 # l1
# n_vehicles = 485 # g1
# --
# n_vehicles = prob['VEHICLES']    
# <<
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
xy = xy + np.arange(xy.shape[0]).reshape(-1, 1) / xy.shape[0] # prevent ties

xy = np.row_stack([
    np.repeat(xy[0].reshape(1, -1), n_vehicles, axis=0),
    xy[1:]
])

dist = squareform(pdist(xy)).round().astype(np.int32)
near = knn_candidates(dist, n_cands=args.n_cands, n_vehicles=n_vehicles)

# --
# Run

cap__data   = demand
cap__maxval = cap

# Init
# <<
# best_route = random_init(n_vehicles, n_nodes)
# best_pen  = np.inf
# best_cost = np.inf
# pos2node = best_route.copy()
# --
for _ in range(2):
    pos2node = best_route.copy()
    node2pre, node2suc, node2route, node2depot, _ = route2lookups(pos2node, n_nodes=n_nodes, n_vehicles=n_vehicles)
    best_cost = suc2cost(node2suc, dist, n_vehicles)
    best_pen  = 0
    # >>

    # Perturb
    node2pre, node2suc, node2route, node2depot, _ = route2lookups(pos2node, n_nodes=n_nodes, n_vehicles=n_vehicles)

    prob = {
        "near" : near, 
        
        "node2pre"    : node2pre,
        "node2suc"    : node2suc,
        "node2route"  : node2route,
        "node2depot"  : node2depot, 
        
        "active"      : np.ones(n_nodes, dtype=bool),
        "route2stale" : np.ones(n_vehicles, dtype=bool),
        
        "n_nodes"    : n_nodes, 
        "n_vehicles" : n_vehicles, 
        
        # @CONSTRAINT -- params
        "cap__data"   : cap__data,
        "cap__maxval" : cap__maxval, 
        # <<
        
        "validate" : True,
        # "improving_only" : True,
    }

    t = time()
    _, _ = do_ce(dist, **prob)
    print(time() - t)
    
    print(suc2cost(prob['node2suc'], dist, n_vehicles))
