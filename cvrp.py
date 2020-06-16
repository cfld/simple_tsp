#!/usr/bin/env python

"""
    cvrp.py
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # !! Right now, only works if route is initialized to have 0 penalty
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    # !! LK assumes that penalties are permutation invariant
    
    # !! Perturbations should target routes w/ penalties
    # !! Perturb with penalties, double bridge kick is too large, I think
        - How are penalties implemented in RoutingSolver?
    
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

from clark_wright import clark_wright

from simple_tsp.helpers import set_seeds
from simple_tsp.perturb import double_bridge_kick, double_bridge_kick_targeted
from simple_tsp.prep import load_problem, random_init, route2lookups, knn_candidates, augment_vehicles
from simple_tsp.constraints.cap import routes2pen

from simple_tsp.moves.rc import do_rc
from simple_tsp.moves.ce import do_ce
# from simple_tsp.moves.ropt import do_ropt
from simple_tsp.moves.lk import lk_solve

from simple_tsp.helpers import suc2cost, walk_routes, walk_route

PENALTY_INCREMENT = 7

# --
# Helpers

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
        subnear = knn_candidates(subdist, n_cands=min(len(route), n_cands), n_vehicles=1)
        
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

xy     = np.row_stack(list(prob['NODE_COORD_SECTION'].values()))
demand = np.array(list(prob['DEMAND_SECTION'].values()))
cap    = prob['CAPACITY']

depot_id = 0
dist     = squareform(pdist(xy)).round().astype(np.int32)

# --
# Compute initial solution w/ Clark-Wright savings algorithm

cw_routes  = clark_wright(dist, demand, cap, depot_id, n_close=100)
n_vehicles = len(cw_routes)

# --
# Convert to appropriate format

dist, demand, best_route = augment_vehicles(dist, demand, cw_routes, n_vehicles)
n_nodes = dist.shape[0]

near = knn_candidates(dist, n_cands=args.n_cands, n_vehicles=n_vehicles)

print(best_route)

# --
# Constraints

cap__data   = demand
cap__maxval = cap

# --
# Init

pos2node = best_route.copy()
node2pre, node2suc, node2route, node2depot, _ = route2lookups(pos2node, n_nodes=n_nodes, n_vehicles=n_vehicles)
best_cost  = suc2cost(node2suc, dist, n_vehicles)
best_pen   = routes2pen(node2suc, n_vehicles, cap__data, cap__maxval)
assert best_pen == 0

print(best_cost, best_pen)
raise Exception()

# --
# Run

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
    
    # "validate"       : True,
    # "improving_only" : True,
}

pdist = dist.copy()
cost  = dist.copy().astype(np.float64)
pens  = np.zeros(dist.shape, dtype=np.int64)

# --
# Optimize

p_iters = 64

print('cvrp.py: start', file=sys.stderr)

t = time()
inner_time = 0
outer_iter = 0

while True:
    outer_iter += 1
    
    # prob['active'][:] = True
    
    inner_iter = 0
    
    ref_cost = suc2cost(prob['node2suc'], dist, n_vehicles)
    while True:
        tt = time()
        
        # --
        # CE
        
        _, _ = do_ce(dist, **prob)
        
        # --
        # LK
        
        prob['node2pre'], prob['node2suc'], prob['node2route'], prob['node2depot'], _ = \
                dumb_lk(dist, prob['node2suc'], n_nodes, n_vehicles, route2stale=prob['route2stale'])
        
        # --
        # RC
        
        _, _ = do_rc(dist, **prob)
        
        # --
        # LK
        
        prob['node2pre'], prob['node2suc'], prob['node2route'], prob['node2depot'], _ = \
                dumb_lk(dist, prob['node2suc'], n_nodes, n_vehicles, route2stale=prob['route2stale'])
        
        # --
        # Score
        
        inner_time += time() - tt
        
        new_cost = suc2cost(prob['node2suc'], dist, n_vehicles)
        print(outer_iter, inner_iter, best_cost, new_cost, time() - t, inner_time)
        sys.stdout.flush()
        
        if new_cost < ref_cost:
            ref_cost = new_cost
        else:
            break
        
        inner_iter += 1
    
    if new_cost < best_cost:
        best_cost = new_cost
    
    # --
    # Perturb
    
    prob['active'][:] = False
    for _ in range(p_iters):
        node2suc   = prob['node2suc']
        node2pre   = prob['node2pre']
        node2depot = prob['node2depot']
        
        edge_lengths = cost[np.arange(n_nodes), node2suc]
        
        n = edge_lengths.argmax()
        m = node2suc[n]
        
        if node2depot[n] == 1: n = node2depot == 1 # change cost for all depot nodes
        if node2depot[m] == 1: m = node2depot == 1
        
        pens[n, m]  += 1
        pens[m, n]  += 1
        
        pdist[n, m] += PENALTY_INCREMENT
        pdist[m, n] += PENALTY_INCREMENT
        
        cost[n, m] *= (pens[n, m] / (1 + pens[n, m]))
        cost[m, n] *= (pens[m, n] / (1 + pens[m, n]))
        
        prob['active'][n] = True
        prob['active'][m] = True
        
        _, _ = do_ce(pdist, **prob)
        _, _ = do_rc(pdist, **prob)

        prob['active'][n] = False
        prob['active'][m] = False
    
    
    # prob['improving_only'] = np.random.choice([True, False], p=[0.95, 0.05])
    prob['active'][:] = prob['route2stale'][prob['node2route']]
    prob['node2pre'], prob['node2suc'], prob['node2route'], prob['node2depot'], _ = \
        dumb_lk(dist, prob['node2suc'], n_nodes, n_vehicles, route2stale=prob['route2stale'])
    
    if outer_iter == 2_500:
        p_iters = int(p_iters / 2)
    
    if outer_iter == 5_000:
        p_iters = int(p_iters / 2)
    
    if outer_iter == 10_000:
        p_iters = int(p_iters / 2)

    if outer_iter == 50_000:
        p_iters = int(p_iters / 2)
