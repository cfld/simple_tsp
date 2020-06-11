#!/usr/bin/env python

"""
    run_cam.py
    
    # !! Should run LK on individual routes
    # !! Perturbations should target routes w/ penalties
    # !! Perturb with penalties, double bridge kick is too large, I think
        - How are penalties implemented in RoutingSolver?
    
    # !! With multiple nodes as depot, some neighbors are going to be all depot
"""

import os
import argparse
import numpy as np
from time import time
from scipy.spatial.distance import squareform, pdist

from simple_tsp.prep import load_problem
from simple_tsp.helpers import set_seeds
from simple_tsp.perturb import double_bridge_kick, double_bridge_kick_targeted

from simple_tsp.moves.rc import do_rc
from simple_tsp.moves.ropt import do_ropt
from simple_tsp.moves.ce_cam import do_camce

from simple_tsp.cam_helpers import knn_candidates, routes2cost, walk_routes
from simple_tsp.cam_init import random_pos2node, init_routes

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
n_vehicles = 203
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

dist = squareform(pdist(xy))# .astype(np.int32)
near = knn_candidates(dist, n_cands=args.n_cands, n_vehicles=n_vehicles)

# --
# Run

cap__data   = demand
cap__maxval = cap

tt = 0
_ = np.random.seed(123)
total = 0

# Init
# <<
# best_route = random_pos2node(n_vehicles, n_nodes)
# best_pen  = np.inf
# best_cost = np.inf
# pos2node = best_route.copy()
# --
from simple_tsp.cam_helpers import routes2cost
pos2node = best_route.copy()
node2pre, node2suc, node2route, node2depot, _ = init_routes(pos2node, n_vehicles, n_nodes)
best_cost = routes2cost(dist, node2suc, n_vehicles)
best_pen  = 0
# >>

print(best_cost, best_pen)

tt = time()
for it in range(1):
    
    # Perturb
    node2pre, node2suc, node2route, node2depot, _ = init_routes(pos2node, n_vehicles, n_nodes)
    
    # Optimize    
    tmp_cost = routes2cost(dist, node2suc, n_vehicles)
    
    improved = True
    while improved:
        improved = False
        
        # new_cost, new_pen = do_camk(
        #     dist,
        #     near, 
            
        #     node2pre,
        #     node2suc,
        #     node2route,
        #     node2depot, 
            
        #     n_nodes, 
        #     n_vehicles, 
            
        #     max_depth=args.max_depth,
            
        #     # @CONSTRAINT -- params
        #     cap__data=cap__data,
        #     cap__maxval=cap__maxval, 
        #     # <<
        # )
        
        # CE move
        new_cost, new_pen, changed = do_camce(
            dist,
            near, 
            
            node2pre,
            node2suc,
            node2route,
            node2depot, 
            
            n_nodes, 
            n_vehicles, 
            
            # @CONSTRAINT -- params
            cap__data=cap__data,
            cap__maxval=cap__maxval, 
            # <<
        )
        
        # >> LK the slow way
        from simple_tsp.lk import lk_solve
        from simple_tsp.cam_helpers import walk_route
        
        pos2node = np.hstack(walk_routes(n_vehicles, node2suc))
        node2pre, node2suc, node2route, node2depot, pos2route = init_routes(pos2node, n_vehicles, n_nodes)
        
        for depot in changed:
            route   = walk_route(depot, node2suc)
            proute  = np.arange(len(route))
            proute  = np.hstack([proute, [0]])
            
            subdist = dist[route][:,route]
            subnear = knn_candidates(subdist, n_cands=10, n_vehicles=1)
            
            lk_opt = lk_solve(subdist, subnear, proute, depth=4)
            pos2node[pos2route == depot] = route[lk_opt[:-1]]
        
        node2pre, node2suc, node2route, node2depot, pos2route = init_routes(pos2node, n_vehicles, n_nodes)
        # >>
        
        new_cost, new_pen = do_rc(
            dist,
            near, 
            
            node2pre,
            node2suc,
            node2route,
            node2depot, 
            
            n_nodes, 
            n_vehicles, 
            
            # @CONSTRAINT -- params
            cap__data=cap__data,
            cap__maxval=cap__maxval, 
            # <<
        )

        # >> LK the slow way
        from simple_tsp.lk import lk_solve
        from simple_tsp.cam_helpers import walk_route
        
        pos2node = np.hstack(walk_routes(n_vehicles, node2suc))
        node2pre, node2suc, node2route, node2depot, pos2route = init_routes(pos2node, n_vehicles, n_nodes)
        
        for depot in range(n_vehicles):
            route   = walk_route(depot, node2suc)
            proute  = np.arange(len(route))
            proute  = np.hstack([proute, [0]])
            
            subdist = dist[route][:,route]
            subnear = knn_candidates(subdist, n_cands=10, n_vehicles=1)
            
            lk_opt = lk_solve(subdist, subnear, proute, depth=4)
            pos2node[pos2route == depot] = route[lk_opt[:-1]]
        
        node2pre, node2suc, node2route, node2depot, pos2route = init_routes(pos2node, n_vehicles, n_nodes)
        # >>
        
        new_cost = routes2cost(dist, node2suc, n_vehicles)
        
        print('post_cost', new_cost)
        if new_cost < tmp_cost:
            tmp_cost = new_cost
            improved = True
    
    # Record
    if (new_pen, new_cost) < (best_pen, best_cost):
        best_route = np.hstack(walk_routes(n_vehicles, node2suc))
        assert len(best_route) == n_nodes
        best_cost  = new_cost
        best_pen   = new_pen
    
    total += new_cost
    print(it, new_cost, new_pen, best_cost, best_pen, time() - tt)

    # Perturb
    # <<
    # pos2node = double_bridge_kick(best_route)
    # --
    # pos2node = double_bridge_kick_targeted(best_route, dist, node2suc, n_vehicles)
    # --
    # from simple_tsp.cam_constraints import cap__route2pen
    # from simple_tsp.perturb import double_bridge_kick_weighted
    # weights = np.array([cap__route2pen(i, node2suc, cap__data, cap__maxval) for i in range(n_vehicles)])
    # weights = weights[node2route]
    # weights = weights / weights.sum()
    # pos2node = double_bridge_kick_weighted(best_route, weights)
    # >>
    
    pos2node[pos2node < n_vehicles] = np.arange(n_vehicles)
