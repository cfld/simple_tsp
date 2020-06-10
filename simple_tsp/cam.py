import numba
from typing import NamedTuple

import numpy as np
from numba import njit
from scipy.spatial.distance import pdist, squareform

# --
# Modify routes

@njit(cache=True)
def switch_depot(n, new_depot, node2pre, node2suc, node2route, node2depot, dir=1):
    if node2depot[n]: return
    
    if dir == 1:
        while True:
            nn = node2suc[n]
            node2route[n] = new_depot
            if node2depot[nn]:
                node2suc[n] = new_depot
                node2pre[new_depot] = n
                return
            n = nn
    else:
        while True:
            nn = node2pre[n]
            node2route[n] = new_depot
            if node2depot[nn]:
                node2pre[n] = new_depot
                node2suc[new_depot] = n
                return
            n = nn


@njit(cache=True)
def flip_route(depot, node2pre, node2suc, node2depot):
    n = depot
    while True:
        nn = node2suc[n]
        node2suc[n] = node2pre[n]
        node2pre[n] = nn
        
        if node2depot[nn]:
            break

        n = nn


@njit(cache=True)
def change_edge(n0, n1, r, node2pre, node2suc, node2route, node2depot):
    switch_depot(n0, r, node2pre, node2suc, node2route, node2depot)
    n1_suc           = n0 if not node2depot[n0] else r
    node2suc[n1]     = n1_suc
    node2pre[n1_suc] = n1


@njit(cache=True)
def execute_move(move, depth, node2pre, node2suc, node2route, node2depot):
    n_moves = depth + 1
    
    # Flip routes
    for i in range(n_moves):
        n0, n1, r = move[i]
        flip = node2suc[n0] != n1
        if flip:
            flip_route(r, node2pre, node2suc, node2depot)
    
    # Change edges
    for i in range(n_moves):
        j   = (i + 1) % n_moves
        n01 = move[i, 1]
        n10 = move[j, 0]
        r   = move[j, 2]
        change_edge(n01, n10, r, node2pre, node2suc, node2route, node2depot)

# --
# Run

@njit(cache=True)
def do_camk(dist, near, node2pre, node2suc, node2route, node2depot, node2pen, cap, n_nodes, n_vehicles, max_depth=3):
    move = np.zeros((max_depth, 3), dtype=np.int64) - 1
    
    loads = np.zeros((max_depth, 2), dtype=np.int64) - 1
    
    old_cost = route2cost(n_vehicles, node2suc, dist)
    old_pen  = all_pen(n_vehicles, node2suc, node2pen, cap)
    
    improved = True
    while improved:
        improved = False
        for n00 in range(n_nodes):
            for d0 in [1, -1]:
                r0  = node2route[n00]
                n01 = node2suc[n00] if d0 == 1 else node2pre[n00]
                
                move[0, 0] = n00
                move[0, 1] = n01
                move[0, 2] = r0
                
                loads[0, 0] = partial_load(n00, d0 == 1, node2suc, node2pre, node2depot, node2pen)
                loads[0, 1] = partial_load(n01, d0 != 1, node2suc, node2pre, node2depot, node2pen)
                
                sav_init  = dist[n00, n01]
                move, depth, gain, sav = _camk(
                    move,
                    loads,
                    sav_init,
                    old_pen,
                    dist, 
                    near,
                    
                    node2pre, 
                    node2suc, 
                    node2route, 
                    node2depot,
                    
                    node2pen,
                    cap,
                    n_nodes, 
                    depth=1, 
                    max_depth=max_depth
                )
                
                if (gain > 0) or (gain == 0 and sav > 0):
                    improved = True
                    execute_move(move, depth, node2pre, node2suc, node2route, node2depot)
                    old_pen -= gain


class CostModel:
    def __init__(self, n):
        self._i = 0
        self._n = n

    def __call__(self, expr, caller, callee):
        ret = self._i < self._n
        self._i += 1
        return ret

@njit(cache=True, inline=CostModel(4))
def _camk(move, loads, sav, pen, dist, near, node2pre, node2suc, node2route, node2depot, node2pen, cap, n_nodes, depth, max_depth):
    
    fin        = move[0, 0]
    act        = move[depth - 1, 1]
    act_depot  = node2depot[act]
    fin_depot  = node2depot[fin]
    
    act_pload = loads[depth - 1, 1]
    fin_pload = loads[0, 0]
    
    for nd0 in near[act]:
        if act_depot and node2depot[nd0]: continue # no depot-depot connections
        
        rd = node2route[nd0]
        
        if depth >= 1:
            if rd == move[0, 2]: continue
        if depth >= 2: 
            if rd == move[1, 2]: continue
        if depth >= 3: 
            if rd == move[2, 2]: continue
        if depth >= 4: 
            if rd == move[3, 2]: continue
        if depth >= 5: 
            if rd == move[4, 2]: continue
                
        sav1 = sav - dist[act, nd0]
        
        for d1 in [1, -1]:
            
            nd1  = node2suc[nd0] if d1 == 1 else node2pre[nd0]
            
            sav2 = sav1 + dist[nd0, nd1]

            move[depth, 0] = nd0
            move[depth, 1] = nd1
            move[depth, 2] = rd
                        
            loads[depth, 0] = partial_load(nd0, d1 == 1, node2suc, node2pre, node2depot, node2pen)
            loads[depth, 1] = partial_load(nd1, d1 != 1, node2suc, node2pre, node2depot, node2pen)
            
            if loads[depth, 0] + act_pload - cap > pen: continue
            
            if not (fin_depot and node2depot[nd1]):
                if loads[depth, 1] + fin_pload - cap <= pen:
                    sav_close = sav2 - dist[nd1, fin]
                    
                    gain = le_additive_gain(loads, depth, maxval=cap)
                    if (gain > 0) or (gain == 0 and sav_close > 0):
                        return move, depth, gain, sav_close
            
            if depth < max_depth - 1:
                dmove, ddepth, dgain, dsav = _camk(
                    move,
                    loads, 
                    sav2,
                    pen,
                    dist, 
                    near, 
                    node2pre, 
                    node2suc, 
                    node2route, 
                    node2depot, 
                    node2pen,
                    cap,
                    n_nodes, 
                    depth=depth + 1, 
                    max_depth=max_depth
                )
                
                if (dgain > 0) or (dgain == 0 and dsav > 0):
                    return dmove, ddepth, dgain, dsav

    return move, -1, -1, -1