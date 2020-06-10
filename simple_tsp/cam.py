import numba
from typing import NamedTuple

import numpy as np
from numba import njit
from scipy.spatial.distance import pdist, squareform

from .cam_helpers import routes2cost
from .cam_constraints import *

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
def do_camk(
        dist,
        near, 
        
        node2pre,
        node2suc,
        node2route,
        node2depot, 
        
        n_nodes, 
        n_vehicles, 
        
        max_depth,
        
        # >> @CONSTRAINT -- params
        cap__data,
        cap__maxval, 
        # <<
    ):
    
    move = np.zeros((max_depth, 3), dtype=np.int64) - 1
    
    cost_init = routes2cost(dist, node2suc, n_vehicles)
    
    # >> @CONSTRAINT -- init
    cap__acc = np.zeros((max_depth, 2), dtype=np.int64) - 1
    pen_init = cap__routes2pen(n_vehicles, node2suc, cap__data, cap__maxval)
    # <<
        
    improved = True
    while improved:
        improved = False
        for n00 in np.random.permutation(n_nodes):
            for d0 in [1, -1]:
                r0  = node2route[n00]
                n01 = node2suc[n00] if d0 == 1 else node2pre[n00]
                
                move[0, 0] = n00
                move[0, 1] = n01
                move[0, 2] = r0
                
                # >> @CONSTRAINT -- remove_edge
                cap__acc[0, 0] = cap__add_node(n00, d0 == 1, node2suc, node2pre, node2depot, cap__data)
                cap__acc[0, 1] = cap__add_node(n01, d0 != 1, node2suc, node2pre, node2depot, cap__data)
                # << 
                
                sav_init  = dist[n00, n01]
                move, depth, gain, sav = _camk(
                    move=move,
                    
                    dist=dist,
                    near=near,
                    
                    node2pre=node2pre,
                    node2suc=node2suc,
                    node2route=node2route,
                    node2depot=node2depot, 
                    
                    sav_init=sav_init, 
                    pen_init=pen_init,
                    n_nodes=n_nodes,
                    depth=1, 
                    max_depth=max_depth,

                    # >> @CONSTRAINT -- params
                    cap__acc=cap__acc,
                    cap__data=cap__data,
                    cap__maxval=cap__maxval,
                    # <<
                )
                
                if (gain > 0) or (gain == 0 and sav > 0):
                    improved = True
                    execute_move(move, depth, node2pre, node2suc, node2route, node2depot)
                    pen_init  -= gain
                    cost_init -= sav
                    print(sav, gain)
    
    return cost_init, pen_init
    

class CostModel:
    def __init__(self, n):
        self._i = 0
        self._n = n

    def __call__(self, expr, caller, callee):
        ret = self._i < self._n
        self._i += 1
        return ret

@njit(cache=True, inline=CostModel(4))
def _camk(
        move,
        dist, near,
        node2pre, node2suc, node2route, node2depot,

        sav_init,
        pen_init,
        n_nodes,
        depth,
        max_depth,

        # >> @CONSTRAINT -- params
        cap__acc,
        cap__data,
        cap__maxval,
        # <<
    ):
    
    fin        = move[0, 0]
    act        = move[depth - 1, 1]
    act_depot  = node2depot[act]
    fin_depot  = node2depot[fin]
    
    act_pload = cap__acc[depth - 1, 1]
    fin_pload = cap__acc[0, 0]
    
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
        if depth >= 6: raise Exception('!! depth too large')
                
        sav_add = sav_init - dist[act, nd0]
        
        for d1 in [1, -1]:
            
            nd1 = node2suc[nd0] if d1 == 1 else node2pre[nd0]
            
            sav_add_drop = sav_add + dist[nd0, nd1]

            move[depth, 0] = nd0
            move[depth, 1] = nd1
            move[depth, 2] = rd
            
            # >> @CONSTRAINT -- add edge
            cap__acc[depth, 0] = cap__add_node(nd0, d1 == 1, node2suc, node2pre, node2depot, cap__data)
            cap__acc[depth, 1] = cap__add_node(nd1, d1 != 1, node2suc, node2pre, node2depot, cap__data)
            # <<
            
            # >> @CONSTRAINT -- prune
            if cap__acc[depth, 0] + act_pload - cap__maxval > pen_init: continue
            # <<
            
            if not (fin_depot and node2depot[nd1]):
                
                # >> @CONSTRAINT -- prune
                if cap__acc[depth, 1] + fin_pload - cap__maxval <= pen_init:
                # <<
                
                    sav_close = sav_add_drop - dist[nd1, fin]
                    
                    # >> @ CONSTRAINT -- compute gain
                    gain = cap__compute_gain(cap__acc, depth, cap__maxval)
                    # <<
                    
                    if (gain > 0) or (gain == 0 and sav_close > 0):
                        return move, depth, gain, sav_close
            
            if depth < max_depth - 1:
                _move, _depth, _gain, _sav = _camk(
                    move=move,
                    
                    dist=dist,
                    near=near,
                    
                    node2pre=node2pre,
                    node2suc=node2suc,
                    node2route=node2route,
                    node2depot=node2depot, 
                    
                    sav_init=sav_add_drop, 
                    pen_init=pen_init,
                    n_nodes=n_nodes,
                    depth=depth + 1, 
                    max_depth=max_depth,
                    
                    # >> @CONSTRAINT -- params
                    cap__acc=cap__acc,
                    cap__data=cap__data,
                    cap__maxval=cap__maxval,
                    # <<
                )
                
                if (_gain > 0) or (_gain == 0 and _sav > 0):
                    return _move, _depth, _gain, _sav

    return move, -1, -1, -1


# @njit(cache=True, inline=CostModel(4))
# def _camk(
#         move,
#         dist, near,
#         node2pre, node2suc, node2route, node2depot,

#         sav_init,
#         pen_init,
#         n_nodes,
#         depth,
#         max_depth,

#         # >> @CONSTRAINT -- params
#         cap__acc,
#         cap__data,
#         cap__maxval,
#         # <<
#     ):
    
#     fin        = move[0, 0]
#     act        = move[0, 1]
#     act_depot  = node2depot[act]
#     fin_depot  = node2depot[fin]
    
#     act_pload = cap__acc[0, 1]
#     fin_pload = cap__acc[0, 0]
    
#     for nd0 in near[act]:
#         if act_depot and node2depot[nd0]: continue # no depot-depot connections
        
#         rd = node2route[nd0]
        
#         if depth >= 1:
#             if rd == move[0, 2]: continue
        
#         sav_add = sav_init - dist[act, nd0]
        
#         for d1 in [1, -1]:
            
#             nd1 = node2suc[nd0] if d1 == 1 else node2pre[nd0]
            
#             sav_add_drop = sav_add + dist[nd0, nd1]

#             move[1, 0] = nd0
#             move[1, 1] = nd1
#             move[1, 2] = rd
            
#             # execute_move
            
#             # search routes for another CAM that gets better gain
#             # if found, execute
#             # else, restore
            
            
            

#     return move, -1, -1, -1