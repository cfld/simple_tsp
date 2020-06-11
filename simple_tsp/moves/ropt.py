import numba
from typing import NamedTuple

import numpy as np
from numba import njit
from scipy.spatial.distance import pdist, squareform

from .cam_helpers import routes2cost
from .cam_constraints import *

# --
# Run

@njit(cache=True)
def do_ropt(
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
    
    move = np.zeros((max_depth, 4), dtype=np.int64) - 1
    
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
                move[0, 3] = d0 == -1

                # >> @CONSTRAINT -- remove_edge
                cap__acc[0, 0] = cap__add_node(n00, d0 == 1, node2suc, node2pre, node2depot, cap__data)
                cap__acc[0, 1] = cap__add_node(n01, d0 != 1, node2suc, node2pre, node2depot, cap__data)
                # << 

                sav_init  = dist[n00, n01]
                move, depth, gain, sav = _ropt(
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
                    n_vehicles=n_vehicles,
                    depth=1, 
                    max_depth=max_depth,

                    # >> @CONSTRAINT -- params
                    cap__acc=cap__acc,
                    cap__data=cap__data,
                    cap__maxval=cap__maxval,
                    # <<
                )
                
                if (gain > 0) or (gain == 0 and sav > 1e-5):
                    improved = True
                    
                    execute_move(move, depth, node2pre, node2suc, node2route, node2depot)
                    
                    pen_init  -= gain
                    cost_init -= sav
    
    return cost_init, pen_init
    

class CostModel:
    def __init__(self, n):
        self._i = 0
        self._n = n

    def __call__(self, expr, caller, callee):
        ret = self._i < self._n
        self._i += 1
        return ret


# @njit(cache=True, inline=CostModel(4))
# def _cam_ce(
#         move,
#         dist, near,
#         node2pre, node2suc, node2route, node2depot,

#         sav_init,
#         pen_init,
#         n_nodes,
#         n_vehicles,
#         depth,
#         max_depth,

#         # >> @CONSTRAINT -- params
#         cap__acc,
#         cap__data,
#         cap__maxval,
#         # <<
#     ):
    
#     fin        = move[0, 0]
#     act        = move[depth - 1, 1]
#     act_depot  = node2depot[act]
#     fin_depot  = node2depot[fin]
    
#     act_pload = cap__acc[depth - 1, 1]
#     fin_pload = cap__acc[0, 0]

#     move2     = np.zeros((2, 4), dtype=np.int64) - 1
#     cap__acc2 = np.zeros((2, 2), dtype=np.int64) - 1

#     for nd0 in near[act]:
#         if act_depot and node2depot[nd0]: continue # no depot-depot connections
        
#         rd = node2route[nd0]
        
#         if rd == move[0, 2]: continue
                
#         sav_add = sav_init - dist[act, nd0]
        
#         for d1 in [1, -1]:
            
#             nd1 = node2suc[nd0] if d1 == 1 else node2pre[nd0]
            
#             sav_add_drop = sav_add + dist[nd0, nd1]

#             move[depth, 0] = nd0
#             move[depth, 1] = nd1
#             move[depth, 2] = rd
#             move[depth, 3] = d1 == -1
            
#             cap__acc[depth, 0] = cap__add_node(nd0, d1 == 1, node2suc, node2pre, node2depot, cap__data)
#             cap__acc[depth, 1] = cap__add_node(nd1, d1 != 1, node2suc, node2pre, node2depot, cap__data)
            
#             # close now
#             # if not (fin_depot and node2depot[nd1]): 
#             #     sav_close = sav_add_drop - dist[nd1, fin]
                
#             #     # >> @ CONSTRAINT -- compute gain
#             #     gain = cap__compute_gain(cap__acc, depth, cap__maxval)
#             #     # <<
                
#             #     if (gain > 0) or (gain == 0 and sav_close > 0):
#             #         return move, depth, gain, sav_close
            
#             if (fin_depot and node2depot[nd1]): continue
#             sav_close = sav_add_drop - dist[nd1, fin]
            
#             if sav_close < 0: continue # optional -- but big speedup
            
#             gain = cap__compute_gain(cap__acc, depth, cap__maxval)
            
#             # execute move
#             # a_suc = node2suc.copy()
#             # a_pre = node2pre.copy()
#             # a_rte = node2route.copy()
#             execute_move(move, 1, node2pre, node2suc, node2route, node2depot)
                        
#             for _ in range(2):
#                 for _ in range(2):
#                     n0 = move[0, 0]
#                     r0 = move[0, 2]
#                     cap__acc2[0, 0] = cap__add_node(n0, True, node2suc, node2pre, node2depot, cap__data)
#                     cap__acc2[0, 1] = cap__add_node(node2suc[n0], False, node2suc, node2pre, node2depot, cap__data)

#                     n1 = n1_orig = move[1, 0]
#                     r1 = move[1, 2]
#                     cap__acc2_10 = cap__add_node(n1, True, node2suc, node2pre, node2depot, cap__data)
#                     cap__acc2_11 = cap__add_node(node2suc[n1], False, node2suc, node2pre, node2depot, cap__data)

#                     while True:
#                         n0_next = node2suc[n0]
                        
#                         n1 = n1_orig
#                         cap__acc2[1, 0] = cap__acc2_10
#                         cap__acc2[1, 1] = cap__acc2_11
                        
#                         while True:
#                             n1_next = node2suc[n1]
                                                        
#                             gain2 = cap__compute_gain(cap__acc2, 1, cap__maxval)
                            
#                             if gain + gain2 >= 0:
#                                 xsav = (
#                                     + dist[n0, n0_next]
#                                     + dist[n1, n1_next]
#                                     - dist[n0, n1_next]
#                                     - dist[n1, n0_next]
#                                 )
                                
#                                 if sav_close + xsav > 0:
#                                     move2[0, 0] = n0
#                                     move2[0, 1] = n0_next
#                                     move2[0, 2] = r0
#                                     move2[0, 3] = False
#                                     move2[1, 0] = n1
#                                     move2[1, 1] = n1_next
#                                     move2[1, 2] = r1
#                                     move2[1, 3] = False
#                                     execute_move(move2, 1, node2pre, node2suc, node2route, node2depot)
#                                     return move2, 1, gain + gain2, sav_close + xsav
                            
#                             n1 = n1_next
#                             cap__acc2[1, 0] += cap__data[n1]
#                             cap__acc2[1, 1] -= cap__data[n1]
#                             if n1 == r1: break
                            
#                             if cap__acc2[0, 1] + cap__acc2[1, 0] > cap__maxval: break # additive pruning
                        
#                         n0 = n0_next
#                         cap__acc2[0, 0] += cap__data[n0]
#                         cap__acc2[0, 1] -= cap__data[n0]
#                         if n0 == r0: break
                    
#                     flip_route(r1, node2pre, node2suc, node2depot)
                
#                 flip_route(r0, node2pre, node2suc, node2depot)
            
#             reverse_move(move, 1, node2pre, node2suc, node2route, node2depot)
#             # b_suc = node2suc.copy()
#             # b_pre = node2pre.copy()
#             # b_rte = node2route.copy()
#             # assert (a_suc == b_suc).all()
#             # assert (a_pre == b_pre).all()
#             # assert (a_rte == b_rte).all()

#     return move, -1, -1, -1
    
    
@njit(cache=True, inline=CostModel(5))
def _ropt(
        move,
        dist, near,
        node2pre, node2suc, node2route, node2depot,

        sav_init,
        pen_init,
        n_nodes,
        n_vehicles,
        
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
            move[depth, 3] = d1 == -1
            
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
                _move, _depth, _gain, _sav = _ropt(
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
                    n_vehicles=n_vehicles,
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


