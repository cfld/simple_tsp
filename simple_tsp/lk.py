#!/usr/bin/env python

"""
    simple_tsp/lk.py
"""

import numba
import numpy as np
from numba import njit
from typing import NamedTuple

from simple_tsp.lk_checkers import check_move

# --
# Data types

class RouteState(NamedTuple):
    node2pos : numba.int64[:]
    pos2node : numba.int64[:]
    pres     : numba.int64[:]
    sucs     : numba.int64[:]

@njit(cache=True)
def copy_route_state(rs):
    return RouteState(
        node2pos = rs.node2pos.copy(),
        pos2node = rs.pos2node.copy(),
        pres     = rs.pres.copy(),
        sucs     = rs.sucs.copy(),
    )

class MoveState(NamedTuple):
    cs  : numba.int64[:]
    csh : numba.int64[:]
    old : numba.int64[:,:]
    new : numba.int64[:,:]

# --
# Helpers

@njit(cache=True)
def edge_exists(u, v, n):
    return (
        (u == v + 1) or 
        (v == u + 1) or 
        (u == 0 and v == n - 1) or
        (v == 0 and u == n - 1)
    )


@njit(cache=True)
def contains(u, v, db):
    for i in range(db.shape[0]):
        if db[i, 0] == -1:
            return False
        
        if (
            (u == db[i, 0] and v == db[i, 1]) or
            (u == db[i, 1] and v == db[i, 0])
        ):
            return True
    
    return False

# --
# Optimize route

@njit(cache=True)
def init_route_state(route):
    n_nodes  = len(route)
    pos2node = route.copy()
    node2pos = np.argsort(pos2node)
    
    rs = RouteState(
        pos2node = pos2node,
        node2pos = node2pos,
        pres     = np.array([pos2node[(node2pos[i] - 1) % n_nodes] for i in range(n_nodes)]),
        sucs     = np.array([pos2node[(node2pos[i] + 1) % n_nodes] for i in range(n_nodes)]),
    )
    
    return rs


@njit(cache=True)
def lk_solve(dist, near, route, max_depth=5, lk_neibs=10, use_dlb=False):
    
    assert route[0] == 0, 'route[0] != 0'
    assert route[-1] == 0, 'route[-1] != 0'
    
    near    = near[:,:lk_neibs]
    
    route   = route[:-1]
    n_nodes = len(route)
    rs      = init_route_state(route)
    
    offset   = 0
    improved = True
    
    dlb = np.zeros(n_nodes)
    
    while improved:
        if use_dlb:
            dlb.fill(0)
        
        improved = False
        for _ in range(n_nodes):
            
            c1 = (offset) % n_nodes
            c2 = (offset + 1) % n_nodes
            offset += 1
            
            improved, rs = lk_move(near, dist, rs, c1, c2, n_nodes, max_depth, dlb)
            if improved: break
            
            improved, rs = lk_move(near, dist, rs, c2, c1, n_nodes, max_depth, dlb)
            if improved: break
            
            if use_dlb:
                dlb[c1] = 1
    
    return np.hstack((rs.pos2node, np.array([0])))

# --
# Find best move

@njit(cache=True)
def init_move_state(c1, c2, max_depth):
    ms = MoveState(
        cs  = np.zeros(2 * max_depth, dtype=np.int64) - 1,
        csh = np.zeros(2 * max_depth, dtype=np.int64) - 1,
        old = np.zeros((max_depth, 2), dtype=np.int64) - 1,
        new = np.zeros((max_depth - 1, 2), dtype=np.int64) - 1,
    )
    
    ms.cs[0]  = c1
    ms.cs[1]  = c2
    ms.csh[0] = 0
    ms.csh[1] = 1
    ms.old[0] = (c1, c2)
    
    return ms

@njit(cache=True)
def lk_move(near, dist, rs, c1, c2, n_nodes, max_depth, dlb):
    ms  = init_move_state(c1, c2, max_depth)
    sav = dist[rs.pos2node[c1], rs.pos2node[c2]] # remove 12
    return _lk_move(near, dist, rs, ms, sav, n_nodes, 0, max_depth, dlb)


class CostModel:
    def __init__(self, n):
        self._i = 0
        self._n = n

    def __call__(self, expr, caller, callee):
        ret = self._i < self._n
        self._i += 1
        return ret


@njit(cache=True, inline=CostModel(5))
def _lk_move(near, dist, rs, ms, sav, n_nodes, depth, max_depth, dlb):
    
    fin = ms.cs[0]
    act = ms.cs[2 * depth + 1] # positions
    rev = (ms.cs[1] - fin) % n_nodes == 1
    
    for cp1 in rs.node2pos[near[rs.pos2node[act]]]:
        
        if dlb[cp1] == 1:                  continue
        if cp1 == -1:                      continue
        if cp1 == fin:                     continue
        if cp1 == act:                     continue
        if edge_exists(act, cp1, n_nodes): continue
        if contains(act, cp1, ms.new):     continue
        
        sav_n23 = sav - dist[rs.pos2node[act], rs.pos2node[cp1]] # add 23
        
        if sav_n23 > 0:
            
            for cp2 in [(cp1 - 1) % n_nodes, (cp1 + 1) % n_nodes]:
                if cp2 == -1:                  continue # impossible
                if cp2 == fin:                 continue
                if contains(cp1, cp2, ms.old): continue
                
                ms.new[depth]               = (act, cp1)
                ms.old[depth + 1]           = (cp1, cp2)
                ms.cs[2 * (depth + 1)]      = cp1
                ms.cs[2 * (depth + 1) + 1]  = cp2
                ms.csh[2 * (depth + 1)]     = (cp1 - fin) % n_nodes if rev else abs(((cp1 - fin) % n_nodes) - n_nodes) % n_nodes
                ms.csh[2 * (depth + 1) + 1] = (cp2 - fin) % n_nodes if rev else abs(((cp2 - fin) % n_nodes) - n_nodes) % n_nodes
                
                sav_n23_o34 = sav_n23 + dist[rs.pos2node[cp1], rs.pos2node[cp2]] # remove 34
                
                # exit now
                if not edge_exists(cp2, fin, n_nodes):
                    sav_closed = sav_n23_o34 - dist[rs.pos2node[cp2], rs.pos2node[fin]]
                    if sav_closed > 0:
                        if check_move(ms.csh[:2 * (depth + 2)], n_nodes):
                            
                            rs = execute_move(ms.cs[:2 * (depth + 2)], rs, n_nodes)
                            
                            return True, rs
                
                # search deeper
                if depth < max_depth - 2:
                    deeper_improved, deeper_rs = _lk_move(near, dist, rs, ms, sav_n23_o34, n_nodes, depth + 1, max_depth, dlb)
                    
                    if deeper_improved:
                        return True, deeper_rs
    
    return False, rs

# --
# Execute move

@njit(cache=True)
def execute_move(cs, rs, n_nodes):
    cs_nodes = rs.pos2node[cs]
    for i in range(len(cs)):
        a = cs_nodes[(i - 1) % len(cs)]
        b = cs_nodes[i]
        c = cs_nodes[(i + 1) % len(cs)]
        
        if i % 2 == 1:
            if rs.pres[b] == a: rs.pres[b] = c
            if rs.sucs[b] == a: rs.sucs[b] = c
        else:
            if rs.pres[b] == c: rs.pres[b] = a
            if rs.sucs[b] == c: rs.sucs[b] = a
    
    max_cs = max(cs)
    min_cs = min(cs)
    
    last = rs.pos2node[(min_cs - 1) % n_nodes]
    curr = rs.pos2node[min_cs]
    
    for step in range(min_cs, max_cs + 1):
        # if step < min_cs:
        #     assert rs.pos2node[step] == curr
        
        rs.pos2node[step] = curr
        rs.node2pos[curr] = step
        
        if rs.pres[curr] != last:
            last = curr
            curr = rs.pres[curr]
        else:
            last = curr
            curr = rs.sucs[curr]
    
    # !! Have to walk big portion of the route anyway, so can compute the penalty function here.
    # !! Need to be able to roll things back if penalty gets rejected
    # !! Can start from earliest changed node but have to walk all the way to the end
    
    return rs


