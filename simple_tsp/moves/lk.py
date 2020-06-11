#!/usr/bin/env python

"""
    simple_tsp/lk.py
"""

import numpy as np
from numba import njit

from simple_tsp.lk_checkers import check_move

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
    return ((u, v) in db) or ((v, u) in db)

# --
# Optimize route

@njit(cache=True)
def lk_solve(dist, near, route, depth=5, lk_neibs=10, use_dlb=False):
    
    assert route[0] == 0, 'route[0] != 0'
    assert route[-1] == 0, 'route[-1] != 0'
    
    route   = route[:-1]
    n_nodes = len(route)
    
    pos2node = route.copy()
    node2pos = np.argsort(pos2node)
    sucs     = np.array([pos2node[(node2pos[i] + 1) % n_nodes] for i in range(n_nodes)])
    pres     = np.array([pos2node[(node2pos[i] - 1) % n_nodes] for i in range(n_nodes)])
    
    near = near[:,:lk_neibs]
    
    offset        = 0
    ever_improved = False
    iter_improved = True
    
    dlb = np.zeros(n_nodes)
    
    while iter_improved:
        iter_improved = False
        
        if use_dlb:
            dlb.fill(0)
        
        for _ in range(n_nodes):
            
            c1 = (offset) % n_nodes
            c2 = (offset + 1) % n_nodes
            offset += 1
            
            sav, cs = lk_move(near, dist, pos2node, node2pos, c1, c2, n_nodes, depth, dlb)
            
            if sav > 0:
                iter_improved = True
                ever_improved = True
                pos2node, node2pos, pres, sucs = execute_move(cs, pos2node, node2pos, pres, sucs, n_nodes)
                break
            
            sav, cs = lk_move(near, dist, pos2node, node2pos, c2, c1, n_nodes, depth, dlb)
            
            if sav > 0:
                iter_improved = True
                ever_improved = True
                pos2node, node2pos, pres, sucs = execute_move(cs, pos2node, node2pos, pres, sucs, n_nodes)
                break
            
            if use_dlb:
                dlb[c1] = 1
    
    return np.hstack((pos2node, np.array([0])))

# --
# Find best move

@njit(cache=True)
def lk_move(neibs, dist, pos2node, node2pos, c1, c2, n_nodes, max_depth, dlb):
    
    cs  = np.zeros(2 * max_depth, dtype=np.int64) - 1
    csh = np.zeros(2 * max_depth, dtype=np.int64) - 1
    old = [(-1, -1) for _ in range(max_depth)]
    new = [(-1, -1) for _ in range(max_depth - 1)]
    
    cs[0]  = c1
    cs[1]  = c2
    csh[0] = 0
    csh[1] = 1
    old[0] = (c1, c2)
    
    sav = dist[pos2node[c1], pos2node[c2]] # remove 12
    
    return _lk_move(neibs, dist, pos2node, node2pos, cs, csh, new, old, sav, n_nodes, 0, max_depth, dlb)


class CostModel:
    def __init__(self, n):
        self._i = 0
        self._n = n

    def __call__(self, expr, caller, callee):
        ret = self._i < self._n
        self._i += 1
        return ret

@njit(cache=True, inline=CostModel(5))
def _lk_move(neibs, dist, pos2node, node2pos, cs, csh, new, old, saving, n_nodes, depth, max_depth, dlb):
    fin = cs[0]
    act = cs[2 * depth + 1] # positions
    rev = (cs[1] - fin) % n_nodes == 1
    
    for cp1 in node2pos[neibs[pos2node[act]]]:
        
        if dlb[cp1] == 1:                  continue
        if cp1 == -1:                      continue
        if cp1 == fin:                     continue
        if cp1 == act:                     continue
        if edge_exists(act, cp1, n_nodes): continue
        if contains(act, cp1, new):        continue
        
        saving_n23 = saving - dist[pos2node[act], pos2node[cp1]] # add 23
        
        if saving_n23 > 0:
            
            for cp2 in [(cp1 - 1) % n_nodes, (cp1 + 1) % n_nodes]:
                if cp2 == -1:               continue # impossible
                if cp2 == fin:              continue
                if contains(cp1, cp2, old): continue
                
                new[depth]               = (act, cp1)
                old[depth + 1]           = (cp1, cp2)
                cs[2 * (depth + 1)]      = cp1
                cs[2 * (depth + 1) + 1]  = cp2
                csh[2 * (depth + 1)]     = (cp1 - fin) % n_nodes if rev else abs(((cp1 - fin) % n_nodes) - n_nodes) % n_nodes
                csh[2 * (depth + 1) + 1] = (cp2 - fin) % n_nodes if rev else abs(((cp2 - fin) % n_nodes) - n_nodes) % n_nodes
                
                saving_n23_o34 = saving_n23 + dist[pos2node[cp1], pos2node[cp2]] # remove 34
                
                # exit now
                if not edge_exists(cp2, fin, n_nodes):
                    saving_closed = saving_n23_o34 - dist[pos2node[cp2], pos2node[fin]]
                    if saving_closed > 0:
                        if check_move(csh[:2 * (depth + 2)], n_nodes):
                            return saving_closed, cs[:2 * (depth + 2)]
                
                # search deeper
                if depth < max_depth - 2:
                    deeper_saving, deeper_cs = _lk_move(neibs, dist, pos2node, node2pos, cs, csh, new, old, saving_n23_o34, n_nodes, depth + 1, max_depth, dlb)
                    
                    if deeper_saving > 0:
                        return deeper_saving, deeper_cs
    
    return -1, cs

# --
# Execute move

@njit(cache=True)
def execute_move(cs, pos2node, node2pos, pres, sucs, n_nodes):
    cs_nodes = pos2node[cs]
    for i in range(len(cs)):
        a = cs_nodes[(i - 1) % len(cs)]
        b = cs_nodes[i]
        c = cs_nodes[(i + 1) % len(cs)]
        
        if i % 2 == 1:
            if pres[b] == a: pres[b] = c
            if sucs[b] == a: sucs[b] = c
        else:
            if pres[b] == c: pres[b] = a
            if sucs[b] == c: sucs[b] = a
    
    max_cs = max(cs)
    min_cs = min(cs)
    
    last = pos2node[(min_cs - 1) % n_nodes]
    curr = pos2node[min_cs]
    
    for step in range(min_cs, max_cs + 1):
        if step < min_cs:
            assert pos2node[step] == curr
        
        pos2node[step] = curr
        node2pos[curr] = step
        
        if pres[curr] != last:
            last = curr
            curr = pres[curr]
        else:
            last = curr
            curr = sucs[curr]
    
    return pos2node, node2pos, pres, sucs
