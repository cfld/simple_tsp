#!/usr/bin/env python

"""
    simple_tsp/moves/rc.py
"""

import numpy as np
from numba import njit

from ..helpers import suc2cost
from ..constraints import cap
from ..execute import execute_relocate, reverse_relocate
from simple_tsp.utils import CostModel, EPS

@njit(cache=True)
def do_rc(
        dist,
        near, 
        
        node2pre,
        node2suc,
        node2route,
        node2depot, 
        
        n_nodes, 
        n_vehicles, 
        
        # @CONSTRAINT -- params
        cap__data,
        cap__maxval, 
        # <<
    ):
    
    changed = set([-1]); changed.clear()
    
    cost = suc2cost(node2suc, dist, n_vehicles)
    
    # >> @CONSTRAINT
    slacks = np.array([cap.route2slack(r, node2suc, cap__data, cap__maxval) for r in range(n_vehicles)])
    # <<
    
    improved = True
    while improved:
        improved = False
        for r0 in np.random.permutation(n_vehicles):
            n0 = node2suc[r0]
            while not node2depot[n0]:
                n0_pre = node2pre[n0]
                n0_suc = node2suc[n0]
                
                sav, gain  = _find_insert(n0, n0_pre, n0_suc, r0, dist, near, node2pre, node2suc, node2route, node2depot, n_nodes, n_vehicles, cap__data, cap__maxval, slacks)
                if sav > EPS:
                    improved = True
                    cost -= sav
                    
                    # >>
                    # VALIDATE
                    c = suc2cost(node2suc, dist, n_vehicles)
                    assert ((c - cost) ** 2) < EPS
                    
                    p = cap.route2pen(r0, node2suc, cap__data, cap__maxval)
                    assert p == 0
                    # <<
                    
                    break
                
                n0     = n0_suc
                n0_pre = node2pre[n0]
                n0_suc = node2suc[n0]
    
    return cost, 0

@njit(cache=True, inline=CostModel(4))
def _find_insert(n0, n0_pre, n0_suc, r0, dist, near, node2pre, node2suc, node2route, node2depot, n_nodes, n_vehicles, cap__data, cap__maxval, slacks, sav=0, go_deeper=2):
    savp = sav + (
        + dist[n0, n0_pre]
        + dist[n0, n0_suc]
        - dist[n0_pre, n0_suc]
    )
    
    for n1 in near[n0]:
        
        r1 = node2route[n1]
        if r0 == r1: continue # no connect same route -- necessary?
        
        for d1 in [1, -1]:
            forward1 = d1 == 1
            
            n1_neib = node2suc[n1] if forward1 else node2pre[n1]
            
            sav0 = savp + (
                + dist[n1, n1_neib]
                - dist[n0, n1]
                - dist[n0, n1_neib]
            )
            
            # exit now
            if sav0 < EPS: continue
            
            # >> @CONSTRAINT
            if slacks[r1] >= cap__data[n0] and slacks[r0] + cap__data[n0] >= 0:
            # <<
                execute_relocate(n0, n0_pre, n0_suc, n1, n1_neib, forward1, node2pre, node2suc, node2route)
                # >> @CONSTRAINT
                slacks[r0] += cap__data[n0]
                slacks[r1] -= cap__data[n0]
                # <<
                return sav0, 0
            
            # go deeper
            if go_deeper > 0:
                execute_relocate(n0, n0_pre, n0_suc, n1, n1_neib, forward1, node2pre, node2suc, node2route)
                slacks[r0] += cap__data[n0]
                slacks[r1] -= cap__data[n0]
                
                n2 = node2suc[r1]
                while not node2depot[n2]:
                    n2_pre = node2pre[n2]
                    n2_suc = node2suc[n2]
                    
                    # >> @CONSTRAINT
                    if slacks[r1] + cap__data[n2] >= 0:
                    # <<
                        sav1, gain = _find_insert(n2, n2_pre, n2_suc, r1, dist, near, node2pre, node2suc, node2route, node2depot, n_nodes, n_vehicles, cap__data, cap__maxval, slacks, sav=sav0, go_deeper=go_deeper - 1)
                        if sav1 > EPS:
                            return sav1, gain
                    
                    n2     = n2_suc
                    n2_pre = node2pre[n2]
                    n2_suc = node2suc[n2]
                
                reverse_relocate(n0, n0_pre, n0_suc, n1, n1_neib, forward1, node2pre, node2suc, node2route)
                # >> @CONSTRAINT
                slacks[r0] -= cap__data[n0]
                slacks[r1] += cap__data[n0]
                # <<

    
    return 0, 0