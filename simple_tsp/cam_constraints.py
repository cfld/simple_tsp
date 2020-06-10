import numpy as np
from numba import njit

@njit(cache=True)
def partial_load(node, before, node2suc, node2pre, node2depot, val):
    p = 0
    while not node2depot[node]:
        p += val[node]
        node = node2pre[node] if before else node2suc[node]
    
    return p


@njit(cache=True)
def route_load(depot, node2suc, node2pen):
    node = node2suc[depot]
    p = 0
    while node != depot:
        p += node2pen[node]
        node = node2suc[node]
    
    return p


@njit(cache=True)
def all_pen(n_vehicles, node2suc, node2pen, cap):
    p = 0
    for r in range(n_vehicles):
        pp = route_load(r, node2suc, node2pen)
        if pp > cap:
            p += pp - cap
    
    return p

@njit(cache=True)
def le_additive_gain(acc, depth, maxval):
    p_new, p_old = 0, 0
    for i in range(depth + 1):
        pp = acc[i, 0] + acc[i, 1]
        if pp > maxval: p_old += pp - maxval
        
        pp = acc[i, 1] + acc[(i + 1) % (depth + 1), 0]
        if pp > maxval: p_new += pp - maxval
    
    gain = p_old - p_new
    return gain


@njit(cache=True)
def le_additive_break_edge(n0, n1, forward, depth, acc, val, node2suc, node2pre, node2depot):
    acc[depth, 0] = partial_load(n0, forward, node2suc, node2pre, node2depot, val)
    acc[depth, 1] = partial_load(n1, not forward, node2suc, node2pre, node2depot, val)
    return acc