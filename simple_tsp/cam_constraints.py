import numpy as np
from numba import njit

@njit(cache=True)
def cap__add_node(node, before, node2suc, node2pre, node2depot, val):
    p = 0
    while not node2depot[node]:
        p += val[node]
        node = node2pre[node] if before else node2suc[node]
    
    return p


@njit(cache=True)
def cap__compute_gain(acc, depth, cap):
    p_new, p_old = 0, 0
    for i in range(depth + 1):
        pp = acc[i, 0] + acc[i, 1]
        if pp > cap: p_old += pp - cap
        
        pp = acc[i, 1] + acc[(i + 1) % (depth + 1), 0]
        if pp > cap: p_new += pp - cap
    
    gain = p_old - p_new
    return gain

@njit(cache=True)
def cap__route2pen(depot, node2suc, node2pen, cap):
    node = node2suc[depot]
    p = 0
    while node != depot:
        p += node2pen[node]
        node = node2suc[node]
    
    if p > cap:
        return p - cap
    else:
        return 0

@njit(cache=True)
def cap__route2slack(depot, node2suc, node2pen, cap):
    node = node2suc[depot]
    p = 0
    while node != depot:
        p += node2pen[node]
        node = node2suc[node]
    
    return cap - p

@njit(cache=True)
def cap__routes2pen(n_vehicles, node2suc, node2pen, cap):
    p = 0
    for r in range(n_vehicles):
        p += cap__route2pen(r, node2suc, node2pen, cap)
    
    return p