
import numpy as np
from numba import njit

# --
# Generic

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

# --
# Add edge

@njit(cache=True)
def add_edge(n0, n1, r, node2pre, node2suc, node2route, node2depot):
    switch_depot(n0, r, node2pre, node2suc, node2route, node2depot)
    n1_suc           = n0 if not node2depot[n0] else r
    node2suc[n1]     = n1_suc
    node2pre[n1_suc] = n1

# --
# Flip-flops

@njit(cache=True)
def execute_ropt(move, depth, node2pre, node2suc, node2route, node2depot):
    n_moves = depth + 1
    
    # Flip routes
    for i in range(n_moves):
        n0, n1, r, flip = move[i]
        if flip:
            flip_route(r, node2pre, node2suc, node2depot)
    
    # Change edges
    for i in range(n_moves):
        j   = (i + 1) % n_moves
        n01 = move[i, 1]
        n10 = move[j, 0]
        r   = move[j, 2]
        change_edge(n01, n10, r, node2pre, node2suc, node2route, node2depot)


@njit(cache=True)
def reverse_ropt(move, depth, node2pre, node2suc, node2route, node2depot):
    n_moves = depth + 1
    
    # REVERSE Flip routes
    for i in range(n_moves):
        n0, n1, r, flip = move[i]
        if flip == 0:
            change_edge(n1, n0, r, node2pre, node2suc, node2route, node2depot)
        else:
            change_edge(n1, n0, r, node2pre, node2suc, node2route, node2depot)
            flip_route(r, node2pre, node2suc, node2depot)

# --
# Relocate

@njit(cache=True, inline='always')
def execute_relocate(n0, n0_pre, n0_suc, n1, n1_neib, forward1, node2pre, node2suc, node2route):
    node2suc[n0_pre] = n0_suc
    node2pre[n0_suc] = n0_pre
    
    node2route[n0] = node2route[n1]
    if forward1:
        node2suc[n1] = n0
        node2pre[n0] = n1
        
        node2suc[n0] = n1_neib
        node2pre[n1_neib] = n0
    else:
        node2suc[n1_neib] = n0
        node2pre[n0] = n1_neib
        
        node2suc[n0] = n1
        node2pre[n1] = n0