import numba
from typing import NamedTuple

import numpy as np
from numba import njit
from scipy.spatial.distance import pdist, squareform

# --
# Helpers

def knn_candidates(dist, n_cands, n_vehicles):
    big_val = 2 * dist.max()
    mask    = np.eye(dist.shape[0]) * big_val # can't be near self
    mask[:n_vehicles,:n_vehicles] = big_val   # depots can't be close
    
    cand_idx = np.argsort(dist + mask, axis=-1)
    cand_idx = cand_idx[:,:n_cands]
    return cand_idx


@njit(cache=True)
def route2cost(n_vehicles, node2suc, dist):
    cost = 0
    counter = 0
    for depot in range(n_vehicles):
        n = depot
        while True:
            cost += dist[n, node2suc[n]]
            n = node2suc[n]
            if n == depot: break
            counter += 1
            if counter > node2suc.shape[0]: raise Exception('!! loop detected')
    
    return cost


def walk_routes(n_vehicles, node2suc, verbose=False):
    if verbose: print('-' * 50)
    routes = []
    counter = 0
    for depot in range(n_vehicles):
        route = []
        node = depot
        while True:
            if verbose: print(node)
            
            route.append(node)
            node = node2suc[node]
            if node == depot: break
            counter += 1
            if counter > len(node2suc) + 2: raise Exception('!! loop detected')
        
        if verbose: print('-' * 10)
        routes.append(route)
        
    return routes

# --
# Constraints

@njit(cache=True, inline='always')
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

@njit(cache=True, inline='always')
def le_additive_gain(acc, depth, maxval):
    p_new, p_old = 0, 0
    for i in range(depth + 1):
        pp = acc[i, 0] + acc[i, 1]
        if pp > maxval: p_old += pp - maxval
        
        pp = acc[i, 1] + acc[(i + 1) % (depth + 1), 0]
        if pp > maxval: p_new += pp - maxval
    
    gain = p_old - p_new
    return gain


@njit(cache=True, inline='always')
def le_additive_break_edge(n0, n1, forward, depth, acc, val, node2suc, node2pre, node2depot):
    acc[depth, 0] = partial_load(n0, forward, node2suc, node2pre, node2depot, val)
    acc[depth, 1] = partial_load(n1, not forward, node2suc, node2pre, node2depot, val)
    return acc

# --
# Initialization

@njit(cache=True)
def random_pos2node(n_vehicles, n_nodes):
    pos2node = np.hstack((np.array([0]), 1 + np.random.permutation(n_nodes - 1)))
    pos2node[pos2node < n_vehicles] = np.arange(n_vehicles) # sort depots
    return pos2node

@njit(cache=True)
def init_routes(pos2node, n_vehicles, n_nodes):
    node2pos   = np.argsort(pos2node)
    pos2depot  = (pos2node < n_vehicles).astype(np.int32)
    pos2route  = np.cumsum(pos2depot) - 1
    node2route = pos2route[node2pos]
    node2depot = (np.arange(n_nodes) < n_vehicles).astype(np.int32)
    
    node2suc = np.zeros(n_nodes, dtype=np.int32) - 1
    node2pre = np.zeros(n_nodes, dtype=np.int32) - 1
    for depot in range(n_vehicles):
        node = depot
        suc  = pos2node[(node2pos[node] + 1) % n_nodes]
        while not node2depot[suc]:
            node2suc[node] = suc
            node2pre[suc]  = node
            node = suc
            suc  = pos2node[(node2pos[node] + 1) % n_nodes]
        
        node2suc[node] = depot
        node2pre[depot] = node
    
    return node2pre, node2suc, node2route, node2depot, pos2node

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
                        
            loads = le_additive_break_edge(nd0, nd1, d1 == 1, depth, loads, node2pen, node2suc, node2pre, node2depot)
            # loads[depth, 0] = partial_load(nd0, d1 == 1, node2suc, node2pre, node2depot, node2pen)
            # loads[depth, 1] = partial_load(nd1, d1 != 1, node2suc, node2pre, node2depot, node2pen)
            
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

# # --
# # Hardcoded depth

# @njit(cache=True)
# def _cam3(move, dist, near, node2pre, node2suc, node2route, node2depot, n_nodes):
#     n00 = move[0, 0]
#     n01 = move[0, 1]
#     r0  = move[0, 2]
    
#     n00_depot = node2depot[n00]
#     n01_depot = node2depot[n01]
    
#     for n10 in near[n01]:
#         for d1 in [1, -1]:
#             r1  = node2route[n10]
#             if r0 == r1: continue

#             n11 = node2suc[n10] if d1 == 1 else node2pre[n10]
            
#             n10_depot = node2depot[n10]
#             n11_depot = node2depot[n11]
#             if n01_depot and n10_depot: continue # no depot-depot
            
#             sav0 = dist[n00, n01] + dist[n10, n11] - dist[n01, n10]
            
#             # exit now
#             if not (n11_depot and n00_depot):
#                     sav_close = sav0 - dist[n11, n00]
#                     if sav_close > 0:
#                         move[1, 0] = n10
#                         move[1, 1] = n11
#                         move[1, 2] = r1
#                         return move, 1, sav_close
            
#             for n20 in near[n11]:
#                 for d2 in [1, -1]:
#                     r2 = node2route[n20]
#                     if r2 == r0: continue
#                     if r2 == r1: continue
                    
#                     n21 = node2suc[n20] if d2 == 1 else node2pre[n20]
                    
#                     n20_depot = node2depot[n20]
#                     n21_depot = node2depot[n21]
#                     if n11_depot and n20_depot: continue # no depot-depot
                    
#                     sav1 = sav0 + dist[n20, n21] - dist[n11, n20]
                    
#                     if not (n21_depot and n00_depot):
#                         sav_close = sav1 - dist[n21, n00]
#                         if sav_close > 0:
#                             move[0, 0] = n00
#                             move[0, 1] = n01
#                             move[0, 2] = r0
#                             move[1, 0] = n10
#                             move[1, 1] = n11
#                             move[1, 2] = r1
#                             move[2, 0] = n20
#                             move[2, 1] = n21
#                             move[2, 2] = r2
#                             return move, 2, sav_close
    
#     return move, -1, 0

# @njit(cache=True)
# def do_cam3(dist, near, node2pre, node2suc, node2route, node2depot, n_nodes):
#     move = np.zeros((3, 3), dtype=np.int64) - 1
#     improved = True
#     while improved:
#         improved = False
#         for n00 in range(n_nodes):
#             for d0 in [1, -1]:
#                 r0  = node2route[n00]
#                 n01 = node2suc[n00] if d0 == 1 else node2pre[n00]
                
#                 move[0, 0] = n00
#                 move[0, 1] = n01
#                 move[0, 2] = r0
                
#                 move, depth, sav = _cam3(
#                     move, 
#                     dist, 
#                     near, 
#                     node2pre, 
#                     node2suc, 
#                     node2route, 
#                     node2depot, 
#                     n_nodes
#                 )
                
#                 if sav > 0:
#                     execute_move(move, depth, node2pre, node2suc, node2route, node2depot)
#                     improved = True
