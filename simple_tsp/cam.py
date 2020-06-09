import numpy as np
from numba import njit
from scipy.spatial.distance import pdist, squareform

# --
# Helpers

def knn_candidates(dist, n_cands):
    tmp      = dist + (np.eye(dist.shape[0]) * dist.max()) # Can't be near self
    cand_idx = np.argsort(tmp, axis=-1)
    cand_idx = cand_idx[:,:n_cands]
    return cand_idx

def init_routes(n_vehicles, n_nodes):
    pos2node   = np.hstack([[0], 1 + np.random.permutation(n_nodes - 1)])
    pos2node[pos2node < n_vehicles] = np.arange(n_vehicles) # sort depots
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
    
    return node2route, node2depot, node2suc, node2pre, pos2node


def init_dist(n_vehicles, n_nodes):
    xy = np.random.uniform(0, 100, (n_nodes, 2))
    xy[:n_vehicles] = xy[0]
    dist = squareform(pdist(xy)).astype(np.int32)
    return dist

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
def execute_move(move, node2pre, node2suc, node2route, node2depot):
    n_moves = 0
    for i in range(move.shape[0]):
        if move[i, 0] == -1: break
        n_moves += 1
    
    rs = node2route[move[:,0]]
    
    # Flip routes
    for i in range(n_moves):
        n0, n1 = move[i]
        flip = node2suc[n0] != n1
        if flip:
            flip_route(rs[i], node2pre, node2suc, node2depot)
    
    # Splice routes
    for i in range(n_moves):
        j   = (i + 1) % n_moves
        n01 = move[i, 1]
        n10 = move[j, 0]
        change_edge(n01, n10, rs[j], node2pre, node2suc, node2route, node2depot)


@njit(cache=True)
def compute_move(dist, near, node2pre, node2suc, node2route, node2depot, n_nodes):
    move = np.zeros((3, 2), dtype=np.int64) - 1
    
    for n00 in range(n_nodes):
        for d0 in [1, -1]:
            r0  = node2route[n00]
            
            n01 = node2suc[n00] if d0 == 1 else node2pre[n00]
            
            n00_depot = node2depot[n00]
            n01_depot = node2depot[n01]
            
            for n10 in near[n01]:
                for d1 in [1, -1]:
                    r1  = node2route[n10]
                    if r0 == r1: continue

                    n11 = node2suc[n10] if d1 == 1 else node2pre[n10]
                    
                    n10_depot = node2depot[n10]
                    n11_depot = node2depot[n11]
                    if n01_depot and n10_depot: continue # no depot-depot
                    
                    # exit now
                    if not (n11_depot and n00_depot):
                            sav = (
                                + dist[n00, n01]
                                + dist[n10, n11]
                                
                                - dist[n01, n10]
                                - dist[n11, n00]
                            )
                            
                            if sav > 0:
                                move[0, 0] = n00
                                move[0, 1] = n01
                                move[1, 0] = n10
                                move[1, 1] = n11
                                return move, sav
                    
                    for n20 in near[n11]:
                        for d2 in [1, -1]:
                            r2 = node2route[n20]
                            if r2 == r0: continue
                            if r2 == r1: continue
                            
                            n21 = node2suc[n20] if d2 == 1 else node2pre[n20]
                            
                            n20_depot = node2depot[n20]
                            n21_depot = node2depot[n21]
                            if n11_depot and n20_depot: continue # no depot-depot
                            
                            if not (n21_depot and n00_depot):
                                sav = (
                                    + dist[n00, n01]
                                    + dist[n10, n11]
                                    + dist[n20, n21]
                                    
                                    - dist[n01, n10]
                                    - dist[n11, n20]
                                    - dist[n21, n00]
                                )
                        
                                if sav > 0:
                                    move[0, 0] = n00
                                    move[0, 1] = n01
                                    move[1, 0] = n10
                                    move[1, 1] = n11
                                    move[2, 0] = n20
                                    move[2, 1] = n21
                                    return move, sav
    
    return move, 0


