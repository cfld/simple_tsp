import numpy as np
from numba import njit

def knn_candidates(dist, n_cands, n_vehicles):
    big_val = 2 * dist.max()
    mask    = np.eye(dist.shape[0]) * big_val # can't be near self
    mask[:n_vehicles,:n_vehicles] = big_val   # depots can't be close
    
    cand_idx = np.argsort(dist + mask, axis=-1)
    cand_idx = cand_idx[:,:n_cands]
    return cand_idx


@njit(cache=True)
def routes2cost(dist, node2suc, n_vehicles):
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