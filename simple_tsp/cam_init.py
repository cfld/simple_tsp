import numpy as np
from numba import njit

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
    
    return node2pre, node2suc, node2route, node2depot, pos2route