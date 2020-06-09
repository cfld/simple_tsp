import numpy as np
from scipy.spatial.distance import pdist, squareform

# --
# Helpers

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
    
    return node2route, node2depot, node2suc, node2pre

def init_dist(n_vehicles, n_nodes):
    xy = np.random.uniform(0, 100, (n_nodes, 2))
    xy[:n_vehicles] = xy[0]
    dist = squareform(pdist(xy)).astype(np.int32)
    return dist

def route2cost():
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

def walk_routes(verbose=True):
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


def switch_depot(n, new_depot, dir=1):
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

def flip_route(depot):
    n = depot
    while True:
        nn = node2suc[n]
        node2suc[n] = node2pre[n]
        node2pre[n] = nn
        
        if node2depot[nn]:
            break

        n = nn


def execute_move(n00, n01, n10, n11):
    seg00, seg01, seg10, seg11 = [], [], [], []
    
    r0 = node2route[n00]
    r1 = node2route[n10]
    
    if node2suc[n00] != n01: flip_route(r0)    
    if node2suc[n10] != n11: flip_route(r1)
    
    switch_depot(n11, r0)
    node2suc[n00] = n11 if not node2depot[n11] else r0
    node2pre[n11 if not node2depot[n11] else r0] = n00
    
    switch_depot(n01, r1)
    node2suc[n10] = n01 if not node2depot[n01] else r1
    node2pre[n01 if not node2depot[n01] else r1] = n10


def compute_move():
    for n00 in range(n_nodes):
        for d0 in [1, -1]:
            n01 = node2suc[n00] if d0 == 1 else node2pre[n00]
            r0  = node2route[n00]
            
            for n10 in range(n_nodes):
                for d1 in [1, -1]:
                    n11 = node2suc[n10] if d1 == 1 else node2pre[n10]
                    r1  = node2route[n10]
                    
                    if r0 == r1: continue
                    if node2depot[n00] and node2depot[n11]: continue # can't connect depot to depot
                    if node2depot[n01] and node2depot[n10]: continue
                    
                    sav = (
                        + dist[n00, n01]
                        + dist[n10, n11]
                        - dist[n01, n10]
                        - dist[n11, n00]
                    )
                    
                    if sav > 0:
                        return (n00, n01, n10, n11), sav
    
    return None, 0

# --
# Run

_ = np.random.seed(123)

n_vehicles = 10
n_nodes    = 100

node2route, node2depot, node2suc, node2pre = init_routes(n_vehicles, n_nodes)

dist = init_dist(n_vehicles, n_nodes)

it = 0
while True:
    old_cost  = route2cost()
    move, sav = compute_move()
    if sav == 0: break
    execute_move(*move)
    new_cost = route2cost()
    print(it, sav, old_cost - sav, new_cost)
    it += 1