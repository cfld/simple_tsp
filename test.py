import numpy as np
from scipy.spatial.distance import pdist, squareform

# --
# Helpers

def init_routes(n_vehicles, n_nodes):
    pos2node   = np.hstack([[0], 1 + np.random.permutation(n_nodes - 1)])
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
    for depot in range(n_vehicles):
        n = depot
        while True:
            print(n)
            cost += dist[n, node2suc[n]]
            n = node2suc[n]
            if n == depot: break
        
        print('-')
    
    return cost

def walk_routes(node2suc):
    routes = []
    for depot in range(n_vehicles):
        route = []
        node = depot
        while True:
            route.append(node)
            node = node2suc[node]
            if node == depot: break
        
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

_ = np.random.seed(123)

n_vehicles = 2
n_nodes    = 16

node2route, node2depot, node2suc, node2pre = init_routes(n_vehicles, n_nodes)
dist = init_dist(n_vehicles, n_nodes)

print(walk_routes(node2suc))

execute_move(11, 8, 9, 12)
print(walk_routes(node2suc))

raise Exception()
# <<




def _move(cost):
    for n00 in range(n_nodes):
        for d0 in [-1]:
            n01 = node2suc[n00] if d0 == 1 else node2pre[n00]
            r0  = node2route[n00]
            
            for n10 in range(n_nodes):
                for d1 in [1]:
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
                        print('est new_cost', cost - sav)
                        print(n00, n01, n10, n11)
                        
                        if d0 == 1 and d1 == 1: # forward-forward
                            raise Exception()
                            # node2suc[n00] = n11 ; node2pre[n11] = n00
                            # node2suc[n10] = n01 ; node2pre[n01] = n10
                            
                            # n = n11
                            # while True:
                            #     node2route[n] = r0
                            #     if node2depot[node2suc[n]]:
                            #         node2suc[n] = r0
                            #         break
                                
                            #     n = node2suc[n]
                            
                            # n = n01
                            # while True:
                            #     node2route[n] = r1
                            #     if node2depot[node2suc[n]]:
                            #         node2suc[n] = r1
                            #         break
                                
                            #     n = node2suc[n]
                        
                        elif d0 == -1 and d1 == 1: # backward-forward
                            node2suc[n00] = n11 ; node2pre[n11] = n00
                            
                            n = n11
                            while True:
                                print('n11 walk', n)
                                node2route[n] = r0
                                if node2depot[node2suc[n]]:
                                    node2suc[n] = r0
                                    break
                                
                                n = node2suc[n]
                            
                            n = n00
                            while True:
                                print('n00 walk', n, node2suc[n], node2pre[n])
                                
                                if node2depot[node2pre[n]]:
                                    print('-')
                                    break
                               
                                tmp         = node2suc[n]
                                node2suc[n] = node2pre[n]
                                node2pre[n] = tmp
                                
                                n = node2suc[n]

                            node2suc[n10] = n01 ; node2pre[n01] = n10
                            
                            n = n01
                            while True:
                                print('n01 walk', n, node2suc[n], node2pre[n])
                                node2route[n] = r1
                                if node2depot[node2pre[n]]:
                                    node2pre[n] = r1
                                    break
                                
                                n = node2pre[n]
                            
                            raise Exception()
                        
                        return

print_route()
_move(0)
print_route()
# _move(0)

# for _ in range(2):
#     # cost = route2cost()
#     # print('pre cost', cost)
#     _move(0)
#     # print('act new_cost', route2cost())
#     # print('-' * 50)

# print_route()