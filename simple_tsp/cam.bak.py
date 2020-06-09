from numba import njit

@njit(cache=True)
def edge_exists(u, v, n):
    return (
        (u == v + 1) or 
        (v == u + 1) or 
        (u == 0 and v == n - 1) or
        (v == 0 and u == n - 1)
    )

@njit
def cam(p00, p01, near, dist, rs, n_nodes, max_depth, dlb, demand, cap, counter):    
    n00  = rs.pos2node[p00]
    n01  = rs.pos2node[p01]
    
    fwd0 = p01 > p00
    if fwd0:
        r0 = rs.pos2route[p00] # edge belongs to same route as earlier node
    else:
        r0 = rs.pos2route[p01]
    
    for p10 in rs.node2pos[near[n01]]:
        if edge_exists(p01, p10, n_nodes): continue
        
        n10 = rs.pos2node[p10]
        
        for fwd1 in [True, False]:
            
            if fwd1:
                p11 = p10 + 1
                r1  = rs.pos2route[p10]
            else:
                p11 = p10 - 1
                r1  = rs.pos2route[p11]
            
            if r0 == r1: continue   # Edges can't be in same route
            
            n11 = rs.pos2node[p11]
            if n00 == n11: continue # skip for now -- though this is actually valid move (cat two routes)
            
            n_nodes = len(set([n00, n01, n10, n11])) # check 4 distinct nodes
            assert n_nodes == 4
            
            sav = (
                + dist[n00, n01]
                + dist[n10, n11]
                - dist[n01, n10]
                - dist[n11, n00]
            )
                        
            if sav > 0:
                # execute move
                
        