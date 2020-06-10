import numpy as np
from numba import njit

from .cam_helpers import routes2cost
from .cam_constraints import cap__add_node, cap__compute_gain, cap__routes2pen
from .cam import execute_move, reverse_move, flip_route

@njit(cache=True)
def cap__slide_node(acc_row, node, forward, vals):
    val = vals[node]
    acc_row[0] += val
    acc_row[1] -= val


@njit(cache=True)
def do_camce(
        dist,
        near,
        
        node2pre,
        node2suc,
        node2route,
        node2depot,
        
        n_nodes,
        n_vehicles,
        
        cap__data,
        cap__maxval,
    ):

    cost = routes2cost(dist, node2suc, n_vehicles)
    
    # >> @CONSTRAINT -- init
    pen = cap__routes2pen(n_vehicles, node2suc, cap__data, cap__maxval)
    # <<
    
    move0 = np.zeros((2, 4), dtype=np.int64) - 1
    cap__acc0 = np.zeros((2, 2), dtype=np.int64) - 1
    
    improved = True
    while improved:
        improved = False

        for n00 in range(n_nodes):
            for d0 in [1, -1]:
                forward0 = d0 == 1
                
                r0 = node2route[n00]
                
                n01 = node2suc[n00] if forward0 else node2pre[n00]
                
                move0[0] = (n00, n01, r0, np.int64(not forward0))
                
                cap__acc0[0] = (
                    cap__add_node(n00,     forward0, node2suc, node2pre, node2depot, cap__data),
                    cap__add_node(n01, not forward0, node2suc, node2pre, node2depot, cap__data),
                )
                
                gain, sav = _find_move0(
                    move0,
                    cap__acc0,

                    dist,
                    near,

                    node2pre,
                    node2suc,
                    node2route,
                    node2depot,

                    n_nodes,
                    n_vehicles,

                    cap__data,
                    cap__maxval,
                )
                if (gain > 0) or (gain == 0 and sav > 1e-5):
                    improved = True
                    cost -= sav
                    pen  -= gain
    
    return cost, pen

@njit(cache=True)
def _find_move0(
            move0,
            cap__acc0,

            dist,
            near,

            node2pre,
            node2suc,
            node2route,
            node2depot,

            n_nodes,
            n_vehicles,

            cap__data,
            cap__maxval,
    ):
    
    (n00, n01, r0, _) = move0[0]
    for n10 in near[n01]:
        if node2depot[n01] and node2depot[n10]: continue # no depot-depot
        
        r1 = node2route[n10]
        if r1 == r0: continue
        
        for d1 in [1, -1]:
            forward1 = d1 == 1
            
            n11 = node2suc[n10] if forward1 else node2pre[n10]
            if node2depot[n00] and node2depot[n11]: continue # no depot-depot
            
            sav0 = (
                + dist[n00, n01]
                + dist[n10, n11]
                - dist[n00, n11]
                - dist[n01, n10]
            )
            
            # if sav0 < 0: continue # improving moves -- optional
            
            move0[1] = (n10, n11, r1, np.int64(not forward1))
            
            cap__acc0[1] = (
                cap__add_node(n10,     forward1, node2suc, node2pre, node2depot, cap__data),
                cap__add_node(n11, not forward1, node2suc, node2pre, node2depot, cap__data),
            )
            
            gain0 = cap__compute_gain(cap__acc0, 1, cap__maxval)
            
            execute_move(move0, 1, node2pre, node2suc, node2route, node2depot)
            
            move1, gain, sav = _find_move1(
                gain0,
                sav0,
                move0,
                
                n00,
                n10,
                r0,
                r1,
                
                dist,
                near,
                
                node2pre,
                node2suc,
                node2route,
                node2depot,
                
                n_nodes,
                n_vehicles,
                
                cap__data,
                cap__maxval,
            )
            if (gain > 0) or (gain == 0 and sav > 0):
                execute_move(move1, 1, node2pre, node2suc, node2route, node2depot)
                return gain, sav
            else:
                reverse_move(move0, 1, node2pre, node2suc, node2route, node2depot)
    
    return 0, 0


@njit(cache=True)
def _find_move1(
        gain0,
        sav0,
        move0,
        
        n00,
        n10,
        r0,
        r1,

        dist,
        near,

        node2pre,
        node2suc,
        node2route,
        node2depot,

        n_nodes,
        n_vehicles,

        cap__data,
        cap__maxval,
    ):
    
    
    cap__acc1 = np.zeros((2, 2), dtype=np.int64) - 1
    
    for xd0 in [1, -1]: # xforward0
        xforward0 = xd0 == 1
        
        x00_ = n00
        x01_ = node2suc[x00_] if xforward0 else node2pre[x00_]
        tmp0 =  (
            cap__add_node(x00_,     xforward0, node2suc, node2pre, node2depot, cap__data),
            cap__add_node(x01_, not xforward0, node2suc, node2pre, node2depot, cap__data),
        )
        
        for xd1 in [1, -1]: # xforward1
            xforward1 = xd1 == 1
            
            x00          = x00_
            x01          = x01_
            cap__acc1[0] = tmp0
            
            x10_ = n10
            x11_ = node2suc[x10_] if xforward1 else node2pre[x10_]
            tmp1 = (
                cap__add_node(x10_,     xforward1, node2suc, node2pre, node2depot, cap__data),
                cap__add_node(x11_, not xforward1, node2suc, node2pre, node2depot, cap__data),
            )
            
            while True: # x00
                
                x10 = x10_
                x11 = x11_
                cap__acc1[1] = tmp1
                
                while True: # x10
                    gain1 = gain0 + cap__compute_gain(cap__acc1, 1, cap__maxval)
                    if gain1 >= 0:
                        sav1 = sav0 + (
                            + dist[x00, x01]
                            + dist[x10, x11]
                            - dist[x00, x11]
                            - dist[x01, x10]
                        )
                        if sav1 > 0:
                            move1 = np.array((
                                (x00, x01, r0, np.int64(not xforward0)),
                                (x10, x11, r1, np.int64(not xforward1)),
                            ), dtype=np.int64)
                            return move1, gain1, sav1
                    
                    if node2depot[x11]: break
                    x10 = x11
                    x11 = node2suc[x10] if xforward1 else node2pre[x10]
                    cap__slide_node(cap__acc1[1], x10, xforward1, cap__data)
                    
                    
                    if cap__acc1[0, 1] + cap__acc1[1, 0] > cap__maxval: break # more pruning
                
                if node2depot[x01]: break
                x00 = x01
                x01 = node2suc[x00] if xforward0 else node2pre[x00]
                cap__slide_node(cap__acc1[0], x00, xforward0, cap__data)
    
    return move1, 0, 0
    
    

