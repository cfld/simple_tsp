import numpy as np
from numba import njit

from simple_tsp.helpers import suc2cost
from simple_tsp.constraints import cap
from simple_tsp.execute import execute_ropt, reverse_ropt
from simple_tsp.utils import EPS


@njit(cache=True)
def do_ce(
        dist,
        near,
        
        node2pre,
        node2suc,
        node2route,
        node2depot,
        
        active,
        route2stale,
        
        n_nodes,
        n_vehicles,
        
        # >> @CONSTRAINT
        cap__data,
        cap__maxval,
        # <<
        
        improving_only=True,
        
        validate=False,
    ):

    cost = suc2cost(node2suc, dist, n_vehicles)
    
    # >> @CONSTRAINT -- init
    cap__acc0 = np.zeros((2, 2), dtype=np.int64) - 1
    pen = cap.routes2pen(node2suc, n_vehicles, cap__data, cap__maxval)
    # <<
    
    move0 = np.zeros((2, 4), dtype=np.int64) - 1
    
    improved = True
    while improved:
        improved = False

        active_nodes = np.where(active)[0]
        for n00 in np.random.permutation(active_nodes):
            
            for d0 in [1, -1]:
                forward0 = d0 == 1
                
                r0 = node2route[n00]
                
                n01 = node2suc[n00] if forward0 else node2pre[n00]
                
                move0[0] = (n00, n01, r0, np.int64(not forward0))
                
                # >> @CONSTRAINT
                cap__acc0[0] = (
                    cap.partial(n00,     forward0, node2suc, node2pre, node2depot, cap__data),
                    cap.partial(n01, not forward0, node2suc, node2pre, node2depot, cap__data),
                )
                # <<
                
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
                    
                    improving_only=improving_only
                )
                if (gain > 0) or (gain == 0 and sav > EPS):
                    improved = True
                    cost -= sav
                    pen  -= gain
                    
                    # if validate:
                    #     c = suc2cost(node2suc, dist, n_vehicles)
                    #     assert ((c - cost) ** 2) < EPS
                        
                    #     # >> @CONSTRAINT
                    #     p = cap.routes2pen(node2suc, n_vehicles, cap__data, cap__maxval)
                    #     assert p == pen
                    #     # <<
                    
                    route2stale[move0[0, 2]] = True
                    route2stale[move0[1, 2]] = True
    
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
            improving_only,
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
            
            if improving_only and sav0 < EPS: continue # improving moves -- optional
            
            move0[1] = (n10, n11, r1, np.int64(not forward1))
            
            # >> @CONSTRAINT
            cap__acc0[1] = (
                cap.partial(n10,     forward1, node2suc, node2pre, node2depot, cap__data),
                cap.partial(n11, not forward1, node2suc, node2pre, node2depot, cap__data),
            )
            gain0 = cap.compute_gain(cap__acc0, 1, cap__maxval)
            # <<
            
            execute_ropt(move0, 1, node2pre, node2suc, node2route, node2depot)
            
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
                execute_ropt(move1, 1, node2pre, node2suc, node2route, node2depot)
                return gain, sav
            else:
                reverse_ropt(move0, 1, node2pre, node2suc, node2route, node2depot)
    
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
            cap.partial(x00_,     xforward0, node2suc, node2pre, node2depot, cap__data),
            cap.partial(x01_, not xforward0, node2suc, node2pre, node2depot, cap__data),
        )
        
        for xd1 in [1, -1]: # xforward1
            xforward1 = xd1 == 1
            
            # Reset nodes for 0
            x00          = x00_
            x01          = x01_
            cap__acc1[0] = tmp0
            
            x10_ = n10
            x11_ = node2suc[x10_] if xforward1 else node2pre[x10_]
            tmp1 = (
                cap.partial(x10_,     xforward1, node2suc, node2pre, node2depot, cap__data),
                cap.partial(x11_, not xforward1, node2suc, node2pre, node2depot, cap__data),
            )
            
            while True: # x00
                
                # Reset nodes for 1
                x10 = x10_
                x11 = x11_
                cap__acc1[1] = tmp1
                
                while True: # x10
                    if cap__acc1[0, 1] + cap__acc1[1, 0] > cap__maxval: break # more pruning
                    if cap__acc1[0, 0] + cap__acc1[1, 1] <= cap__maxval:
                        gain1 = gain0 + cap.compute_gain(cap__acc1, 1, cap__maxval)
                        # gain1 = gain0 + cap.compute_gain2(cap__acc1, cap__maxval)
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
                    # cap.slide_node(cap__acc1[1], x10, xforward1, cap__data)
                    cap__acc1[1, 0] += cap__data[x10]
                    cap__acc1[1, 1] -= cap__data[x10]
                    
                
                if node2depot[x01]: break
                x00 = x01
                x01 = node2suc[x00] if xforward0 else node2pre[x00]
                
                # cap.slide_node(cap__acc1[0], x00, xforward0, cap__data)
                cap__acc1[0, 0] += cap__data[x00]
                cap__acc1[0, 1] -= cap__data[x00]
    
    return move1, 0, 0
    
    

