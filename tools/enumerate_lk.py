import numpy as np
from tqdm import trange
from copy import deepcopy
from collections import defaultdict

def edge_exists(a, b, n):
    return (
        (a == b + 1) or 
        (b == a + 1) or 
        (a == 0 and b == n - 1) or
        (b == 0 and a == n - 1)
    )

def contains(q, db):
    return (q in db) or ((q[1], q[0]) in db)

def check_subtour(n_nodes, old, new, close):
    neibs = [[((i - 1) % n_nodes), (i + 1) % n_nodes] for i in range(n_nodes)]
    for (a, b) in old:
        neibs[a].remove(b)
        neibs[b].remove(a)
    
    for (a, b) in new:
        neibs[a].append(b)
        neibs[b].append(a)
    
    neibs[close[0]].append(close[1])
    neibs[close[1]].append(close[0])
    
    p = 0
    n = neibs[0][0]
    connected_nodes = 1
    while connected_nodes < n_nodes and n != 0:
        # print(neibs, n)
        if neibs[n][0] != p:
            p = n
            n = neibs[n][0]
        else:
            p = n
            n = neibs[n][1]
        
        connected_nodes += 1
    
    return (n != 0) or (connected_nodes < n_nodes)


n_nodes = 12
edges   = [(i, (i + 1) % n_nodes) for i in range(n_nodes)]

old0     = []
new0     = []

c1, c2 = edges[0]
old0.append((c1, c2))

touched0 = set()
touched0 |= set([c1, c2])

valid   = set()
invalid = set()

eq_valid = set()

for c3 in trange(n_nodes):
    if c3 == c1:                     continue
    if c3 == c2:                     continue
    # if c3 in touched0:               continue
    if edge_exists(c2, c3, n_nodes): continue
    if contains((c2, c3), new0):     continue
    
    for c4 in [(c3 - 1) % n_nodes, (c3 + 1) % n_nodes]:
        if c4 == c1:                 continue
        # if c4 in touched0:           continue
        if contains((c3, c4), old0): continue
        
        old1, new1 = deepcopy(old0), deepcopy(new0)
        touched1 = deepcopy(touched0)
        new1.append((c2, c3))
        old1.append((c3, c4))
        touched1 |= set([c3, c4])
        
        sig = tuple(np.argsort((c1, c2, c3, c4)))
        if not check_subtour(n_nodes, old1, new1, (c4, c1)):
            valid.add(sig)
        else:
            invalid.add(sig)
        
        for c5 in range(n_nodes):
            if c5 == c1:                     continue
            if c5 == c4:                     continue
            # if c5 in touched1:               continue
            if edge_exists(c4, c5, n_nodes): continue
            if contains((c4, c5), new1):     continue
            
            for c6 in [(c5 - 1) % n_nodes, (c5 + 1) % n_nodes]:
                if c6 == c1:                 continue
                # if c6 in touched1:           continue
                if contains((c5, c6), old1): continue
                
                old2, new2 = deepcopy(old1), deepcopy(new1)
                touched2 = deepcopy(touched1)
                new2.append((c4, c5))
                old2.append((c5, c6))
                touched2 |= set([c5, c6])
                
                sig = tuple(np.argsort((c1, c2, c3, c4, c5, c6)))
                if not check_subtour(n_nodes, old2, new2, (c6, c1)):
                    valid.add(sig)
                else:
                    invalid.add(sig)
                    
                for c7 in range(n_nodes):
                    if c7 == c1:                     continue
                    if c7 == c6:                     continue
                    # if c7 in touched2:               continue
                    if edge_exists(c6, c7, n_nodes): continue
                    if contains((c6, c7), new1):     continue
                    
                    for c8 in [(c7 - 1) % n_nodes, (c7 + 1) % n_nodes]:
                        if c8 == c1:                 continue
                        # if c8 in touched2:           continue
                        if contains((c7, c8), old2): continue
                        
                        old3, new3 = deepcopy(old2), deepcopy(new2)
                        touched3 = deepcopy(touched2)
                        new3.append((c6, c7))
                        old3.append((c7, c8))
                        touched3 |= set([c7, c8])
                        
                        sig = tuple(np.argsort((c1, c2, c3, c4, c5, c6, c7, c8)))
                        if not check_subtour(n_nodes, old3, new3, (c8, c1)):
                            valid.add(sig)
                        else:
                            invalid.add(sig)
                        
                        for c9 in range(n_nodes):
                            if c9 == c1:                     continue
                            if c9 == c8:                     continue
                            # if c9 in touched3:               continue
                            if edge_exists(c8, c9, n_nodes): continue
                            if contains((c8, c9), new1):     continue
                            
                            for c10 in [(c9 - 1) % n_nodes, (c9 + 1) % n_nodes]:
                                if c10 == c1:                 continue
                                # if c10 in touched3:           continue
                                if contains((c9, c10), old3): continue
                                
                                old4, new4 = deepcopy(old3), deepcopy(new3)
                                touched4 = deepcopy(touched3)
                                new4.append((c8, c9))
                                old4.append((c9, c10))
                                touched4 |= set([c9, c10])
                                
                                sig = tuple(np.argsort((c1, c2, c3, c4, c5, c6, c7, c8, c9, c10)))
                                if not check_subtour(n_nodes, old4, new4, (c10, c1)):
                                    valid.add(sig)
                                    # assert check((c1, c2, c3, c4, c5, c6, c7, c8, c9, c10))
                                else:
                                    invalid.add(sig)
                                    # assert not check((c1, c2, c3, c4, c5, c6, c7, c8, c9, c10))


assert len(valid & invalid) == 0

_2opt = [v for v in valid if len(v) == 4]
_3opt = [v for v in valid if len(v) == 6]
_4opt = [v for v in valid if len(v) == 8]
_5opt = [v for v in valid if len(v) == 10]

# for v in sorted(_2opt):
#     print(v)

# for v in sorted(_3opt):
#     print(v)

# for v in sorted(_4opt):
#     print(v)

# for v in sorted(_5opt):
#     print(v)

# --

from collections import defaultdict

def ddict():
    return defaultdict(ddict)

# def print_clause(tup, indent):
#     out = ' ' * indent
#     out += f'if cs[{tup[0]}] <= cs[{tup[1]}]:'
#     print(out)

def print_clause(tup, indent):
    out = ' ' * indent
    out += f'if(cs[{tup[0]}] <= cs[{tup[1]}]) {{'
    print(out)

def print_tree(tree, indent=0):
    if len(tree) > 0:
        for k in sorted(tree.keys()):
            print_clause(k, indent)
            print_tree(tree[k], indent + 2)
            print(' ' * indent, '}')
    else:
        print(' ' * indent, 'return true;')

def print_fn(sigs):
    tree = ddict()
    
    for v in sigs:
        keys = [v[i:i+2] for i in range(len(v) - 1)]
        
        tmp = tree
        for k in keys:
            tmp = tmp[k]
    
    print(f'bool check{len(sigs[0]) // 2}(Int* cs) {{')
    print_tree(tree, indent=2)
    print('\n  return false;')
    print('}')
    

print_fn(_5opt)
