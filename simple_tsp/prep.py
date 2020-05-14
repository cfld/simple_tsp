#!/usr/bin/env python

"""
    simple_tsp/prep.py
"""

import numpy as np
from tsplib95.parser import parse as parse_tsplib
from scipy.spatial.distance import squareform, pdist

# --
# Load problem + compute distance matrix

def load_problem(inpath):
    prob = parse_tsplib(open(inpath).read())
    
    n_nodes          = prob['DIMENSION']
    edge_weight_type = prob['EDGE_WEIGHT_TYPE']
    
    assert edge_weight_type in ['EUC_2D', 'CEIL_2D']
    
    xy   = np.row_stack(list(prob['NODE_COORD_SECTION'].values()))
    dist = squareform(pdist(xy))
    
    if edge_weight_type == 'EUC_2D':
        dist = np.round(dist).astype(np.int64)
    elif edge_weight_type == 'CEIL_2D':
        dist = np.ceil(dist).astype(np.int64)
    
    return dist, n_nodes

# --
# Generate candidate edges

def knn_candidates(dist, n_cands):
    cand_idx = np.argsort(dist, axis=-1)
    cand_idx = cand_idx[:,1:n_cands + 1]
    
    return cand_idx

# --
# Generate initial route

def random_init(n_nodes):
    return np.hstack([[0], 1 + np.random.permutation(n_nodes - 1), [0]])