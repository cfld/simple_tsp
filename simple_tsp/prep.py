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
    
    assert edge_weight_type in ['EUC_2D', 'CEIL_2D', 'EXPLICIT']
    
    if edge_weight_type in ['EUC_2D', 'CEIL_2D']:
        xy   = np.row_stack(list(prob['NODE_COORD_SECTION'].values()))
        dist = squareform(pdist(xy))
        
        if edge_weight_type == 'EUC_2D':
            dist = np.round(dist).astype(np.int64)
        elif edge_weight_type == 'CEIL_2D':
            dist = np.ceil(dist).astype(np.int64)
    
    elif edge_weight_type == 'EXPLICIT':
        edge_weights       = prob['EDGE_WEIGHT_SECTION']
        edge_weight_format = prob['EDGE_WEIGHT_FORMAT']
        
        if edge_weight_format == 'LOWER_DIAG_ROW':
            dist = np.zeros((n_nodes, n_nodes), dtype=np.int64)
            cols = np.hstack([np.arange(i + 1) for i in range(n_nodes)])
            rows = np.hstack([np.repeat(i, i + 1) for i in range(n_nodes)])
            dist[(rows, cols)] = edge_weights
            dist = dist + dist.T
            
        elif edge_weight_format == 'UPPER_DIAG_ROW':
            dist = np.zeros((n_nodes, n_nodes), dtype=np.int64)
            cols = np.hstack([np.arange(i, n_nodes) for i in range(n_nodes)])
            rows = np.hstack([np.repeat(i, n_nodes - i) for i in range(n_nodes)])
            dist[(rows, cols)] = edge_weights
            dist = dist + dist.T
            
        elif edge_weight_format == 'FULL_MATRIX':
            dist = np.array(edge_weights).reshape(n_nodes, n_nodes).astype(np.int64)
        
        elif edge_weight_format in ['UPPER_ROW', 'FUNCTION']:
            raise Exception('!! No parser for `EDGE_WEIGHT_FORMAT` in [`UPPER_ROW`, `FUNCTION`]')
    
    return dist, n_nodes

def load_solution(inpath):
    prob = parse_tsplib(open(inpath).read())
    
    tour = prob['TOUR_SECTION'][0]
    tour = np.array(tour + [1]) - 1
    return tour

# --
# Generate candidate edges

def knn_candidates(dist, n_cands):
    cand_idx = np.argsort(dist, axis=-1)
    cand_idx = cand_idx[:,1:n_cands + 1]
    return cand_idx

# --
# Generate initial route

def random_init(n_nodes):
    return np.hstack([[0], 1 + np.random.permutation(n_nodes - 1)])

