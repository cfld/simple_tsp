#!/usr/bin/env python

"""
    simple_tsp/prep.py
"""

import numpy as np
from numba import njit
from tsplib95.parser import parse as parse_tsplib
from scipy.spatial.distance import squareform, pdist

# --
# Load problem + compute distance matrix

def load_problem(inpath):
    return parse_tsplib(open(inpath).read())

def get_distance_matrix(prob, n_vehicles=1):
    edge_weight_type = prob['EDGE_WEIGHT_TYPE']
    
    assert edge_weight_type in ['EUC_2D', 'CEIL_2D', 'EXPLICIT']
    
    if edge_weight_type in ['EUC_2D', 'CEIL_2D']:
        xy = np.row_stack(list(prob['NODE_COORD_SECTION'].values()))
        
        if n_vehicles > 1:
            # !! Duplicate depot for extra vehicles
            xy = np.row_stack([
                np.repeat(xy[:1], n_vehicles, axis=0),
                xy[1:]
            ])
            assert xy.shape[0] == prob['DIMENSION'] + n_vehicles - 1
        
        dist = squareform(pdist(xy))
        
        # if edge_weight_type == 'EUC_2D':
        #     dist = np.round(dist).astype(np.int64)
        # elif edge_weight_type == 'CEIL_2D':
        #     dist = np.ceil(dist).astype(np.int64)
    
    elif edge_weight_type == 'EXPLICIT':
        assert n_vehicles == 1
        
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
    
    return dist

def load_solution(inpath):
    prob = parse_tsplib(open(inpath).read())
    
    tour = prob['TOUR_SECTION'][0]
    tour = np.array(tour + [1]) - 1
    return tour

# --
# Generate candidate edges

@njit(cache=True)
def knn_candidates(dist, n_cands, n_vehicles=1):
    # !! Could be rewritten w/ argpartition
    big_val = 2 * dist.max()
    mask    = np.eye(dist.shape[0]) * big_val # can't be near self
    mask[:n_vehicles,:n_vehicles] = big_val   # depots can't be close
    mdist = dist + mask
    
    cand_idx = np.zeros((dist.shape[0], n_cands), dtype=np.int64)
    for i in range(dist.shape[0]):
        cand_idx[i] = np.argsort(mdist[i])[:n_cands]
    
    return cand_idx
    
# --
# Generate initial route

def random_init(n_nodes, n_vehicles=1, random_state=None):
    rng = np.random if random_state is None else np.random.RandomState(random_state)
    
    pos2node = np.hstack([[0], 1 + rng.permutation(n_nodes - 1)])
    pos2node[pos2node < n_vehicles] = np.arange(n_vehicles) # sort depots
    return pos2node

# --
# Initialize datastructures

@njit(cache=True)
def route2lookups(pos2node, n_nodes, n_vehicles):
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