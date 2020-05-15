#!/usr/bin/env python

"""
    baselines/dmishin_solver.py
    
    https://github.com/dmishin/tsp-solver
"""

import json
import argparse
import numpy as np
from time import time
from random import random, seed
from scipy.spatial.distance import squareform, pdist

from tsp_solver import greedy

from simple_tsp.prep import load_problem, load_solution

np.random.seed(123)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default='data/tsplib/kroC100.tsp')
    return parser.parse_args()

args = parse_args()
dist, n_nodes = load_problem(args.inpath)

# --
# Run

t       = time()
path    = greedy.solve_tsp(dist, optim_steps=10)
elapsed = time() - t

# --
# Eval

cost = 0
for i in range(len(path)):
    cost += dist[path[i], path[(i + 1) % n_nodes]]

print(json.dumps({
    'cost'    : int(cost),
    'elapsed' : float(elapsed)
}))