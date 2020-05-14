
import argparse
import numpy as np
from time import time

import numba
from numba import njit, prange

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n',         type=int, default=2000)
    parser.add_argument('--n-threads', type=int, default=1)
    return parser.parse_args()

args = parse_args()
numba.set_num_threads(args.n_threads)

@njit
def matmul(x):
    return (x @ x).sum()

@njit(parallel=True)
def p_matmul(n):
    x = np.ones((n, n))
    
    z = 0
    for _ in prange(100):
        z += matmul(x)
    
    return z

print('first run')
t = time()
_ = p_matmul(args.n)
print(time() - t)

print('second run')
t = time()
_ = p_matmul(args.n)
print(time() - t)