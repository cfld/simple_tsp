#!/bin/bash

# run.sh

# --
# Setup environment

conda create -y -n simple_tsp_env python=3.7
conda activate simple_tsp_env

conda install -y scipy
conda install -y numba
pip install git+https://github.com/bkj/tsplib95

# --

pip install -e .