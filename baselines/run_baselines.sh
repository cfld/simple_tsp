#!/bin/bash

# run_baselines.py

# --
# Compare to https://github.com/dmishin/tsp-solver

INPATH="data/tsplib/a280.tsp"
python -m simple_tsp.main          --inpath $INPATH
python baselines/dmishin_solver.py --inpath $INPATH

INPATH="data/tsplib/d493.tsp"
python -m simple_tsp.main          --inpath $INPATH
python baselines/dmishin_solver.py --inpath $INPATH

INPATH="data/tsplib/kroC100.tsp"
python -m simple_tsp.main          --inpath $INPATH
python baselines/dmishin_solver.py --inpath $INPATH

INPATH="data/tsplib/pcb442.tsp"
python -m simple_tsp.main          --inpath $INPATH
python baselines/dmishin_solver.py --inpath $INPATH

# Haven't tested extensively, but `simple_lkh` appears to be faster/better