# simple_tsp

An implementation of a Lin-Kernighan-style local search for traveling salesmen problems (TSPs).

Written in Python + Numba.

## Motivation

There are a handful of high-performance, high-quality solvers for routing problems (eg, [Concorde](http://www.math.uwaterloo.ca/tsp/concorde.html) and [LKH](http://akira.ruc.dk/~keld/research/LKH-3/).  However, good performance in this domain comes at the expense of complexity.  This project was motivated by the need for a reasonably-high-performance, reasonable-high-quality solver that can be used as a starting point for further experimentation.

Another motivation -- there has been an increasing amount of literature exploring the intersection of combinatorial optimization and machine learning.  Typically, some component (or even all) of the classical solver is replaced w/ a learned component.  Most classical solvers are implemented in C/C++, but most ML research is done in the Python ecosystem , which potentially makes rapid experimentation difficult.  Having a "pretty good" Python solver may lower the barrier of entry for ML people interested in routing problems.

## Installation

See `./install.sh` for installation

## Usage

See `./run.sh` for example runs.

__Note:__ `numba` compiles functions the first time you run the code, so the first run may appear to hang for ~ 1 minute.  Compiled functions are cached, so subsequent runs will be much faster.

## Notes

- `simple_tsp` is _slower_ than state-of-the-art solvers (eg. [LKH-3](http://akira.ruc.dk/~keld/research/LKH-3/)), and lacks many of the bells and whistles that are necessary to get _optimal_ performance.  However, `simple_tsp` achieves pretty good performance despite being substantially simpler (< 500 lines of code vs > 10K for LKH-3).  If you want to solve _a_ TSP, use Concorde or LKH-3 -- if you want to experiment w/ solvers, `simple_tsp` might be a good place to start.
- See `dev/parallel` branch for an implementation that parallelizes over perturbations.

## Known Issues

- [ ] The `execute_move` function is suboptimally implemented -- per the LKH-3 publication, routes can be modified more efficiently using a linked-list or double-linked-list data structure.
- [ ] LKH-3 allows for a series of improving k-opt moves that don't yield an improving solution when closed.  Currently, `simple_tsp` only allows a single k-opt move.
- [ ] These kinds of local search algorithms are typically run inside of some kind of global optimization metaheuristic (genetic algorithm, guided local search, simulated annealing, etc).  Currently `simple_tsp` just uses a perturbation + random restart.
