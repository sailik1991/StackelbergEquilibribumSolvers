# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains solvers for computing Stackelberg equilibria in security games. The code implements MIQP (Mixed Integer Quadratic Programming) and MILP (Mixed Integer Linear Programming) formulations for defender-attacker games, originally developed for research papers at AAMAS 2016-2017.

## Requirements

### Gurobi Solvers
- **Gurobi Optimizer**: Required for `BSG_miqp.py`, `BSG_milp.py`, and `whatToFix.py` (http://www.gurobi.com)
- Python packages: `gurobipy`, `numpy`, `networkx`

### OR-Tools Solvers (Gurobi-free alternative)
- **Google OR-Tools**: Required for `BSG_miqp_ortools.py`, `whatToFix_ortools.py`, and `cost_BSG_miqp_ortools.py`
- Python packages: `ortools`

## Virtual Environment Setup

To set up and use the virtual environment for OR-Tools solvers:

```bash
# From the repository root, create the virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install OR-Tools
pip install ortools

# To deactivate when done
deactivate
```

## Running Tests

Tests for the OR-Tools solvers are located in `src/DOBSS/tests/` and `src/switch_cost_DOBSS/tests/`.

```bash
# From the repository root, activate the virtual environment
source .venv/bin/activate

# Run all DOBSS tests
python3 -m unittest discover src/DOBSS/tests/ -v

# Run all switch cost DOBSS tests
python3 -m unittest discover src/switch_cost_DOBSS/tests/ -v

# Run a specific test file
python3 src/DOBSS/tests/test_bsg_miqp_ortools.py
python3 src/DOBSS/tests/test_whatToFix_ortools.py
python3 src/switch_cost_DOBSS/tests/test_cost_bsg_miqp_ortools.py
```

## Running the Solvers

### OR-Tools Solvers (recommended)

```bash
# Activate virtual environment first
source .venv/bin/activate

# DOBSS solver
python3 src/DOBSS/BSG_miqp_ortools.py src/DOBSS/input.txt

# WhatToFix solver (finds optimal attack to remove)
python3 src/DOBSS/whatToFix_ortools.py src/DOBSS/input.txt
```

### Gurobi Solvers

Gurobi solvers are run via `gurobi.sh` (the Gurobi Python shell):

```bash
cd src/DOBSS
gurobi.sh BSG_miqp.py <input_file>
# Or use the convenience script:
./run.sh
```

### DOBSS with Switching Costs

OR-Tools version (recommended):
```bash
python3 src/switch_cost_DOBSS/cost_BSG_miqp_ortools.py src/switch_cost_DOBSS/cost_BSSG_input.txt <alpha>
# alpha is the switching cost weight parameter (e.g., 0.5)
```

Gurobi version:
```bash
cd src/switch_cost_DOBSS
python cost_BSG_miqp.py cost_BSSG_input.txt <alpha>
# alpha is the switching cost weight parameter
```

### Resource Allocation (Homogeneous Schedules)
```bash
cd src/ResourcesHomogeneousScheduleSingleton
python BSG_multi_lp.py BSSG_input.txt
# Then to get mixed strategy distribution:
python strategy_generator.py
```

## Architecture

### Solver Modules

- **`src/DOBSS/`**: Core Stackelberg game solvers
  - `BSG_miqp.py`: MIQP formulation using Gurobi
  - `BSG_miqp_ortools.py`: MILP formulation using OR-Tools (linearized via McCormick envelopes)
  - `BSG_milp.py`: MILP formulation using Gurobi with auxiliary z variables
  - `whatToFix.py`: Finds optimal attack to remove using Gurobi
  - `whatToFix_ortools.py`: Finds optimal attack to remove using OR-Tools
  - `tests/`: Unit tests for OR-Tools solvers

- **`src/switch_cost_DOBSS/`**: Extended model with configuration switching costs
  - `cost_BSG_miqp.py`: MIQP formulation using Gurobi
  - `cost_BSG_miqp_ortools.py`: MILP formulation using OR-Tools (linearized via McCormick envelopes)
  - `tests/`: Unit tests for OR-Tools solver
  - Uses McCormick envelopes to approximate non-convex transition cost terms

- **`src/ResourcesHomogeneousScheduleSingleton/`**: Multi-LP solver for resource allocation games
  - `BSG_multi_lp.py`: Solves separate LPs for each potential attack target
  - `constrained_birkhoff_von_neumann.py`: Implements constrained BvN decomposition
  - `strategy_generator.py`: Orchestrates the full pipeline

### Input File Format

All solvers use text-based input files specifying:
1. Number of defender strategies (X)
2. Number of attacker types (L)
3. For each attacker type:
   - Probability of that attacker type
   - Number of attack actions (Q)
   - Attack names separated by `|`
   - X × Q utility matrix with entries `defender_reward,attacker_reward`

Example structure (see `src/DOBSS/input.txt`):
```
4           # defender strategies
4           # attacker types
0.5         # probability of attacker type 1
3           # attack actions for type 1
Attack1|Attack2|Attack9
-2,6 5,-8 0,0    # row 1: utilities for each attack
...
```

For switching cost games, an additional X × X switching cost matrix precedes the attacker data.
