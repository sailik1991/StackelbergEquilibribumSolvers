import pickle
import subprocess
import numpy as np
from constrained_birkhoff_von_neumann import constrained_birkhoff_von_neumann_decomposition

'''
This code can be used to run the set of MILPs follwed by a constrained BVN decomppostion to
obtain the probabilities of the {T \choose R} strategies of a player in a Stackelberg Allocation Game.
'''

__author__ = "Sailik Sengupta"
__version__ = "1.0"
__email__ = "link2sailik [at] gmail [dot] com"

subprocess.call(["/opt/gurobi701/linux64/bin/gurobi.sh", "BSG_milp.py", "BSSG_input.txt"])

f = open(r'best_marg_prob.pkl', 'rb')
best_mp = pickle.load(f)

#best_mp = [ [0.7, 0.2, 0.1],[0. , 0.3, 0.7]]

NUM_RESOURCES = len(best_mp)
NUM_TARGETS = len(best_mp[0])

''' add constrains for the decomposition '''
constraints = {}

# All rows should add up to 1 (all resources are allocated)
for r in range(NUM_RESOURCES):
	row = []
	for t in range(NUM_TARGETS):
		row.append((r,t))
	constraints[frozenset(row)] = (1,1)

# All columns should be 0 or 1 (all targets are either covered by one resource or none)
for t in range(NUM_TARGETS):
	col = []
	for r in range(NUM_RESOURCES):
		col.append((r,t))
	constraints[frozenset(col)] = (0,1)

print(best_mp)
print(constraints)

result = constrained_birkhoff_von_neumann_decomposition(np.array(best_mp), constraints)

print(result)
