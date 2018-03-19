import pickle, subprocess, sys
import numpy as np
from constrained_birkhoff_von_neumann import constrained_birkhoff_von_neumann_decomposition

'''
This code can be used to run the set of MILPs follwed by a constrained BVN decomppostion to
obtain the probabilities of the {T \choose R} strategies of a player in a Stackelberg Allocation Game.
'''

__author__ = "Sailik Sengupta"
__version__ = "1.0"
__email__ = "link2sailik [at] gmail [dot] com"

def get_marg_probs(filename="BSSG_input.txt"):
    subprocess.call(["/opt/gurobi701/linux64/bin/gurobi.sh", "BSG_multi_milp.py", filename])

'''
Obtain marginal probabilty data from pickel file and generate
mixed strategy for the game using the BvN theorem.
'''
def obtain_mixed_strategy():
    f = open(r'best_marg_prob.pkl', 'rb')
    best_mp = pickle.load(f)

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

    return constrained_birkhoff_von_neumann_decomposition(np.array(best_mp), constraints)

''' Post process results for the homogeneous resource case '''
def homog_probs(result):
    probs = result[0]
    strategies = result[1]
    print(strategies)
    homog_strategies = []
    for s in strategies:
        homog_strategies.append(np.sum(s, axis=0))
    print(len(probs))
    print(homog_strategies)
    print("{} = {}".format(probs,sum(probs)))

def main(filename="BSSG_input.txt"):
    get_marg_probs(filename)
    homog_probs(obtain_mixed_strategy())

if __name__ == '__main__':
    try:
        main(sys.argv[1])
    except:
        main()
