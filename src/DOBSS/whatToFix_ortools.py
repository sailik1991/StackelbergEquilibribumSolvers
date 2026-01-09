#!/usr/bin/python

#   maximize
#       p[l] * Ri[l][i][j] * x[i] * q[l][j]
#   subject to
#       Sum x[i] = 1
#       For each l,  q[l][j] = 1
#       For each l & all i, 0 <= a[l] - C[l][i][j] * x[i]
#       For each l & all i, a[l] - C[l][i][j] * x[i] <= (1-q[l][j])M
#       x[i] <= 1
#       x[i] >= 0
#       For each l, q[l][j] binary
#       a[l] is Real
#
# Note: The quadratic term x[i] * q[j] is linearized using auxiliary variables z[i][j]
# Since q[j] is binary and x[i] in [0,1], we use McCormick envelopes:
#   z[i][j] <= x[i]
#   z[i][j] <= q[j]
#   z[i][j] >= x[i] + q[j] - 1
#   z[i][j] >= 0

from ortools.linear_solver import pywraplp
from itertools import combinations
import sys

__author__ = "Sailik Sengupta"
__version__ = "2.0"
__email__ = "link2sailik [at] gmail [dot] com"


def getAllAttacks():
    """
    Reads the input file to obtain attacks for all attackers
    And then make a list of unique attacks
    """
    f = open(sys.argv[1], "r")
    f_lines = f.read().split("\n")
    attacks = list()
    for line in f_lines:
        if "|" in line:
            a = line.split("|")
            for x in a:
                attacks.append(x)
    f.close()
    return list(set(attacks))


def solveBSG(invalidAttacks):
    # Create a new model using SCIP solver (supports MILP)
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        print("Could not create SCIP solver")
        sys.exit(1)

    f = open(str(sys.argv[1]), "r")
    """
    ------ Input file ------
    No. of defender strategies (X)
    No. of attackers (L)
    | Probability for an attacker (p_l)
    | No. of attack strategies for an attacker (Q_l)
    | [
    |       Matrix ( X * Q_l) with
    |       values r, c
    | ]
    | where r,c are rewards for defender and attacker respectively
    ------ Example (see BSSG_input.txt)------
    | 2
    | 2
    | 0.5
    | 2
    | 8,2 6,0
    | 7,0 2,6
    | 0.5
    | 2
    | 5,0 4,2
    | 4,2 5,0
    |----------------------------------------
    """

    def isAttackValid(invalidAttacks, cve_name):
        for a in invalidAttacks:
            if a in cve_name:
                return False
        return True

    # Add defender strategies to the model
    X = int(f.readline())
    x = []
    for i in range(X):
        n = "x-" + str(i)
        x.append(solver.NumVar(0, 1, n))

    # Add defender strategy constraints: sum of x[i] = 1
    solver.Add(sum(x[i] for i in range(X)) == 1)

    """ Start processing for attacker types """

    L = int(f.readline())
    M = 100000000
    infinity = solver.infinity()

    # Objective function (linear after substitution with z variables)
    objective = solver.Objective()

    for l in range(L):

        # Probability of l-th attacker
        v = f.readline().strip()
        p = float(v)

        # Add l-th attacker info to the model
        Q = int(f.readline())
        q = []
        cve_names = f.readline().strip().split("|")

        # Track valid attack indices
        valid_indices = []
        for j in range(Q):
            if isAttackValid(invalidAttacks, cve_names[j]):
                valid_indices.append(j)
                n = str(l) + "-" + cve_names[j]
                q.append(solver.IntVar(0, 1, n))

        a = solver.NumVar(-infinity, infinity, "a-" + str(l))

        # Get reward for attacker and defender (only for valid attacks)
        R = []
        C = []
        for i in range(X):
            rewards = f.readline().split()
            r = []
            c = []
            for j in range(Q):
                if isAttackValid(invalidAttacks, cve_names[j]):
                    r_and_c = rewards[j].split(",")
                    r.append(r_and_c[0])
                    c.append(r_and_c[1])
            R.append(r)
            C.append(c)

        # Linearize x[i] * q[j] using auxiliary variables z[i][j]
        # McCormick envelope for product of continuous [0,1] and binary {0,1}
        z = []
        for i in range(X):
            z_row = []
            for j in range(len(q)):
                z_name = "z-" + str(l) + "-" + str(i) + "-" + str(j)
                z_ij = solver.NumVar(0, 1, z_name)

                # z[i][j] <= x[i]
                solver.Add(z_ij <= x[i])
                # z[i][j] <= q[j]
                solver.Add(z_ij <= q[j])
                # z[i][j] >= x[i] + q[j] - 1
                solver.Add(z_ij >= x[i] + q[j] - 1)

                z_row.append(z_ij)
            z.append(z_row)

        # Update objective function: sum of p * R[i][j] * z[i][j]
        for i in range(X):
            for j in range(len(q)):
                coef = p * float(R[i][j])
                objective.SetCoefficient(z[i][j], coef)

        # Add constraints to make attacker have a pure strategy: sum of q[j] = 1
        solver.Add(sum(q[j] for j in range(len(q))) == 1)

        # Add constraints to make attacker select dominant pure strategy
        for j in range(len(q)):
            # a - sum(C[i][j] * x[i]) >= 0
            solver.Add(a - sum(float(C[i][j]) * x[i] for i in range(X)) >= 0)
            # a - sum(C[i][j] * x[i]) <= (1 - q[j]) * M
            solver.Add(a - sum(float(C[i][j]) * x[i] for i in range(X)) <= (1 - q[j]) * M)

    f.close()

    # Set objective to maximize
    objective.SetMaximization()

    # Solve
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        return (solver.Objective().Value(), solver.variables())
    else:
        return (float('-inf'), [])


""" Main code starts here """
attack_list = getAllAttacks()

# Gets K-set permutations of attack actions
# NO-OP is not a member of the permutations sets
k = 1
attack_sets = combinations(attack_list, k)

# Obtain the subtracted attack_set that gives the highest reward
maxObj = -1000000
bestSet = []
maxVar = []
allSet = []
for attacks in attack_sets:
    obj, var = solveBSG(attacks)
    allSet.append((attacks, obj))
    if obj > maxObj:
        maxObj = obj

print("=====")
print(allSet)
print("=====")
print("Best Obj value -> " + str(maxObj))

for attacks, obj in allSet:
    if obj == maxObj:
        print(attacks)
