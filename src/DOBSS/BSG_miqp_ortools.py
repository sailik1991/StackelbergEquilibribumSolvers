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
import sys

__author__ = "Sailik Sengupta"
__version__ = "2.0"
__email__ = "link2sailik [at] gmail [dot] com"

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
    for j in range(Q):
        n = str(l) + "-" + cve_names[j]
        q.append(solver.IntVar(0, 1, n))

    a = solver.NumVar(-infinity, infinity, "a-" + str(l))

    # Get reward for attacker and defender
    R = []
    C = []
    for i in range(X):
        rewards = f.readline().split()
        r = []
        c = []
        for j in range(Q):
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
        for j in range(Q):
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
        for j in range(Q):
            coef = p * float(R[i][j])
            objective.SetCoefficient(z[i][j], coef)

    # Add constraints to make attacker have a pure strategy: sum of q[j] = 1
    solver.Add(sum(q[j] for j in range(Q)) == 1)

    # Add constraints to make attacker select dominant pure strategy
    for j in range(Q):
        # a - sum(C[i][j] * x[i]) >= 0
        solver.Add(a - sum(float(C[i][j]) * x[i] for i in range(X)) >= 0)
        # a - sum(C[i][j] * x[i]) <= (1 - q[j]) * M
        solver.Add(a - sum(float(C[i][j]) * x[i] for i in range(X)) <= (1 - q[j]) * M)

f.close()

# Set objective to maximize
objective.SetMaximization()

# Solve
status = solver.Solve()


# Print out values
def printSeperator():
    print("---------------")


if status == pywraplp.Solver.OPTIMAL:
    printSeperator()
    for var in solver.variables():
        # Only print x and q variables (skip auxiliary z variables)
        if not var.name().startswith("z-"):
            print("%s -> %g" % (var.name(), var.solution_value()))

    printSeperator()
    print("Obj -> %g" % solver.Objective().Value())
    printSeperator()
else:
    print("The problem does not have an optimal solution. Status:", status)
