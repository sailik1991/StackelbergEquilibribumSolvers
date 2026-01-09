#!/usr/bin/python

#   maximize
#       p[l] * R[l][i][j] * x[i] * q[l][j] - alpha * cost[i][j] * w[i][j]
#   subject to
#       Sum x[i] = 1
#       For each l, Sum q[l][j] = 1
#       For each l & all j, 0 <= a[l] - C[l][i][j] * x[i]
#       For each l & all j, a[l] - C[l][i][j] * x[i] <= (1-q[l][j])M
#       x[i] in [0, 1]
#       For each l, q[l][j] binary
#       a[l] is Real
#
# Note: The quadratic terms are linearized using auxiliary variables:
#   - z[l][i][j] = x[i] * q[l][j] (for reward computation)
#   - w[i][j] = x[i] * x[j] (for switching cost computation)
#
# McCormick envelope for z[i][j] (x continuous [0,1], q binary {0,1}):
#   z[i][j] <= x[i]
#   z[i][j] <= q[j]
#   z[i][j] >= x[i] + q[j] - 1
#   z[i][j] >= 0
#
# McCormick envelope for w[i][j] (x[i], x[j] continuous [0,1]):
#   w[i][j] >= 0
#   w[i][j] >= x[i] + x[j] - 1
#   w[i][j] <= x[i]
#   w[i][j] <= x[j]
#   Plus: sum_j(w[i][j]) = x[i] for all i (flow conservation)
#         sum_i(w[i][j]) = x[j] for all j (flow conservation)
#         sum_i,j(w[i][j]) = 1 (total probability)
#         w[i][i] = 0 (no self-transition cost)

from ortools.linear_solver import pywraplp
import sys

__author__ = "Sailik Sengupta"
__version__ = "2.0"
__email__ = "link2sailik [at] gmail [dot] com"

"""
Input Format:
# X - Num of defender actions
# ---
# X * X matrix -- (i, j) represents cost to switch from configuration i to j
# ---
# Num Attackers
# Prob of each attacker
# Attack names of each attacker separated by |
# ---
# X * Attack actions -- Utility matrix showcasing (Reward for defender, Reward for attacker)
# ---
"""

# Create a new model using SCIP solver (supports MILP)
solver = pywraplp.Solver.CreateSolver('SCIP')
if not solver:
    print("Could not create SCIP solver")
    sys.exit(1)

if len(sys.argv) < 3:
    print("Usage: python cost_BSG_miqp_ortools.py <input_file> <alpha>")
    print("  alpha: switching cost weight parameter (e.g., 0.5)")
    sys.exit(1)

f = open(str(sys.argv[1]), "r")
alpha = float(sys.argv[2])

infinity = solver.infinity()

# Add defender strategies to the model
X = int(f.readline())
x = []
for i in range(X):
    n = "x-" + str(i)
    x.append(solver.NumVar(0, 1, n))

# Add defender's switching cost matrix
cost = []
for i in range(X):
    cost.append([int(j) for j in f.readline().split()])

# Add defender strategy constraints: sum of x[i] = 1
solver.Add(sum(x[i] for i in range(X)) == 1)

# Add transition cost variables w[i][j]
# w[i][j] approximates x[i] * x[j] using McCormick envelopes
w = []
for i in range(X):
    w_row = []
    from_config_sum = []  # sum_j(w[i][j])
    for j in range(X):
        n = "w-" + str(i) + "-" + str(j)
        w_ij = solver.NumVar(0, 1, n)

        if i == j:
            # No self-transition cost
            solver.Add(w_ij == 0)
        else:
            # McCormick envelope constraints for x[i] * x[j]
            solver.Add(w_ij >= 0)
            solver.Add(w_ij >= x[i] + x[j] - 1)
            solver.Add(w_ij <= x[i])
            solver.Add(w_ij <= x[j])

        w_row.append(w_ij)
        from_config_sum.append(w_ij)
    w.append(w_row)
    # Flow conservation: sum_j(w[i][j]) = x[i]
    solver.Add(sum(from_config_sum) == x[i])

# Flow conservation: sum_i(w[i][j]) = x[j]
for j in range(X):
    to_config_sum = [w[i][j] for i in range(X)]
    solver.Add(sum(to_config_sum) == x[j])

# Total probability constraint: sum_i,j(w[i][j]) = 1
all_w = []
for i in range(X):
    for j in range(X):
        all_w.append(w[i][j])
solver.Add(sum(all_w) == 1)

# Objective function (linear after substitution with auxiliary variables)
objective = solver.Objective()

# Add switching cost terms to objective: -alpha * cost[i][j] * w[i][j]
for i in range(X):
    for j in range(X):
        coef = -alpha * cost[i][j]
        objective.SetCoefficient(w[i][j], coef)

""" Start processing for attacker types """
L = int(f.readline())
M = 100000000

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

    # Linearize x[i] * q[j] using auxiliary variables z[l][i][j]
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
        # Only print x, q, and w variables (skip auxiliary z variables)
        name = var.name()
        if not name.startswith("z-"):
            val = var.solution_value()
            # Only print non-zero values for w variables to reduce output
            if name.startswith("w-"):
                if abs(val) > 1e-6:
                    print("%s -> %g" % (name, val))
            else:
                print("%s -> %g" % (name, val))

    printSeperator()
    print("Obj -> %g" % solver.Objective().Value())
    printSeperator()
else:
    print("The problem does not have an optimal solution. Status:", status)
