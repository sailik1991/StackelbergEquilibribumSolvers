#!/usr/bin/python

from gurobipy import *
import sys

"""
Input Format:
# X - Num of defender actions
# ---
# X * X matrix -- (i, j) represents cost to switch from configuration i to j
# ---
# Num Attackers
# Prob of each attacker
# Attack names of each attacker seperated by |
# ---
# X * Attack actions -- Utility matrix showcasing (Reward for defender, Reward for attacker)
# ---
"""

# Create a new model
m = Model("MIQP")
f = open(str(sys.argv[1]), "r")

# Add defender stategies to the model
X = int(f.readline())
x = []
for i in range(X):
    n = "x-" + str(i)
    x.append(m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=n))
m.update()

# Add defender's switching cost
cost = []
for i in range(X):
    cost.append([int(j) for j in f.readline().split()])

# Add defender stategy constraints
con = LinExpr()
for i in range(X):
    con.add(x[i])
m.addConstr(con == 1)
m.update()

# Add transition cost variables
w = []
to_config_constr = [LinExpr() for i in range(X)]
for i in range(X):
    _w = []
    from_config_constr = LinExpr()
    for j in range(X):
        n = "w-" + str(i) + str(j)
        temp = m.addVar(vtype=GRB.CONTINUOUS, name=n)
        # Use McCormick_envelopes to find upper and lower bounds for the
        # non-convex function x_i * x_j
        if i == j:
            m.addConstr(temp == 0)
        else:
            m.addConstr(temp >= 0)
            m.addConstr(temp >= x[i] + x[j] - 1)
            m.addConstr(temp <= x[i])
            m.addConstr(temp <= x[j])
        _w.append(temp)
        from_config_constr.add(temp)
        to_config_constr[j].add(temp)
    m.addConstr(from_config_constr == x[i])
    w.append(_w)

for i in range(X):
    m.addConstr(to_config_constr[i] == x[i])

m.update()

# subtract costs from the objective function

"""
# Actual computation
obj = QuadExpr()
alpha = 0.85
for i in range(X):
    for j in range(X):
        obj.add( alpha * cost[i][j] * ( x[i] + x[j] ), -1)
"""

# McCormick envelope approximation
obj = QuadExpr()
alpha = float(sys.argv[2])
two_step_configs = LinExpr()
for i in range(X):
    for j in range(X):
        obj.add(alpha * cost[i][j] * w[i][j], -1)
        two_step_configs.add(w[i][j])
m.addConstr(two_step_configs == 1)

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
    for i in range(Q):
        n = str(l) + "-" + cve_names[i]
        q.append(m.addVar(lb=0, ub=1, vtype=GRB.INTEGER, name=n))

    a = m.addVar(
        lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="a-" + str(l)
    )

    m.update()

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

    # Update objective function
    for i in range(X):
        for j in range(Q):
            r = p * float(R[i][j])
            obj.add(r * x[i] * q[j])

    # Add constraints to make attaker have a pure strategy
    con = LinExpr()
    for j in range(Q):
        con.add(q[j])
    m.addConstr(con == 1)

    # Add constrains to make attacker select dominant pure strategy
    for j in range(Q):
        val = LinExpr()
        val.add(a)
        for i in range(X):
            val.add(float(C[i][j]) * x[i], -1.0)
        m.addConstr(val >= 0, q[j].getAttr("VarName") + "lb")
        m.addConstr(val <= (1 - q[j]) * M, q[j].getAttr("VarName") + "ub")

# Set objective funcion as all attackers have now been considered
m.setObjective(obj, GRB.MAXIMIZE)

# Solve MIQP
m.optimize()

# Print out values
def printSeperator():
    print("---------------")

printSeperator()
for v in m.getVars():
    print("%s -> %g" % (v.varName, v.x))

printSeperator()
print("Obj -> %g" % m.objVal)
printSeperator()
