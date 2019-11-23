#!/usr/bin/python

#   maximize
#       p[l] * Ri[l][i][j] *  z[l][i][j]
#   subject to
#       For each l, Sum_i Sum_j z[l][i][j]  = 1
#       For each l,  Sum_j z[l][i][j] <= 1
# 	For each l,  q[l][j] <= Sum_i z[l][i][j] <= 1
# 	For each l,  Sum_j q[l][j] = 1
#       For each l & all i, 0 <= a[l] - C[l][i][j] * Sum_j z[l][i][j]
#       For each l & all i, a[l] - C[l][i][j] * Sum_j z[l][i][j] <= (1-y[l][j])M
#       For all l, Sum_j z[l][i][j] = Sum_j z[0][i][j]
#       z[l][i][j] <= 1
#       z[l][i][j] >= 0
#       For each l, q[l][j] binary
#       a[l] is Real

from gurobipy import *

__author__ = "Sailik Sengupta"
__version__ = "1.0"
__email__ = "link2sailik [at] gmail [dot] com"

try:
    # k-set critical attacks
    k = 1

    # Create a new model
    m = Model("MILP")

    f = open(sys.argv[1], "r")
    """
    ------ Input file ------
    No. of defender strategies (X)
    No. of attackers (L)
    | Probability for an attacker (p_l)
    | No. of attack strategies for an attacker (Q_l)
    | Name of each attack seperated by space
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
    | Attack_Name_1 Attack_Name_2
    | 8,2 6,0
    | 7,0 2,6
    | 0.5
    | 2
    | Attack_Name_1 Attack_Name_2
    | 5,0 4,2
    | 4,2 5,0
    |----------------------------------------
    """

    # Add defender stategies to the model
    X = int(f.readline())
    L = int(f.readline())
    obj = QuadExpr()
    M = 100000000
    x = []

    for l in range(L):

        # Probability of l-th attacker
        p = float(f.readline())

        ##### q #####
        Q = int(f.readline())
        q = []
        cve_names = f.readline().strip().split("|")
        for i in range(Q):
            n = "attacker" + str(l) + "-" + cve_names[i]
            q.append(m.addVar(lb=0, ub=1, vtype=GRB.INTEGER, name=n))
        m.update()

        sum_q_1 = LinExpr()
        for i in range(Q):
            sum_q_1.add(q[i])
        m.addConstr(sum_q_1 == 1)

        ##### z #####
        z = []
        for i in range(X):
            zr = []
            for j in range(Q):
                n = "z" + str(l) + "-x" + str(i) + "-q" + str(j)
                zr.append(m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=n))
            z.append(zr)
        m.update()

        sum_ij = LinExpr()
        for i in range(X):
            sum_j = LinExpr()
            for j in range(Q):
                sum_ij.add(z[i][j])
                sum_j.add(z[i][j])
            m.addConstr(sum_j <= 1)
            if l == 0:
                x.append(sum_j)
            else:
                m.addConstr(sum_j == x[i])
        m.addConstr(sum_ij == 1)

        for j in range(Q):
            sum_i = LinExpr()
            for i in range(X):
                sum_i.add(z[i][j])
            m.addConstr(sum_i <= 1)
            m.addConstr(sum_i >= q[j])

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
                obj.add(r * z[i][j])

        # Add constrains to make attacker select dominant pure strategy
        for j in range(Q):
            val = LinExpr()
            val.add(a)
            for i in range(X):
                x_con = LinExpr()
                for k in range(Q):
                    x_con.add(z[i][k])
                val.add(float(C[i][j]) * x_con, -1.0)
            m.addConstr(val >= 0)
            m.addConstr(val <= (1 - q[j]) * M)

        m.update()

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

    # Prints constrains
    # printSeperator()
    # for c in m.getConstrs():
    #    if c.Slack == 0.0:
    #        print(str(c.ConstrName) + ': ' + str(c.Slack))
    # printSeperator()
except GurobiError:
    print(m.computeIIS())
    m.write("iis.ilp")
