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

from gurobipy import *

__author__ = "Sailik Sengupta"
__version__ = "1.0"
__email__ = "link2sailik [at] gmail [dot] com"

try:

    # Create a new model
    m = Model("MIQP")
    m_ur = Model("MIQP_UR")

    f = open(sys.argv[1], "r")
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

    # Add defender stategies to the model
    X = int(f.readline())
    x = []
    for i in range(X):
        n = "x-" + str(i)
        x.append(m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=n))
    m.update()
    xr = 1.0 / len(x)

    # Add defender stategy constraints
    con = LinExpr()
    for i in range(X):
        con.add(x[i])
    m.addConstr(con == 1)

    """ Start processing for attacker types """

    L = int(f.readline())
    obj = QuadExpr()
    obj_ur = LinExpr()
    M = 100000000

    for l in range(L):

        # Probability of l-th attacker
        p = float(f.readline())

        # Add l-th attacker info to the model
        Q = int(f.readline())
        q = []
        q_ur = []
        cve_names = f.readline().split("|")
        for i in range(Q):
            n = cve_names[i]
            q.append(m.addVar(lb=0, ub=1, vtype=GRB.INTEGER, name=n))
            q_ur.append(m_ur.addVar(lb=0, ub=1, vtype=GRB.INTEGER, name=n))

        a = m.addVar(
            lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="a-" + str(l)
        )
        a_ur = m_ur.addVar(
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name="a_ur-" + str(l),
        )

        m.update()
        m_ur.update()

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
                obj_ur.add(r * xr * q_ur[j])

        # Add constraints to make attaker have a pure strategy
        con = LinExpr()
        con_ur = LinExpr()
        for j in range(Q):
            con.add(q[j])
            con_ur.add(q_ur[j])
        m.addConstr(con == 1)
        m_ur.addConstr(con_ur == 1)

        # Add constrains to make attacker select dominant pure strategy
        for j in range(Q):
            val = LinExpr()
            val.add(a)
            val_ur = LinExpr()
            val_ur.add(a_ur)
            val_sub = 0.0
            for i in range(X):
                val.add(float(C[i][j]) * x[i], -1.0)
                val_sub += float(C[i][j]) * xr
            m.addConstr(val >= 0)
            m.addConstr(val <= (1 - q[j]) * M)
            val_ur.add(val_sub, -1.0)
            m_ur.addConstr(val_ur >= 0)
            m_ur.addConstr(val_ur <= (1 - q_ur[j]) * M)

    # Set objective funcion as all attackers have now been considered
    m.setObjective(obj, GRB.MAXIMIZE)
    m_ur.setObjective(obj_ur, GRB.MAXIMIZE)

    # Solve MIQP
    m.optimize()
    m_ur.optimize()

    # Print out values
    def printSeperator():
        print("---------------")

    printSeperator()
    for v in m.getVars():
        print("%s -> %g" % (v.varName, v.x))

    printSeperator()
    print("Obj -> %g" % m.objVal)
    printSeperator()

    printSeperator()
    for v in m_ur.getVars():
        print("%s -> %g" % (v.varName, v.x))

    printSeperator()
    print("Obj -> %g" % m_ur.objVal)
    printSeperator()

except GurobiError:
    print("Error reported")
    # m_ur.computeIIS();
    # m_ur.write("model.ilp");
