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
from itertools import *
from sys import *
from copy import deepcopy

__author__ = "Sailik Sengupta"
__version__ = "1.0"
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
    # print(list(set(attacks)))
    return list(set(attacks))


def solveBSG(invalidAttacks):
    # Create a new model
    m = Model("MIQP")
    m.setParam("OutputFlag", False)

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

    # Add defender stategy constraints
    con = LinExpr()
    for i in range(X):
        con.add(x[i])
    m.addConstr(con == 1)

    """ Start processing for attacker types """

    L = int(f.readline())
    obj = QuadExpr()
    M = 11

    def isAttackValid(invalidAttacks, cve_name):
        for a in invalidAttacks:
            if a in cve_name:
                return False
        return True

    for l in range(L):

        # Probability of l-th attacker
        p = float(f.readline())

        # Add l-th attacker info to the model
        Q = int(f.readline())
        q = []
        cve_names = f.readline().strip().split("|")

        for i in range(Q):
            if isAttackValid(invalidAttacks, cve_names[i]):
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
            addAttack = True
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

        # Update objective function
        for i in range(X):
            for j in range(len(q)):
                r = p * float(R[i][j])
                obj.add(r * x[i] * q[j])

        # Add constraints to make attaker have a pure strategy
        con = LinExpr()
        for j in range(len(q)):
            con.add(q[j])
        m.addConstr(con == 1)

        # Add constrains to make attacker select dominant pure strategy
        for j in range(len(q)):
            val = LinExpr()
            val.add(a)
            for i in range(X):
                val.add(float(C[i][j]) * x[i], -1.0)

            m.addConstr(val >= 0)
            m.addConstr(val <= (1 - q[j]) * M)

    # Set objective funcion as all attackers have now been considered
    m.setObjective(obj, GRB.MAXIMIZE)

    # Solve MIQP
    m.optimize()

    """
    Prints out the strategies for the defender and attacker
    after the concerned attacks in invalid attacks are taken out
    """
    """
    # Print out values
    def printSeperator():
        print ('---------------')

    printSeperator()
    for v in m.getVars():
        print('%s -> %g' % (v.varName, v.x))

    printSeperator()
    print('Obj -> %g' % m.objVal)
    printSeperator()
    """

    return (m.objVal, m.getVars())


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

# for var in maxVar:
#    print str(var)
print("=====")
print(allSet)
print("=====")
print("Best Obj value -> " + str(maxObj))

for attacks, obj in allSet:
    if obj == maxObj:
        print(attacks)
