#!/usr/bin/python

"""
This code uses the MILP formulation for the heterogeneous agents case with singleton
scendules and adapts it for the homogeneous case (see https://goo.gl/RM1vRZ for details).
"""

__author__ = "Sailik Sengupta"
__version__ = "1.0"
__email__ = "link2sailik [at] gmail [dot] com"

from gurobipy import *
import sys, pickle

NUM_TARGETS = 0
NUM_RESOURCES = 0

# Defender's reward for each resource
# 0-th element denoted reward on covering, 1-st elements denotes (-ve) reward on not covering
R = []

# Attacker's reward for each resource
C = []


def read_data(filename):
    """
    ------ Input file ------
    | No. of targets (n)
    | Defender's resources (rd)
    | R(c)_1 R(u)_1
    | ...
    | R(c)_n R(u)_n
    | C(c)_1 C(u)_1
    | ...
    | C(c)_n C(u)_n
    ------ Example (see BSSG_input.txt)------
    | 4
    | 2
    | 0 -15
    | 0 -10
    | 0 -13
    | 0 -15
    | -5 15
    | -5 10
    | -4 13
    | -6 15
    |----------------------------------------
    """
    f = open(str(filename), "r")

    global NUM_TARGETS
    NUM_TARGETS = int(f.readline())

    global NUM_RESOURCES
    NUM_RESOURCES = int(f.readline())

    # Get defender utilities
    for i in range(NUM_TARGETS):
        R.append(list(map(float, f.readline().strip().split(" "))))

    # Get attacker utilities
    for i in range(NUM_TARGETS):
        C.append(list(map(float, f.readline().strip().split(" "))))


def attack_target(t_attacked):
    print(NUM_TARGETS, NUM_RESOURCES, C, R)
    # Create a new model
    m = Model("MILP")

    # probability of monitoring target i (p-i)
    p = []

    for i in range(NUM_TARGETS):
        name = "p-" + str(i)
        p.append(m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=name))
    m.update()

    # probailities of assigning resource r to monitor target t (p-r-t)
    p_rt = []
    for r in range(NUM_RESOURCES):
        p_r = []
        for t in range(NUM_TARGETS):
            name = "mp-" + str(r) + "-" + str(t)
            p_r.append(m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=name))
        p_rt.append(p_r)
    m.update()

    # For every target t, sum_r (p-r-t) <= p_t
    for t in range(NUM_TARGETS):
        for_all_target_con = LinExpr()
        for r in range(NUM_RESOURCES):
            for_all_target_con.add(p_rt[r][t])
        m.addConstr(for_all_target_con == p[t])
    m.update()

    # For every resource r, sum_t (p-r-t) <= 1
    for r in range(NUM_RESOURCES):
        for_all_resource_con = LinExpr()
        for t in range(NUM_TARGETS):
            for_all_resource_con.add(p_rt[r][t])
        m.addConstr(for_all_resource_con <= 1)
    m.update()

    # Constraints to ensure the target attacked gives attacker the max uitlity
    U_a = []
    for t in range(NUM_TARGETS):
        e = LinExpr()
        e = C[t][0] * p[t] + C[t][1] * (1 - p[t])
        U_a.append(e)
    for t in range(NUM_TARGETS):
        m.addConstr(U_a[t] <= U_a[t_attacked])

    # Add objective function
    obj = LinExpr()
    obj = R[t_attacked][0] * p[t_attacked] + R[t_attacked][1] * (1 - p[t_attacked])

    m.setObjective(obj, GRB.MAXIMIZE)

    # Solve MILP
    m.optimize()

    def_reward = m.objVal
    def_marg_prob = [[0.0 for t in range(NUM_TARGETS)] for r in range(NUM_RESOURCES)]

    print("---------------")
    for v in m.getVars():
        print("%s -> %g" % (v.varName, v.x))
        if "mp" in v.varName:
            waste, r, c = v.varName.split("-")
            def_marg_prob[int(r)][int(c)] = v.x

    print("---------------")
    print("Obj -> %g" % m.objVal)
    print("---------------")

    """
    === Uncomment if needed for debugging === 
    # Prints constrains
    printSeperator()
    for c in m.getConstrs():
        if c.Slack == 0.0:
            print(str(c.ConstrName) + ': ' + str(c.Slack))
    printSeperator()
    """

    m.reset()
    return (def_reward, def_marg_prob)


def main(filename):
    read_data(filename)
    obj_vals = []
    mp = []
    for t in range(NUM_TARGETS):
        val, marg_prob = attack_target(t)
        obj_vals.append(val)
        mp.append(marg_prob)

    best_def_reward = max(obj_vals)
    best_mp = mp[obj_vals.index(best_def_reward)]

    f = open(r"best_marg_prob.pkl", "wb")
    pickle.dump(best_mp, f)
    f.close()


if __name__ == "__main__":
    main(sys.argv[1])
