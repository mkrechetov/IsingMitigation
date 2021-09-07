import networkx as nx
from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

import gurobipy as gp
from gurobipy import GRB

def gurobi_map(J, timelimit=60, silent=True):
    with suppress_stdout():
        # Prepare data
        nodes = list(range(len(J)))
        G = nx.from_numpy_matrix(J)
        edges = list(G.edges())

        # Create a new model"
        m = gp.Model("map")
        if silent:
            m.Params.LogToConsole = 0

        # Create variables
        x = []
        for node in nodes:
            # x[node] in {0, 1} and change of variable is required
            x.append(m.addVar(vtype=GRB.BINARY, name="x_" + str(node)))

        # Set objective
        objective = 0.0

        # Add field
        for node in nodes:
            field = J[node, node]
            objective += field * (2 * x[node] - 1)

        # Add pairwise interactions
        for edge in edges:
            a = edge[0]
            b = edge[1]
            objective += J[a, b] * (2 * x[a] - 1) * (2 * x[b] - 1)

        m.setObjective(objective, GRB.MAXIMIZE)

        # Optimize model
        m.Params.TimeLimit = timelimit
        m.setParam('OutputFlag', 0)
        m.setParam('OptimalityTol', 0.000000001)

        m.optimize()
        m.printStats()

        result = []
        for v in m.getVars():
            result.append(v.x)

        return result


def gurobi_map_explicit(J, h, I, timelimit=60, silent=True):
    # Prepare data
    nodes = list(range(len(J)))

    # Create a new model"
    m = gp.Model("map")
    if silent:
        m.Params.LogToConsole = 0

    # Create variables
    x = []
    for node in nodes:
        # x[node] in {0, 1} and change of variable is required
        x.append(m.addVar(vtype=GRB.BINARY, name="x_" + str(node)))

    # Set objective
    objective = 0.0

    # Add field
    for node in nodes:
        objective += h[node] * (2 * x[node] - 1)

    for node in I:
        m.addConstr(x[node] == 1)

    # Add pairwise interactions
    for a in nodes:
        for b in nodes:
            objective += J[a, b] * (2 * x[a] - 1) * (2 * x[b] - 1) / 2

    m.setObjective(objective, GRB.MAXIMIZE)

    # Optimize model
    # m.Params.TimeLimit = timelimit
    # m.setParam('OutputFlag', 0)
    m.setParam('OptimalityTol', 0.000000001)

    m.optimize()
    m.printStats()

    obj = m.getObjective()
    print("Energy value: ", obj.getValue())

    result = []
    vars = m.getVars()
    for v in vars:
        result.append(v.x)

    return result