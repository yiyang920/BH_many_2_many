import numpy as np


def bellman(J, Q):
    num_nodes = Q.shape[0]
    next_J = np.empty_like(J)
    for v in range(num_nodes):
        next_J[v] = np.min(Q[v, :] + J)
    return next_J


def compute_cost_to_go(Q):
    num_nodes = Q.shape[0]
    J = np.zeros(num_nodes)  # Initial guess
    next_J = np.empty(num_nodes)  # Stores updated guess
    max_iter = 500
    i = 0

    while i < max_iter:
        next_J = bellman(J, Q)
        if np.allclose(next_J, J):
            break
        else:
            J[:] = next_J  # Copy contents of next_J to J
            i += 1

    return J


def get_best_path(J, Q, current_node=0, destination_node=0):
    sum_costs = 0
    best_path = list()
    cum_cost = list()
    while current_node != destination_node:
        best_path.append(current_node)
        # Move to the next node and increment costs
        next_node = np.argmin(Q[current_node, :] + J)
        sum_costs += Q[current_node, next_node]
        cum_cost.append(sum_costs)
        current_node = next_node

    best_path.append(destination_node)
    return best_path, cum_cost

