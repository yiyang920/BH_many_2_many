import networkx as nx
import numpy as np
from itertools import product
from random import sample
from collections import deque
from copy import deepcopy


def BFS(TN, C_init, VP):
    bridges = dict()
    seen = set(C_init)
    PV = {c: {c} for c in C_init}
    queue = deque(list(TN.neighbors(u) for u in C_init if u not in seen))
    while queue:
        u = queue.popleft()
        for v in TN.neighbors(u):
            if v not in seen:
                interset = set(TN.neighbors(v)) & seen
                if len(interset) > 1:
                    bridges[v] = set(VP[n] for n in interset)
                # random select a neighbor which belongs to one supernode
                candidate = sample(interset, 1)[0]
                TN.nodes[v]["part"] = TN.nodes[candidate]["part"]
                VP[v] = VP[candidate]
                PV[VP[candidate]].add(v)
                seen.add(v)
                N2 = list(set(TN.neighbors(u)) - interset)
                if N2:
                    queue.append(N2)
    return TN, bridges, PV, VP


def get_objective(D_uv, PV):
    return sum(
        sum(D_uv[u, v] for c2, p2 in PV.items() if c2 != c1 for v in p2)
        for c1, p1 in PV.items()
        for u in p1
    ) + sum(sum(D_uv[u, v] for (u, v) in product(p, p)) for p in PV.values())


def local_search(TN, D_uv, N, K, config):
    V = set(i for i in range(N))
    E = set((u, v) for (u, v) in TN.edges())
    LV = set((u, v, c) for (u, v, c) in product(V, V, V))
    EL = set((u, v, a, c) for ((u, v), a, c) in product(E, V, V))

    # Generate initial supernodes
    C_init = sample(V, K)
    VP = dict()
    for i, u in enumerate(C_init):
        TN.nodes[u]["part"] = i
        VP[u] = i
    # Generate initial partition by BFS
    TN, bridges, PV, VP = BFS(TN, C_init, VP)

    OBJ = get_objective(D_uv, PV)

    # Conduct local search
    # froze = set()
    PV_temp = deepcopy(PV)
    VP_temp = deepcopy(VP)
    FIND_GRADIANT = False
    while True:
        bridges_random = np.random.permutation(set(bridges.keys()))
        for v in bridges_random:
            if FIND_GRADIANT:
                break
            for part in bridges[v]:
                current_part = VP_temp[v]
                if part != current_part:
                    PV_temp[current_part].remove(v)
                    PV_temp[part].add(v)
                    VP_temp[v] = part

                    OBJ_new = get_objective(D_uv, PV_temp)
                    if OBJ_new < OBJ:
                        OBJ = OBJ_new
                        FIND_GRADIANT = True
                        # froze = {v}
                        break
                    # Recover mappings if OBJ not decreasing
                    PV_temp[current_part].add(v)
                    PV_temp[part].remove(v)
                    VP_temp[v] = current_part
        # Exit local search if cannot find any gradiant after searching all bridges
        break
    return PV_temp, VP_temp
