#!/usr/bin/python

import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
from names import names
from draw_graph import draw_tree

theta = np.asarray([[.65, .35], [.1, .9]])
mu = np.asarray([50., 100.])
sigma = 15.  # std-deviation of both normal distributions

K = 2

use_random_tree = True
use_random_z0_init = False

if use_random_tree:
    N = 100
    T = nx.random_tree(N)
else:
    T = nx.balanced_tree(2,6)
    N = len(T.nodes())

D = max([d for n, d in T.degree()])
A = -np.ones([N, D], dtype='int')  # adjacency
C = np.zeros(N, dtype='int')  # edge count
O = np.zeros(N, dtype='int')  # visit order

edges = T.edges()
nodes = T.nodes()

E = dict()
for (u, v) in edges:
    if u not in E:
        E[u] = list()
    if v not in E:
        E[v] = list()
    E[u].append(v)
    E[v].append(u)

visited = set()

# sample z
z = np.empty(N, dtype='int')
if use_random_z0_init:
    z[0] = np.random.choice(np.arange(K), size=1, replace=True, p=theta[0])
else:
    z[0] = 1
q = list([0])

i = 0
while len(q) != 0:
    s = q[0]
    del q[0]
    O[i] = s
    i += 1
    if s in visited:
        continue
    visited.add(s)
    for ngb in E[s]:
        if ngb not in visited:
            q.append(ngb)
            z[ngb] = np.random.choice(
                np.arange(K), size=1, replace=True, p=theta[z[s]])
            A[s, C[s]] = ngb
            C[s] += 1

y = np.random.randn(N)*sigma + mu[z]

labels = {n: (str(names[n]) + "\n" + str(int(10*v)/10.) + " ({type})".format(type=('G' if z[n] else 'B')) + ("" if n != 0 else "\n==ROOT=="))
          for n, v in zip(nodes, y)}

draw_tree(edges, labels)

np.savez_compressed('tree_data', D=D, A=A, C=C, y=y, z=z, sigma=sigma, O=O)
