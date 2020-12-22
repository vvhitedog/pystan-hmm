#!/usr/bin/python

import numpy as np
from matplotlib import pyplot as plt

theta = np.asarray([[.8,.2],[.1,.9]])

psi = np.asarray([3.,9.])

N = 100
K = 2

# sample z
z = np.empty(N,dtype='int')
z[0] = 1
for i in range(1,N):
    z[i] = np.random.choice(np.arange(K),size=1,replace=True,p=theta[z[i-1]])

y = np.random.randn(N) + psi[z]

plt.figure()
plt.subplot(2,1,1)
plt.plot(z)
plt.subplot(2,1,2)
plt.plot(y)
plt.show()

np.savez_compressed('hmm_data',y=y,z=z)
