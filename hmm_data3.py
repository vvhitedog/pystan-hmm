#!/usr/bin/python

import numpy as np
from matplotlib import pyplot as plt


theta = np.asarray([[.8,.1,.1],[.1,.6,.3],[.4,.5,.1]])

psi = np.asarray([3.,9.,16.])

N = 100
K = 3

for j in range(1,3):
    # sample z
    z = np.empty(N,dtype='int')
    z[0] = 0
    for i in range(1,N):
        z[i] = np.random.choice(np.arange(K),size=1,replace=True,p=theta[z[i-1]])

    y = np.random.randn(N) + psi[z]

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(z)
    plt.subplot(2,1,2)
    plt.plot(y)
    plt.show()

    np.savez_compressed('hmm_data3-{j}'.format(j=j),y=y,z=z)
