import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.sparse.linalg as linalg
import scipy.sparse as sps


#Author= Bamischijf_275

#1D poisson: d^2u/dx^2 = f
#x [0,1]
#Boundary conditions:
# du/dx (0) = 2pi
# u(1) = 0

dx = 0.01
N = int(1/dx)
x = np.arange(0,1,dx)
f = -4*math.pi**2*np.sin(2*math.pi*x)
f[0] = 2*math.pi
f[-1] = 0

#Construct A matrix
A = np.zeros((N,N))
for i in range(1,N-1):
    A[i,i-1] = 1
    A[i, i + 1] = 1
    A[i,i] = -2
A[0,0] = -dx
A[0,1] = dx
A[N-1,N-1] = dx**2
A = (1/dx**2)*A

u = linalg.spsolve(A,f)
plt.plot(x,u)
plt.show()
