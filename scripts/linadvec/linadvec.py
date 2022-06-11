import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation,rc
import math


# Author: Bamischijf_275
#Description: A script that can solve the linear advection equation on a 1D grid, using various spatial discretizations as  well as time marches (to be implemented)


#linear advection equation: du/dt + c* du/dx = 0


#initial condition
dt = 0.1 #timestep
c = 2 #speed of advection
dx = dt
v=0
N = int(100/dt) #number of mesh points (will always assume x[0,100])
x = np.arange(0,100,dt)

U_BC = np.zeros(int(100/dt))

u = np.zeros(int(1/10 * N))
u = np.append(u,np.ones(int(1/10 * N)))
u = np.append(u,np.zeros(int(8/10 * N)))
I = np.identity(N)

#create A







#Central difference du/dx
def centr_diff_dudx(N,c):
    #################################
    #Grabs an empty matrix A and converts it to one that creates the central difference matrix
    #C is the convection speed
    #################################
    A = np.zeros((N, N))

    # move along diagnals to make A:
    for i in range(1, N - 1):
        A[i, i - 1] = -1
        A[i, i + 1] = 1

    # Implement numerical boundary conditions at end
    A[N-1, N-1 - 2] = 1
    A[N-1, N-1 - 1] = -4
    A[N-1, N-1] = 3
    A = A * (-c /(2*dx))
    return A

#artificial viscocity
def art_visc(N,v):
    A = np.zeros((N,N))
    #uses central difference except at the most left boundary (numerical on the right)
    for i in range(1,N-1):
        A[i,i] = -2
        A[i,i-1] = 1
        A[i,i+1] = 1
    A[N-1,N-1-2] = 1
    A[N - 1, N - 1 - 1] = -2
    A[N - 1, N - 1  ] = 1
    A = A * (v/dx**2)
    return A





A_centr = centr_diff_dudx(N,c)
A_visc = art_visc(N,v)
A = A_centr + A_visc
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(x, u)


#plotting
def explicit_euler(u,dt):
    return np.matmul((I + dt*A),u)

def dynplot(u):
    t=0
    i=0
    running = True
    while running:
        u = np.matmul((I + dt*A),u)
        line1.set_ydata(u)
        fig.canvas.draw()
        fig.canvas.flush_events()
        t+=dt
        if t>=10:
            running = False

dynplot(u)







