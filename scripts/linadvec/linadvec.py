import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation,rc
import math
import matplotlib.animation as anime
import scipy.sparse.linalg as linalg
import scipy.sparse as sps


# Author: Bamischijf_275
#Description: A script that can solve the linear advection equation on a 1D grid, using various spatial discretizations as  well as time marches (to be implemented)


#linear advection equation: du/dt + c* du/dx = 0


#initial condition
dt = 0.1 #timestep
dx = 1
T = 20
Xtot = 100
c = 5 #speed of advection

v=0 #artificial dissipation included if v!=0
N = int(Xtot/dx) #number of mesh points (will always assume x[0,100])
x = np.arange(0,Xtot,dx)


u = np.zeros(int(1/10 * N))
u = np.append(u,np.ones(int(1/10 * N)))
u = np.append(u,np.zeros(int(8/10 * N)))
u = np.sin(2*math.pi*x*(1/100))

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
    A[0,1] = 1
    A[N-1,N-2] = -1
    A[0,N-1] = -1
    A[N-1,0] = 1
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
    A[N-1,N-1] = -2
    A[N-1,N-2] = 1
    A[0,N-1] = 1
    A[N-1,0]= 1
    A = A * (v/dx**2)
    return A





A_centr = centr_diff_dudx(N,c)
A_visc = art_visc(N,v)


fig,ax= plt.subplots()
line1, = ax.plot(x, u)

u_vec = []
u_vec.append(u)
def time_march(u,A_centr,A_visc):
    u = np.matmul((I + dt * (A_visc + A_centr)), u)
    u_vec.append(u)

#Save the solutions as a vector so it can be animated and saved
for h in np.arange(0,T,dt):
    u = np.matmul((I + dt * (A_visc + A_centr)), u)
    u_vec.append(u)
print(u_vec)


#function that can give a certain "frame"
def animate(i):
    line1.set_ydata(u_vec[i])
    return line1,

# Init only required for blitting to give a clean slate.
def init():
    line1.set_ydata(np.ma.array(x, mask=True))
    return line1,

anim = anime.FuncAnimation(fig,animate,np.arange(0,int(T/dt)),init_func=init,interval=25,blit=True)

#Save animation as gif
f = r"C://Users/degro/Desktop/animation.gif"
writergif = animation.PillowWriter(fps=30)
anim.save(f, writer=writergif)

plt.show()





