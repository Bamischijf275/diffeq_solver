
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation,rc
import math
import matplotlib.animation as anime
import scipy.sparse.linalg as linalg
import scipy.sparse as sps

#Author = Bamischijf_275
#Solve 1D heat equation: du/dt = v*(d^2u/dx^2)
#Will make use of finite difference discretisation

#Properties
T=5
Lt = [0,T] #Time length
Lx = [0,10]  #Space lenght

dt = 0.001
v = 0.2
Nx = 500 #Mesh points of our solution

x = np.linspace(Lx[0],Lx[1],Nx) #Create mesh space
t_vec = np.linspace(Lt[0],Lt[1],int((Lt[1]-Lt[0])/dt))

dx = x[1]-x[0]


#Initial condition: constant temperature of 293.15 Kelvin
u0 = np.full((np.shape(x)),293.15+50) - 60*x
u0 = np.full((np.shape(x)),293.15)

#Boundary conditions: On the right hand side 293.15 + 50, left hand side 293.15 - 10
u_bc_l = np.full((np.shape(t_vec)), (293.15+50))
u_bc_r = np.full((np.shape(t_vec)), (293.15-10))



def A_gen(Nx,v,dt,dx):
    I = np.identity(Nx)
    g = v*dt/dx**2
    print(g)
    mat = np.zeros((Nx,Nx))
    for j in range(1,Nx-1):
        mat[j,j+1] = 1
        mat[j,j] = -2
        mat[j,j-1] = 1
    mat = g*mat
    A = I + mat
    A[0,0] = 1
    A[Nx-1,Nx-1] = 1
    return A


u_vec = []
u = u0
u_vec.append(u)
A = A_gen(Nx,v,dt,dx)
fig,ax= plt.subplots()
line1, = ax.plot(x, u)
for time in t_vec:
    u_new = np.matmul(A,u)
    u_new[0] = 293.15-10
    u_new[-1] = 293.15-10
    u_new[int(Nx/2)] = 293.15-10
    u_vec.append(u_new)
    u = u_new
    

#function that can give a certain "frame"
def animate(i):
    line1.set_ydata(u_vec[i])
    return line1,

# Init only required for blitting to give a clean slate.
def init():
    line1.set_ydata(np.ma.array(x, mask=True))
    return line1,

    
anim = anime.FuncAnimation(fig,animate,np.arange(0,int(T/dt)),init_func=init,interval=0.2,blit=True)

#Save animation as gif
f = r"C://Users/degro/Desktop/animation.gif"
writergif = animation.PillowWriter(fps=30)
#anim.save(f, writer=writergif)

plt.show()




