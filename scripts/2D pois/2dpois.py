import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation,rc
import math
import matplotlib.animation as anime
import scipy.sparse.linalg as linalg
import scipy.sparse as sp
from time import time
from enum import IntEnum
import matplotlib
import cmocean

#Code mainly from AE2220-II computational modelling course, coding example
#2D poisson solver: d^2u/dx^2 + d^2u/dy^2 = f
#u(x,1) = cos(2pix), u(1,y) = cos(2piy), du/dx (x,0) = -2pi*cos(2pix), du/dy(0,y) = -2picos(2piy)
# f = -8pi^2 cos(2pix)cos(2piy)  x[0,2] y [0,1]


class dir(IntEnum):
    N = 1
    E = 2
    S = 3
    W = 4
Lx = [0,2]
Ly = [0,1]

Nx = 240
Ny = 120

f = lambda x,y : -8*np.pi*np.pi*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)

x = np.linspace(Lx[0],Lx[1],Nx)
y=  np.linspace(Ly[0],Ly[1],Nx)
hx= x[1]-x[0]
hy= y[1]-y[0]


#Boundary conditions
north_bound  = ['Dirichlet', lambda x, y : np.cos(2*np.pi*x)*np.cos(2*np.pi*y)]             # north boundary [type, value]
east_bound   = ['Dirichlet', lambda x, y : np.cos(2*np.pi*x)*np.cos(2*np.pi*y)]             # east boundary [type, value]
south_bound  = ['Neumann', lambda x, y : -2*np.pi*np.cos(2*np.pi*x)*np.sin(2*np.pi*y)]      # south boundary [type, value]
west_bound   = ['Neumann', lambda x, y : -2*np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)]




#Connectivity
mesh_indices = -1*np.ones((Ny+2,Nx+2),dtype= np.int32)
mesh_indices[1:-1,1:-1] = np.arange(0,Nx*Ny,dtype = np.int32).reshape(Ny,Nx)

#connectivity in space

DOF_connectivity = np.empty((Ny,Nx,5),dtype = np.int32)
DOF_connectivity[:,:,0] = np.arange(0,Ny*Nx,dtype=np.int32).reshape(Ny,Nx)
DOF_connectivity[:,:,dir.N] = mesh_indices[2:,1:-1]
DOF_connectivity[:,:,dir.E] = mesh_indices[1:-1,2:]
DOF_connectivity[:,:,dir.S] = mesh_indices[:-2,1:-1]
DOF_connectivity[:,:,dir.W] = mesh_indices[1:-1,:-2]

boundary_idx = np.arange(0,Ny*Nx,dtype=np.int32).reshape(Ny,Nx)
boundary_idx[1:-1,1:-1] = -1
boundary_idx = boundary_idx[boundary_idx>=0] #only return boundary indices


#Solve
tc_start = time()

A = sp.dok_matrix((Nx*Ny,Nx*Ny),dtype = np.float64)
b = np.zeros(Ny*Nx)
A_tmp = {}

#constructing:

for j in range(Ny):
    for i in range(Nx):
        #indices
        idx = DOF_connectivity[j,i,0]
        Nidx = DOF_connectivity[j,i,dir.N]
        Eidx= DOF_connectivity[j,i,dir.E]
        Sidx = DOF_connectivity[j, i, dir.S]
        Widx = DOF_connectivity[j, i, dir.W]

        #Setup interior
        if idx not in boundary_idx:
            A_tmp[(idx,idx)] = -2/(hx**2) -2/(hy**2)
            A_tmp[(idx,Nidx)] = 1/(hy**2)
            A_tmp[(idx,Eidx)] = 1/(hx**2)
            A_tmp[(idx,Sidx)] = 1 / (hy ** 2)
            A_tmp[(idx,Widx)] = 1 / (hx ** 2)
        #boundary conditions
        if Eidx < 0:
            if  east_bound[0] == 'Dirichlet':
                A_tmp[idx,idx] = 1
            elif east_bound[0] == 'Neumann':
                A_tmp[idx,idx] = 1/hx
                A_tmp[idx,Widx] = -1/hx
        elif Widx < 0:
            if west_bound[0] == 'Dirichlet':
                A_tmp[idx, idx] = 1
            elif west_bound[0] == 'Neumann':
                A_tmp[idx, idx] = -1 / hx
                A_tmp[idx, Eidx] = 1 / hx
        elif Nidx < 0:
            if north_bound[0] == 'Dirichlet':
                A_tmp[idx, idx] = 1
            elif north_bound[0] == 'Neumann':
                A_tmp[idx,idx] = 1/hy
                A_tmp[idx,Sidx] = -1/hy
        elif Sidx < 0:
            if south_bound[0] == 'Dirichlet':
                A_tmp[idx,idx] = 1
            elif south_bound[0] == 'Neumann':
                A_tmp[idx,idx]  =-1/hy
                A_tmp[idx,Nidx] = 1/hy

#Convert to sparse matrix
A._update(A_tmp)
A = A.tocsr()
A.eliminate_zeros()

#construct RHS
for j in range(Ny):
    for i in range(Nx):
        #indices
        idx = DOF_connectivity[j,i,0]
        Nidx = DOF_connectivity[j,i,dir.N]
        Eidx= DOF_connectivity[j,i,dir.E]
        Sidx = DOF_connectivity[j, i, dir.S]
        Widx = DOF_connectivity[j, i, dir.W]
        if idx not in boundary_idx:
            b[idx] = f(x[i],y[j])
        else:
            if Eidx<0:
                b[idx] = east_bound[1](x[i],y[j])
            elif Widx<0:
                b[idx] = west_bound[1](x[i], y[j])
            elif Sidx<0:
                b[idx] = south_bound[1](x[i], y[j])
            elif Nidx<0:
                b[idx] = north_bound[1](x[i], y[j])

tc_end = time()

#Solve
ts_start = time()
u = linalg.spsolve(A,b)
ts_end = time()



## Print statistics
print(f"Matrix Construction Time \t= {(tc_end-tc_start):.3f}s")
print(f"Problem Solving Time \t\t= {(ts_end-ts_start):.3f}s")

#### ------------- ####
#### Plot solution ####
#### ------------- ####

# Graphing Parameters
texpsize= [26,28,30]
SMALL_SIZE  = texpsize[0]
MEDIUM_SIZE = texpsize[1]
BIGGER_SIZE = texpsize[2]

plt.style.use('grayscale')
plt.rc('font', size=MEDIUM_SIZE, family='serif')    ## controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)                ## fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)                ## fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)               ## fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)               ## fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)               ## legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)             ## fontsize of the figure title
plt.rc('text', usetex=False)
matplotlib.rcParams['lines.linewidth']  = 1.5
matplotlib.rcParams['figure.facecolor'] = 'white'
matplotlib.rcParams['axes.facecolor']   = 'white'
matplotlib.rcParams["legend.fancybox"]  = False

fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,9))
ax00 = ax[0,0].imshow(u.reshape(Ny,Nx), vmin = u.min(), vmax = u.max(), cmap=cmocean.cm.thermal,
                 extent = [x.min(), x.max(), y.min(), y.max()],
                 interpolation ='nearest', origin ='lower')
ax[0,0].set_xlabel(r"$x\,\,[-]$")
ax[0,0].set_ylabel(r"$y\,\,[-]$")
fig.colorbar(ax00)

plt.show()
