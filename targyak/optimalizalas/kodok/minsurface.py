import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import minimize


#############################################################################################
#######################################################################################

# the frame
def frame(x,y):
    #return 0.5-np.abs(y-0.5)
    return np.sqrt(0.25 -(y -0.5) ** 2)

##########################################################################################
##########################################################################################

# gridpoints
def grid_points(n,m):
    x = np.linspace(0,1,n)
    y = np.linspace(0,1,m)
    xm, ym = np.meshgrid(x,y)
    return xm, ym

##########################################################################################
##########################################################################################


def areaofsurf(vertices,z,n,m):
    # vertices: 3D array of the vertices
    vertices[1:-1,1:-1,2] = np.reshape(z, shape=(m-2,n-2))

    # the area of the small triangles
    area1 = np.linalg.vector_norm(np.cross(vertices[:-1,:-1,:]-vertices[1:,:-1,:], vertices[1:,1:,:]-vertices[1:,:-1,:]), axis=2)
    area2 = np.linalg.vector_norm(np.cross(vertices[:-1,:-1,:]-vertices[:-1,1:,:], vertices[1:,1:,:]-vertices[:-1,1:,:]), axis=2)
    return np.sum(area1+area2) / 2


##########################################################################################
##########################################################################################

# plot the frame

X, Y = grid_points(50,50)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(X[0,:],Y[0,:],frame(X[0,:],Y[0,:]),color='red')
ax.plot(X[-1,:],Y[-1,:],frame(X[-1,:],Y[-1,:]),color='red')
ax.plot(X[:,0],Y[:,0],frame(X[:,0],Y[:,0]),color='red')
ax.plot(X[:,-1],Y[:,-1],frame(X[:,-1],Y[:,-1]),color='red')
plt.title('The frame')
plt.show()

###########################################################################################
#####################################################################################


n=10
m=11

xx, yy = grid_points(n,m)
zf = frame(xx,yy)



vertex_matrix = np.zeros((m,n,3))
vertex_matrix[:,:,0] = xx
vertex_matrix[:,:,1] = yy
vertex_matrix[:,:,2] = zf

z0 = np.reshape(zf[1:-1,1:-1],shape=((n-2)*(m-2),))

objf = lambda z: areaofsurf(vertex_matrix,z,n,m)


zopt = minimize(objf, z0, method = 'BFGS')
print(zopt)
zz = zf.copy()
zz[1:-1,1:-1] = np.reshape(zopt.x, shape=(m-2,n-2))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(xx,yy,zz)
plt.show()
