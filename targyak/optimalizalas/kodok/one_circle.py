import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import minimize


#############################################################################################
#######################################################################################
# a síközpontok (x1,x2) koordinátája, soronként az A-ban
A=np.array([[2.5, 0.0],[0.0,0.0],[4.0,3.0],[1.0,4.0]])


# adott síközpontok esetén a legnagyobb távolság, 
# ha a bázisállomás koordinátái x-ben vannak
#
# A-x --> minden síközpont helyéből kivonja a bázis helyét
# vector_norm --> kiszámolja az eredménymátrix minden sorának 
# (mert axis=1) a normáját
def maxdist(x,A): 
	d=np.max(np.linalg.vector_norm((A-x),axis=1))
	return d 
	
obj = lambda x: maxdist(x,A)

# a minimalizálás, egy jó kezdeti közelítés a síközpontok 
# x1 és x2 koordinátáinak átlaga. Figyeljünk az axis=0-ra, 
# egyébként az A összes elemének átlagával tér vissza
res = minimize(obj,np.mean(A,axis=0),method='BFGS')
print(f'A bázisállomás koordinátái: {res.x}')
print(f'A legnagyobb távolság: {res.fun}')


# ábrázuljunk a síközpontokat és a bázist
plt.plot(A[:,0],A[:,1],'*')
plt.plot(res.x[0],res.x[1],'o')
plt.axis('equal')

# rajzoljuk meg a fedő kört. Ennek középpontja a 
# bázis helye, sugara a célfüggvény értéke itt
alfa = np.linspace(0,2*np.pi)
plt.plot(res.x[0]+res.fun*np.cos(alfa),res.x[1]+res.fun*np.sin(alfa))
plt.show()

