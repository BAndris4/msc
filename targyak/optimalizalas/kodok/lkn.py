import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import optimize


#################################################################################################
#1.feladat
"""t=np.array([1.0, 1.1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
f=np.array([8.0, 8.9, 9.0, 9.8, 10.0, 11.0, 11.5, 11.5, 12.5, 13.0, 13.7, 14.0])
z = np.polyfit(t,f,1)
print(z)

p = np.poly1d(z)
plt.plot(t,f,'r*')
plt.plot([t[0],t[-1]], [p(t[0]),p(t[-1])])
plt.show()"""

#################################################################################################

#2. feladat
"""t=np.array([2.1, 2.2, 2.3, 2.3, 2.5, 2.6, 2.8, 2.9])
f=np.array([2.5, 2, 2.5, 2.7, 3, 4, 5.4, 7])

z = np.polyfit(t,f,2)
print(z)

pontok = np.linspace(2,3)
p = np.poly1d(z)
plt.plot(t,f,'r*')
plt.plot(pontok, p(pontok))
plt.show()
"""

#################################################################################################
#3. feladat
"""t=np.array([0.1, 0.5, 1.2, 1.5, 2.0, 2.1, 2.4, 3.0, 3.2])
f=np.array([3.9, 2.6, -0.8, 0.3, 3.2, 3.8, 3.2, -0.7, -0.9])

A = np.vstack([np.ones(len(t)), np.cos(math.pi * t), np.sin(math.pi * t)]).T

# Solve 1
f2 = f[:, np.newaxis]
x = np.linalg.solve(A.T @ A, A.T @ f2)
print(x)

# Solve 2
res = np.linalg.lstsq(A, f)[0]
print(res)

# Solve 3
def func(t,a,b,c):
    return a + b * np.cos(math.pi * t) + c * np.sin(math.pi * t)

res = optimize.curve_fit(func, xdata = t, ydata = f)[0]
print(res) 

# Rajzolás
xx = np.linspace(0, 3.3, 100)
yy = x[0] + x[1] * np.cos(math.pi * xx) + x[2] * np.sin(math.pi * xx)
plt.plot(t,f,'r*')
plt.plot(xx, yy)
plt.show()
"""

# ----------------------------------
# 3 lépéses autoregresszív cucc

"""f = np.loadtxt(
    "idosor.txt",
    dtype=float
)

plt.plot(f, 'r')

A = np.hstack((f[2:79, np.newaxis], f[1:78, np.newaxis], f[:77, np.newaxis]))

y = f[3:80, np.newaxis]
alfa = np.linalg.lstsq(A, y)[0]
print(alfa)

z = f.copy()
for k in range(80,100):
    z[k] = alfa[0,0]*z[k-1]+alfa[1,0]*z[k-2]+alfa[2,0]*z[k-3]
    
plt.plot(z, 'b')
plt.show()"""

# --------------------------------
# Kimaradt pontok mintavételezés

"""data = np.loadtxt(
    "jel.txt",
    dtype=float
)
tp = list(range(201))

tpobs = np.round((100*data[:,0])).astype(int)
tpmiss = list(set(tp).difference(set(tpobs)))

egyseg = np.identity(201, dtype=int)

S = np.delete(egyseg, tpmiss, axis=0)
Sm = np.delete(egyseg, tpobs, axis=0)

D = np.zeros((199,201), int)
D[range(199), range(1,200)] = -2
D[range(199), range(199)] = 1
D[range(199), range(2, 201)] = 1


m1 = D @ Sm.T
y = data[:, 1]
m2 = -np.matmul((D @ S.T), y[:, np.newaxis])
z = np.linalg.lstsq(m1, m2)[0]

plt.plot(tpobs, y, 'b*')
plt.plot(tpmiss, z, 'r*')
plt.show()
"""
# -----------------------------------------------

# Zajos jel simitas

data = np.loadtxt(
    "zajosjel.txt",
    dtype=float
)

kontrol = 10
n = len(data[:,1])

D = np.zeros((n-1, n), int)
D[range(n-1), range(n-1)] = 1
D[range(n-1), range(1, n)] = -1

print(np.sqrt(kontrol)*D)
y = data[:, 1]
m1 = np.vstack((np.identity(n), np.sqrt(kontrol) * D))
m2 = np.concatenate((y, np.zeros(n-1)))

res = np.linalg.lstsq(m1, m2)[0]

plt.plot(data[:,0], data[:,1], 'r')
plt.plot(data[:,0], res, 'b')

plt.show()

########################################################################################################
# homerseklet BP
t=np.array([15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])
f=np.array([-1.7, 0.1, 5.2, 10.3, 15.8, 18.9, 21.1, 20.3, 16.1, 10.2, 4.2, 0.5])




########################################################################################################
# muhold
fi = np.deg2rad(np.array([48., 88., 150., 221., 247., 311., 359]))
r = np.array([4.32, 2.05, 1.18, 1.26, 1.52, 4.25, 9.98])
