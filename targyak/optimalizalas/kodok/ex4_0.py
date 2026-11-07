import matplotlib.pyplot as plt
import numpy as np


# a celfuggveny
def fv(x, y):
    return (x**4 / 4 + x**3 / 3 - x**2) + 1 + y**4 / 4 + y**3 / 3 - y**2 + 1


##############################################################################################
# a celfuggveny gradiense
def fvgrad(x, y):
    return np.array([(x**3 + x**2 - 2 * x), y**3 + y**2 - 2 * y])


##############################################################################################

fv1 = lambda x: fv(x[0], x[1])
g1 = lambda x: fvgrad(x[0], x[1])


##############################################################################################
##############################################################################################
# gradiens modszer, fix lepeshosszal
# input:
#   fun: a celfuggveny
#   fund: a celfuggveny gradiense
#   x0: kezdovektor
#   alfa: lepeshossz
#   n: maximalis lepesszam
#   tol: tolerancia a leallasi feltetelhez
# output:
#   pontok: a bejart pontok
#   fval: a fuggveny erteke a bejart pontokban
#   k: az elvegzett lepesek szama
def gradm(fun, fund, x0, alfa, n, tol):
    pontok = [x0]
    fval = [fun(x0)]

    k = 0
    while k < n and np.linalg.norm(fund(pontok[-1])) >= tol:
        grad = fund(pontok[-1])

        x = pontok[-1] - alfa * grad
        pontok.append(x)
        fval.append(fun(x))
        k += 1

    return np.array(pontok), np.array(fval), k


##############################################################################################
##############################################################################################


# gradiens modszer, lepeshosszvalasztassal.
# itt alfa a kezdolepeshossz
def gradm2(fun, fund, x0, alfa, n, tol):

    return  # pontok, fval, k


##############################################################################################


##############################################################################################
# teszt, rajzolas
##############################################################################################

p, f, m = gradm(fv1, g1, np.array([-0.2, -0.8]), 0.01, 400, 1e-3)
# p2,f2,m2 = gradm2(fv1,g1,np.array([-0.2,-0.8]),0.01,400,1e-3)

# a celfüggveny szintvonalai
xx = np.linspace(-3, 3, 100)
yy = xx.copy()
xmatrix, ymatrix = np.meshgrid(xx, yy)
zmatrix = fv(xmatrix, ymatrix)
plt.contour(xmatrix, ymatrix, zmatrix, levels=np.linspace(-3, 3, 15))
plt.axis("equal")
plt.plot(p[:, 0], p[:, 1], c="r")
plt.figure()
##############################################################################################

# a celfuggveny valtozasa az elvegzett lepesek szamanak fuggvenyeben
plt.plot(np.linspace(0, m, m + 1), f, c="r")
# plt.plot(np.linspace(0,m2,m2+1),f2,c='b')
plt.xlabel("lepesek szama")
plt.ylabel("fuggvenyertek")
plt.show()
