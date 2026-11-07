import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint, LinearConstraint

##############################################################################################
##############################################################################################
##############################################################################################
# CONSTRAINED OPTIMIZATION
##############################################################################################
##############################################################################################
##############################################################################################


####################
# 1.FELADAT
####################
def objfun(x,y):
    return (x-1) ** 2 + y -2

objfunb = lambda x: objfun(x[0],x[1])

def c1(x,y):
    return (x+1) ** 2 + y ** 2 -9

constr = lambda x: c1(x[0],x[1])


linear_constraint = LinearConstraint([1,1], -np.inf, 2)
nonlinear_constraint = NonlinearConstraint(constr, -np.inf, 0)
res = minimize(objfunb,[0.,0.], method='trust-constr', constraints=[linear_constraint, nonlinear_constraint])
print('CONSTRAINED OPTIMIZATION')


#print(res)
print('The objective function: (x-1) ** 2 + y - 2')
print('The constraints:')
print('x + y <= 2')
print('(x+1) ** 2 + y ** 2 <= 9')
print(f'The minimizer: {res.x}')
print(f'The Lagrange multiplier for the linear constraint: {res.v[0]}, for the nonlinear constraint: {res.v[1]}')


xc = np.linspace(-5,5)
yc = xc.copy()
X, Y = np.meshgrid(xc,yc)
Z = objfun(X,Y)

plt.contour(X,Y,Z)
plt.contour(X,Y,c1(X,Y), levels=[0.0],colors='red')

def linc(x,y):
    return x + y -2

plt.contour(X,Y,linc(X,Y), levels=[0.0],colors='black')
plt.plot(res.x[0],res.x[1],'bo')
plt.axis('equal')
plt.title('Constrained optimization')
plt.show()

print('-'*60)


####################
# 2.FELADAT
####################
# A FUGGVENYEK A RAJZOLASHOZ
def c_obj(x,y):
    return 4 * x ** 2 - x - y - 2.5

def cons1(x,y):
    return -1.5 * x ** 2 + y ** 2 + 2 * x -1

def cons2(x,y):
    return 2 * x **2 + y ** 2 -2 *x -4.25

# A FUGGVENYEK AZ OPTIMALIZALOHOZ
c_obj2 = lambda x: c_obj(x[0],x[1])
cons1b = lambda x: cons1(x[0],x[1])
cons2b = lambda x: cons2(x[0],x[1])


nonlin_con1 = NonlinearConstraint(cons1b, 0, np.inf)
nonlin_con2 = NonlinearConstraint(cons2b, -np.inf, 0)
res2 = minimize(c_obj2, [0., 0.], method='trust-constr', constraints=[nonlin_con1, nonlin_con2])

print("----------------------------------------------------------------------")
print('The objective: 4 * x ** 2 - x - y -2.5')
print('The constraints:')
print('-1.5 * x ** 2 + y ** 2 + 2 * x -1 >= 0')
print('2 * x **2 + y ** 2 -2 *x -4.25 <= 0')
print(f'The minimizer with the built-in trust-constr method: {res2.x}')
print(f'The Lagrange multipliers: {res2.v}')


xc = np.linspace(-4, 4)
yc = np.linspace(-3, 3)
X, Y = np.meshgrid(xc,yc)
Z = c_obj(X,Y)
plt.contour(X,Y,Z)

plt.contour(X,Y,cons1(X,Y), levels=[0.0],colors='black')
plt.contour(X,Y,cons2(X,Y), levels=[0.0],colors='red')
plt.plot(res2.x[0],res2.x[1],'bo')
plt.axis('equal')
plt.title('Constrained optimization')
plt.show()






##############################################################################################
##############################################################################################
# BUNTETOFVEK modszerre
##############################################################################################
##############################################################################################

