import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import math



def cfv(x):
    print(np.linalg.vector_norm(pontok - x))
    return np.max(np.linalg.vector_norm(pontok - x, axis=1))


pontok = np.array([[0,1], [2,3], [5,2]])

print(np.mean(pontok, axis=0))

res = minimize(cfv, np.mean(pontok, axis=0), method='BFGS')
print(res)

plt.plot(pontok[:, 0], pontok[:, 1], 'r*')
plt.plot(res.x[0], res.x[1], 'bo')
alfa = np.linspace(0, 2*math.pi)
xkor = res.x[0] + res.fun * np.cos(alfa)
ykor = res.x[1] + res.fun * np.sin(alfa)
plt.plot(xkor, ykor, 'g-')
plt.axis('equal')
plt.title('Egykor')
plt.show()