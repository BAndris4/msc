import matplotlib.pyplot as plt
import numpy as np


# negativ gradiensmezo
def gradmezo(fun, a=[-5, 5, -5, 5], normalt=False, m=None):
    # az m-ben meg lehet adni rarajzolando pontokat
    # ha normalt=True, akkor normalt negativ gradiensmezo
    xa = np.linspace(a[0], a[1], 100)
    xb = np.linspace(a[2], a[3], 100)
    xmatrix, ymatrix = np.meshgrid(xa, xb)
    zmatrix = fun(xmatrix, ymatrix)

    xga = np.linspace(a[0] + 0.5, a[1] - 0.5, 17)
    xgb = np.linspace(a[2] + 0.5, a[3] - 0.5, 17)
    xgrad, ygrad = np.meshgrid(xga, xgb)
    zgrad = fun(xgrad, ygrad)
    # az orai peldara a pontos gradienssel
    # u =  xgrad ** 3 + xgrad ** 2 - 2 * xgrad
    # v =  ygrad ** 3 + ygrad ** 2 - 2 * ygrad
    # q = plt.quiver(xgrad,ygrad,-u,-v)

    # numerikus gradienssel
    u, v = np.gradient(zgrad)
    if normalt:
        hossz = np.sqrt(u**2 + v**2)
        u = u / hossz
        v = v / hossz
    # a kov sorban u es v forditva!!!!
    q = plt.quiver(xgrad, ygrad, -v, -u)
    plt.contour(xmatrix, ymatrix, zmatrix, levels=np.linspace(-3, 3, 15))
    if m is not None:
        plt.plot(m[:, 0], m[:, 1], "r*")
    plt.axis("equal")
    plt.title("Negativ gradiensmezo")
    plt.show()


##############################################################################################
# a celfuggveny
def obj2(x, y):
    return (x**4 / 4 + x**3 / 3 - x**2) + 1 + y**4 / 4 + y**3 / 3 - y**2 + 2


stacp = np.array(
    [[0, 0], [1, 0], [-2, 0], [0, 1], [0, -2], [1, 1], [1, -2], [-2, 1], [-2, -2]]
)

gradmezo(obj2, a=[-3, 2, -3, 2], m=stacp)
