# task2 megoldasa (gradient, SGD, mini-batch)

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.optimize import minimize
import math


data = np.loadtxt(
    "C:/Users/Andris/Desktop/msc/targyak/optimalizalas/pontok200E.txt",
    dtype=float,
)

#############################################################################################
# az adatok abrazolasa
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(data[:, 0], data[:, 1], data[:, 2])
plt.title("A pontok (pontok200E.txt)")
plt.show()


# a modell
def fun_p(w, x, y):
    return (
        w[0] * np.sin(math.pi * x) * np.exp(-w[1] * np.square(x) - w[2] * np.square(y))
    )


# a celfuggveny
def fun_m(w, d):
    # w: sulyok, d: adatmatrix
    return np.mean(np.square(fun_p(w, d[:, 0], d[:, 1]) - d[:, 2]))


# a celfv gradiense
def fun_m_grad(w, d):
    # w: sulyok, d: adatmatrix
    c = 2 * (fun_p(w, d[:, 0], d[:, 1]) - d[:, 2])
    g1 = (
        c
        * np.sin(math.pi * d[:, 0])
        * np.exp(-w[1] * np.square(d[:, 0]) - w[2] * np.square(d[:, 1]))
    )
    g2 = c * (-np.square(d[:, 0]) * fun_p(w, d[:, 0], d[:, 1]))
    g3 = c * (-np.square(d[:, 1]) * fun_p(w, d[:, 0], d[:, 1]))
    return np.array([np.mean(g1), np.mean(g2), np.mean(g3)])


##############################################################################################
##############################################################################################

fun1 = lambda w: fun_m(w, data)
fun_1_g = lambda w: fun_m_grad(w, data)
##############################################################################################
##############################################################################################


##############################################################################################
# mini-bath SGD. Itt a lepeshossz fix.
import random


def batch_sgd(data0, f, g, x0, eta, bs=1, tol=1e-3, maxit=5000):
    # data0: adattomb
    # f,g: a celfv es a gradiense
    # x0: kezdopont
    # eta: lepeshossz
    # bs:  batch size (bs = 1 --> SGD)
    epoch = 0
    f_matrix = [f(x0, data0)]
    n_data = data0.shape[0]
    # hany batch lesz (lehetnek kimarado elemek)
    m = n_data // bs
    ind = list(range(n_data))
    while np.linalg.norm(g(x0, data0)) > tol and epoch < maxit:
        random.shuffle(ind)
        for k in range(m):
            batch = data0[ind[k * bs : (k + 1) * bs], :]
            grad_val = g(x0, batch)
            x0 -= eta * grad_val
            f_matrix = np.append(f_matrix, [f(x0, data0)], axis=0)
        epoch += 1

    return x0, epoch, f_matrix


##############################################################################################
# az eredmenyek
batch_s = 8
wopt, n, f_matrix = batch_sgd(
    data, fun_m, fun_m_grad, [0.0, 0.1, 0.1], eta=0.1, bs=batch_s
)
print(
    f"A parameterek becslese minibatch SGD eseten: {wopt}, amikor a batch size: {batch_s}"
)

plt.plot(f_matrix)
plt.title(f"A celfuggveny, mini batch SGD, bs: {batch_s}, epochszam: {n}")
plt.show()
