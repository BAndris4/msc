import matplotlib.pyplot as plt
import numpy as np


def fun(x):
    return 100 * (x[1, 0] - x[0, 0] ** 2) ** 2 + (1 - x[0, 0]) ** 2


def grad(x):
    return np.array(
        [
            [-400 * x[0, 0] * (x[1, 0] - x[0, 0] ** 2) + 2 * x[0, 0] - 2],
            [200 * (x[1, 0] - x[0, 0] ** 2)],
        ]
    )


def hesse(x):
    return np.array(
        [
            [1200 * x[0, 0] ** 2 - 400 * x[1, 0] + 2, -400 * x[0, 0]],
            [-400 * x[0, 0], 200],
        ]
    )


def newton(x, eps):
    pontok = np.array(x)
    count = 0
    while np.linalg.norm(grad(x)) > eps:
        p = -np.linalg.inv(hesse(x)) @ grad(x)
        x = x + p
        count += 1
        pontok = np.append(pontok, x, axis=1)
    return pontok, count


def gradm(x, alfa, maxn):
    pontok = np.array(x)
    count = 0

    while count < maxn and np.linalg.norm(grad(x)) >= 1e-3:
        x = x - alfa * grad(x)
        pontok = np.append(pontok, x, axis=1)
        count += 1
    return pontok, count


# print(f"Gradiens: \n {grad(np.array([[10],[10]]))}")
# print(f"Hesse: \n {np.linalg.inv(hesse(np.array([[10],[10]])))}")

x = np.linspace(-1.5, 2, 100)
y = np.linspace(-3, 4, 100)

X, Y = np.meshgrid(x, y)
Z = fun(np.array([[X], [Y]]))
plt.figure(figsize=(8, 6))

npontok, ncount = newton(np.array([[-1.2], [1]]), 1e-8)
gpontok, gcount = gradm(np.array([[-1.2], [1]]), 1e-3, 100000)

print(f"Newton: {ncount} lépés")
print(f"Gradiens: {gcount} lépés")

contour = plt.contour(X, Y, Z, levels=50)
plt.plot(-1.2, 1, "r*")
plt.plot(npontok[0, :], npontok[1, :], "b-")
plt.plot(gpontok[0, :], gpontok[1, :], "g-")
plt.show()
