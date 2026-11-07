import matplotlib.pyplot as plt
import numpy as np


##############################################################################################
##############################################################################################
##############################################################################################
# gradiens modositasai
##############################################################################################
##############################################################################################
##############################################################################################
# gradiens modszer fix lepeshosszal
def grad_meth2(f, g, x0, alfa, tol=1e-3, maxit=5000):
    n = 0
    x_matrix = [x0]
    while np.linalg.norm(g(x0)) > tol and n < maxit:
        x0 -= alfa * g(x0)
        n += 1
        x_matrix = np.append(x_matrix, [x0], axis=0)
    return x0, n, x_matrix


##############################################################################################


# momentum modszer
def momentum_m(f, g, x0, beta, alfa, tol=1e-3, maxit=5000):
    n = 0
    x_matrix = [x0]
    v = 0
    while np.linalg.norm(g(x0)) > tol and n < maxit:
        v = beta * v - alfa * g(x0)
        x0 += v
        n += 1
        x_matrix = np.append(x_matrix, [x0], axis=0)
    return x0, n, x_matrix


##############################################################################################
# pelda gradient vs momentum


def f_ex(x, y):
    # return x**2 + 20 * y**2
    return (x**4 / 4 + x**3 / 3 - x**2) + 1 + y**4 / 4 + y**3 / 3 - y**2 + 1


def g_ex(x):
    return np.array(
        [x[0] ** 3 + x[0] ** 2 - 2 * x[0], x[1] ** 3 + x[1] ** 2 - 2 * x[1]]
    )


f_exb = lambda x: f_ex(x[0], x[1])

xmin1, step1, xmatrix1 = grad_meth2(f_exb, g_ex, [-0.2, -0.8], 0.01)
xmin2, step2, xmatrix2 = momentum_m(f_exb, g_ex, [-0.2, -0.8], 0.5, 0.03)

print("GRADIENS VS MOMENTUM MODSZER")
print(f"A minimumhely kozelitese grad modszerrel: {xmin1}, a lepesek szama: {step1}")
print(
    f"A minimumhely kozelitese momentum modszerrel: {xmin2}, a lepesek szama: {step2}"
)
print("-" * 20)

xm = np.linspace(-3, 3, 100)
ym = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(xm, ym)
Z = f_ex(X, Y)

fig, axs = plt.subplots(2, 1)
axs[0].contour(X, Y, Z, levels=np.linspace(-3, 3, 15))
axs[0].plot(xmatrix1[:, 0], xmatrix1[:, 1], "r*--")
axs[1].contour(X, Y, Z, levels=np.linspace(-3, 3, 15))
axs[1].plot(xmatrix2[:, 0], xmatrix2[:, 1], "r*--")
fig.suptitle("Gradiens vs momentum modszer")
plt.show()

##############################################################################################
##############################################################################################
##############################################################################################
# adagrad


def adagrad(f, g, x0, alfa, tol=1e-3, maxit=5000):
    n = 0
    x_matrix = [x0]
    s = 0
    while np.linalg.norm(g(x0)) > tol and n < maxit:
        s = s + g(x0) ** 2
        x0 -= alfa / (np.sqrt(s) + 10e-5) * g(x0)
        n += 1
        x_matrix = np.append(x_matrix, [x0], axis=0)
    return x0, n, x_matrix


##############################################################################################
# pelda gradiens vs adagrad
xmin1, step1, xmatrix1 = grad_meth2(f_exb, g_ex, [-0.2, -0.8], 0.01)
xmin2, step2, xmatrix2 = adagrad(f_exb, g_ex, [-0.2, -0.8], 0.4)

print("GRADIENS VS ADAGRAD MODSZER")
print(f"A minimumhely kozelitese grad modszerrel: {xmin1}, a lepesek szama: {step1}")
print(f"A minimumhely kozelitese adagrad modszerrel: {xmin2}, a lepesek szama: {step2}")
print("-" * 20)

xm = np.linspace(-3, 3, 100)
ym = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(xm, ym)
Z = f_ex(X, Y)

fig, axs = plt.subplots(2, 1)
axs[0].contour(X, Y, Z, levels=np.linspace(-3, 3, 15))
axs[0].plot(xmatrix1[:, 0], xmatrix1[:, 1], "r*--")
axs[1].contour(X, Y, Z, levels=np.linspace(-3, 3, 15))
axs[1].plot(xmatrix2[:, 0], xmatrix2[:, 1], "r*--")
fig.suptitle("Gradiens vs adagrad modszer")
plt.show()


##############################################################################################
##############################################################################################
##############################################################################################
# rmsprop


def rmsprop(f, g, x0, beta, alfa, tol=1e-3, maxit=5000):
    n = 0
    s = 0
    x_matrix = [x0]
    while np.linalg.norm(g(x0)) > tol and n < maxit:
        s = beta * s + (1 - beta) * g(x0) ** 2
        x0 -= alfa / (np.sqrt(s) + 10e-5) * g(x0)
        n += 1
        x_matrix = np.append(x_matrix, [x0], axis=0)
    return x0, n, x_matrix


##############################################################################################
# pelda gradiens vs rmsprop
xmin1, step1, xmatrix1 = grad_meth2(f_exb, g_ex, [-0.2, -0.8], 0.01)
xmin2, step2, xmatrix2 = rmsprop(f_exb, g_ex, [-0.2, -0.8], 0.9, 0.2)

print("GRADIENS VS RMSPROP MODSZER")
print(f"A minimumhely kozelitese grad modszerrel: {xmin1}, a lepesek szama: {step1}")
print(f"A minimumhely kozelitese rmsprop modszerrel: {xmin2}, a lepesek szama: {step2}")
print("-" * 20)

xm = np.linspace(-3, 3, 100)
ym = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(xm, ym)
Z = f_ex(X, Y)

fig, axs = plt.subplots(2, 1)
axs[0].contour(X, Y, Z, levels=np.linspace(-3, 3, 15))
axs[0].plot(xmatrix1[:, 0], xmatrix1[:, 1], "r*--")
axs[1].contour(X, Y, Z, levels=np.linspace(-3, 3, 15))
axs[1].plot(xmatrix2[:, 0], xmatrix2[:, 1], "r*--")
fig.suptitle("Gradiens vs rmsprop modszer")
plt.show()


##############################################################################################
##############################################################################################
##############################################################################################
# adam


def adam(f, g, x0, beta1, beta2, alfa, tol=1e-3, maxit=5000):
    n = 0
    v = 0
    s = 0
    x_matrix = [x0]
    while np.linalg.norm(g(x0)) > tol and n < maxit:
        v = beta1 * v + (1 - beta1) * g(x0)
        s = beta2 * s + (1 - beta2) * g(x0) ** 2
        vkalap = v / (1 - beta1 ** (n + 1))
        skalap = s / (1 - beta2 ** (n + 1))
        x0 -= alfa * vkalap / (np.sqrt(skalap) + 10e-5)
        n += 1
        x_matrix = np.append(x_matrix, [x0], axis=0)
    return x0, n, x_matrix


##############################################################################################
# pelda gradiens vs adam
xmin1, step1, xmatrix1 = grad_meth2(f_exb, g_ex, [-0.2, -0.8], 0.01)
xmin2, step2, xmatrix2 = adam(f_exb, g_ex, [-0.2, -0.8], 0.8, 0.999, 0.2)

print("GRADIENS VS ADAM MODSZER")
print(f"A minimumhely kozelitese grad modszerrel: {xmin1}, a lepesek szama: {step1}")
print(f"A minimumhely kozelitese adam modszerrel: {xmin2}, a lepesek szama: {step2}")
print("-" * 20)

xm = np.linspace(-3, 3, 100)
ym = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(xm, ym)
Z = f_ex(X, Y)

fig, axs = plt.subplots(2, 1)
axs[0].contour(X, Y, Z, levels=np.linspace(-3, 3, 15))
axs[0].plot(xmatrix1[:, 0], xmatrix1[:, 1], "r*--")
axs[1].contour(X, Y, Z, levels=np.linspace(-3, 3, 15))
axs[1].plot(xmatrix2[:, 0], xmatrix2[:, 1], "r*--")
fig.suptitle("Gradiens vs adam modszer")
plt.show()

# összes kirajzolása
xmin1, step1, xmatrix1 = grad_meth2(f_exb, g_ex, [-0.2, -0.8], 0.01)
xmin2, step2, xmatrix2 = momentum_m(f_exb, g_ex, [-0.2, -0.8], 0.5, 0.03)
xmin3, step3, xmatrix3 = adagrad(f_exb, g_ex, [-0.2, -0.8], 0.4)
xmin4, step4, xmatrix4 = rmsprop(f_exb, g_ex, [-0.2, -0.8], 0.9, 0.2)
xmin5, step5, xmatrix5 = adam(f_exb, g_ex, [-0.2, -0.8], 0.8, 0.999, 0.2)

xm = np.linspace(-3, 3, 100)
ym = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(xm, ym)
Z = f_ex(X, Y)

fig, axs = plt.subplots(3, 2)
axs[0, 0].contour(X, Y, Z, levels=np.linspace(-3, 3, 15))
axs[0, 0].plot(xmatrix1[:, 0], xmatrix1[:, 1], "r*--")
axs[0, 0].set_title("Gradiens")
axs[0, 1].contour(X, Y, Z, levels=np.linspace(-3, 3, 15))
axs[0, 1].plot(xmatrix2[:, 0], xmatrix2[:, 1], "r*--")
axs[0, 1].set_title("Momentum")
axs[1, 0].contour(X, Y, Z, levels=np.linspace(-3, 3, 15))
axs[1, 0].plot(xmatrix3[:, 0], xmatrix3[:, 1], "r*--")
axs[1, 0].set_title("Adagrad")
axs[1, 1].contour(X, Y, Z, levels=np.linspace(-3, 3, 15))
axs[1, 1].plot(xmatrix4[:, 0], xmatrix4[:, 1], "r*--")
axs[1, 1].set_title("RMSprop")
axs[2, 0].contour(X, Y, Z, levels=np.linspace(-3, 3, 15))
axs[2, 0].plot(xmatrix5[:, 0], xmatrix5[:, 1], "r*--")
axs[2, 0].set_title("Adam")
plt.show()
