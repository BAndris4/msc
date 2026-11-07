import math

import matplotlib.pyplot as plt
import numpy as np

#######################################################################################


# a celfuggveny
def fv(x):
    return x**4 / 4 + x**3 / 3 - x**2 + 1


#######################################################################################


# a celfuggveny deriváltja
def fvd(x):
    return x**3 + x**2 - 2 * x


#######################################################################################

# gradiens modszer
# input:
#        x0: kezdopont, n: (maximalis) lepesszam, alfa: lepeshossz
# output:
#        a bejart pontok (ezekben a celfuggveny erteke), elvegzett lepesek szama


def gradm(x0, n, alfa):
    pontok = [x0]
    ertekek = [fv(x0)]
    i = 0

    while i < n and abs(fvd(pontok[-1])) >= 10**-3:
        ujpont = pontok[-1] - alfa * fvd(pontok[-1])
        pontok.append(ujpont)
        ertekek.append(fv(ujpont))
        i += 1

    print("Pontok:", pontok)
    print("Függvényértékek:", ertekek)
    print("Lépések száma:", i)

    return pontok, ertekek, i


def graditer(x0, n, a0, c=0.95, mode=1):
    pontok = [x0]
    ertekek = [fv(x0)]
    i = 0

    while i < n and abs(fvd(pontok[-1])) >= 10**-3:
        if mode == 1:
            ujpont = pontok[-1] - (a0 * c**i) * fvd(pontok[-1])
        if mode == 2:
            ujpont = pontok[-1] - (a0 * math.e ** (-c * i)) * fvd(pontok[-1])
        if mode == 3:
            ujpont = pontok[-1] - (a0 / (1 + c * i)) * fvd(pontok[-1])
        pontok.append(ujpont)
        ertekek.append(fv(ujpont))
        i += 1
        print(f"{i}. learning rate: {a0 * c**i}")

    print("Pontok:", pontok)
    print("Függvényértékek:", ertekek)
    print("Lépések száma:", i)

    return pontok, ertekek, i


def gradadapt(x0, n, a0):
    pontok = [x0]
    ertekek = [fv(x0)]
    i = 0
    a = a0

    while i < n and abs(fvd(pontok[-1])) >= 10**-3:
        ujpont = pontok[-1] - a * fvd(pontok[-1])
        if fv(ujpont) < fv(pontok[-1]):
            print(f"{i}.: Learning rate: {a}")
            pontok.append(ujpont)
            ertekek.append(fv(ujpont))
            a *= 1.05
        else:
            a /= 2
            print(f"{i}.: Feleződés!")
            continue
        i += 1

    print("Pontok:", pontok)
    print("Függvényértékek:", ertekek)
    print("Lépések száma:", i)

    return pontok, ertekek, i


#######################################################################################


# A célfüggvény gráfja
xx = np.linspace(-3, 2, 100)
plt.xlim(-3, 2)
plt.plot(xx, fv(xx))
pontok1, ertekek1, lepesek1 = graditer(-2.4, 200, 0.5, mode=1)
pontok2, ertekek2, lepesek2 = graditer(-2.4, 200, 0.5, mode=2)
pontok3, ertekek3, lepesek3 = graditer(-2.4, 200, 0.5, mode=3)
pontok4, ertekek4, lepesek4 = gradadapt(-2.4, 200, 0.5)
plt.plot(pontok1, ertekek1, "ro-")
plt.plot(pontok2, ertekek2, "bo-")
plt.plot(pontok3, ertekek3, "go-")
plt.plot(pontok4, ertekek4, "yo-")

plt.xlabel("x")
plt.ylabel("Célfüggvény értéke")
plt.title("Célfüggvény és a gradiens módszer iterációi")
plt.legend(["Célfüggvény", "Mode 1", "Mode 2", "Mode 3", "Adaptive"])
plt.grid(True)

#######################################################################################

# A célfüggvény értéke a lépésszám függvényében
plt.figure()
plt.plot([i for i in range(1, lepesek1 + 1)], ertekek1[1:], "ro-")
plt.plot([i for i in range(1, lepesek2 + 1)], ertekek2[1:], "bo-")
plt.plot([i for i in range(1, lepesek3 + 1)], ertekek3[1:], "go-")
plt.plot([i for i in range(1, lepesek4 + 1)], ertekek4[1:], "yo-")
plt.xlabel("Lépésszám")
plt.ylabel("Célfüggvény értéke")
plt.title("Célfüggvény értéke a lépésszám függvényében")
plt.legend(["Mode 1", "Mode 2", "Mode 3", "Adaptive"])
plt.grid(True)
plt.show()
