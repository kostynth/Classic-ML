import numpy as np


def func(x):
    return 2 * x + 0.1 * x ** 3 + 2 * np.cos(3*x)
def df(x):
    return 2 + 0.3 * x ** 2 - 6 * np.sin(3 * x)

x = 4
N  = 200
alfa = 0.8
nu = 0.5
eta = 0.01
G  = 0

for _ in range(N):
    G = alfa * G + (1 - alfa) * df(x) ** 2
    x = x - nu * df(x) / (G ** (1/2) + eta)