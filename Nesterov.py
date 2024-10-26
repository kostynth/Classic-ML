import numpy as np


def func(x):
    return 0.4 * x + 0.1 * np.sin(2*x) + 0.2 * np.cos(3*x)
def df(x):
    return 0.4 + 0.2 * np.cos(2*x) - 0.6 * np.sin(3 * x)
nu = 1
x = 4
N = 500
gamma = 0.7
v = 0

for i in range(N):
    v = gamma * v + (1 - gamma) * nu * df(x  - gamma * v)
    x = x -   v