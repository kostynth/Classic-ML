import numpy as np
qweqw

def func(x):
    return -0.5 * x + 0.2 * x ** 2 - 0.01 * x ** 3 - 0.3 * np.sin(4*x)
def df(x):
    return -0.5 + 0.4 * x - 0.03 * x ** 2 - 1.2 * np.cos(4 * x)
x = -3.5
nu = 0.1
N = 200
gamma = 0.8
v = 0

for i in range(N):
    v = gamma * v + (1- gamma) * df(x)
    x = x - nu * v