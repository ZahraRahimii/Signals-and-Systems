import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

level = 0.001

def integral(x):
    y = np.sum(x) * level
    return y

t = np.arange(-4, 4, level)

def ak(x, k, T0):
    w0 = 2 * np.pi / T0
    param = k * w0 * t
    return 2/T0 * integral(x * np.cos(param))

def bk(x, k, T0):
    w0 = 2 * np.pi / T0
    param = k * w0 * t
    return 2/T0 * integral(x * np.sin(param))

def fourier_series(x, c, T0):
    w0 = 2 * np.pi / T0
    xx = np.zeros(len(t))
    for k in range(1, c, +1):
        xx += ak(x, k, T0) * np.cos(k * w0 * t)
    for k in range(1, c, +1):
        xx += bk(x, k, T0) * np.sin(k * w0 * t)
    a0 = ak(x, 0, T0)
    return a0/2 + xx

fig = plt.figure(figsize=(15, 5))
gs = GridSpec(nrows=1, ncols=2)


#####1
x1 = np.zeros(len(t))
for idx, val in enumerate(t):
    if (1000 <= idx and idx < 2000) :
        x1[idx] = val + 2
    if (2000 <= idx and idx < 4000) :
        x1[idx] = -val
    if (4000 <= idx and idx < 5000) :
        x1[idx] = val - 2

s1 = fig.add_subplot(gs[0, 0])
s1.plot(t, x1)
for c in range (0, 11):
    s1.plot(t, fourier_series(x1, c, 4))


#####2
s2 = fig.add_subplot(gs[0, 1])

x2 = np.zeros(len(t))
x2 = np.heaviside(t-3, 1) - np.heaviside(t, 1)
s2.plot(t, x2)
for c in range (0, 11):
    s2.plot(t, fourier_series(x2, c, 4))

plt.show()