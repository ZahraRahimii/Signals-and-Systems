import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def unit_impulse(x):
    return 1 * (x == 0)

def unit_impulse_array(n):
    res = np.ndarray(shape=n.shape)
    for i in list(n):
        res[i] = unit_impulse(n[i])
    return res


def convolution(x, h):
    level = len(h)
    x = np.concatenate([np.zeros(level-1) , x , np.zeros(level-1)])
    h = h[::-1]
    y = np.zeros(len(x) - level +1)
    for i in range(len(y)):
        y[i] = np.sum(x[i : i + level] * h)
    return y[int(len(y)/4): -1 * int(len(y)/4) -1]

fig = plt.figure(figsize=(24, 15))
gs = GridSpec(nrows=2, ncols=2)
ax0 = fig.add_subplot(gs[0, 0])

## a)
n = np.arange(-10, 10, 1)
xa = np.power(np.e, 2 * n)*(-1 * np.heaviside(n-2, 1) + np.heaviside(n+3, 1))

# a) 1.
ha1 = np.heaviside(n+10, 1) - np.heaviside(n, 1)
ya1 = convolution(ha1, xa)
ax0 = fig.add_subplot(gs[0, 0])
# fig, s1 = plt.subplots()
ax0.stem(n, xa, 'g', markerfmt="go", label="x[n]=e^2b(−u[n−2]+u[n+3])" )
ax0.stem(n, ha1, 'b', markerfmt="bo", label="h[n]=u[n+10]−u[n]")
ax0.stem(n, ya1, 'y', markerfmt="yo", label= "x[n]*h[n]")
ax0.set(title="(a. 1)")
ax0.set_xlabel('n')
ax0.legend()
ax0.grid()

# a) 2.
ha2 = 3 * unit_impulse_array(n - 5) - unit_impulse_array(n)
ya2 = convolution(ha2, xa)
# fig, s2 = plt.subplots()
ax1 = fig.add_subplot(gs[0, 1])
ax1.stem(n, xa, 'g', markerfmt="go", label="x[n]=e^2b(−u[n−2]+u[n+3])" )
ax1.stem(n, ha2, 'b', markerfmt="bo", label="h[n]=3.impulse(n-5)-impulse(n)")
ax1.stem(n, ya2, 'y', markerfmt="yo", label= "x[n]*h[n]")
ax1.set(title="(a. 2)")
ax1.set_xlabel('n')
ax1.legend()
ax1.grid()

## b)
t = np.arange(-15, 15, 0.1)
xb = np.power(1/4, 2 * t) * np.heaviside(t+3, 1)

# b) 1.
hb1 = np.abs(t) * (np.heaviside(t - 2, 1) - np.heaviside(t, 1))
yb1 = convolution(hb1, xb)
# fig, s3 = plt.subplots()
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(t, xb, 'g', label="x(t) = (1/4)^2t.u(t+3)")
ax2.plot(t, hb1, 'b', label="h(t) = |t|.(u(t-2)-u(t))")
ax2.plot(t, yb1, 'y', label="y(t) = x(t)*h(t)")
ax2.set(title = '(b. 1)')
ax2.set_xlabel('t')
ax2.grid()
ax2.legend()

# b) 2.
hb2 =  np.heaviside(t, 1) - np.heaviside(t - 5, 1)
yb2 = convolution(xb, hb2)
# fig, s4 = plt.subplots()
ax2 = fig.add_subplot(gs[1, 1])
ax2.plot(t, xb, 'g', label="x(t) = (1/4)^2t.u(t+3)")
ax2.plot(t, hb2, 'b', label="h(t) = u(t+5)")
ax2.plot(t, yb2, 'y', label="y(t) = x(t)*h(t)")
ax2.set(title = '(b. 2)')
ax2.set_xlabel('t')
ax2.grid()
ax2.legend()

plt.show()