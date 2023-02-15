import matplotlib.pyplot as plt
import numpy as np


step = 32

x1 = np.linspace(0, 10, 1000)
f0 = 5
argument = 2 * np.pi * f0 * x1
y1 = np.cos(argument)

plt.plot(x1, y1, 'ro')
plt.xlabel('X')
plt.ylabel('cos^2(X)')
plt.show()
