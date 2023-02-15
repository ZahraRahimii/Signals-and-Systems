import matplotlib.pyplot as plt
import numpy as np

x1 = np.linspace(-6, 6, 1000)
y1 = 0

for i in range(-20, 20, +1):
    y1 += np.exp(-1 * np.abs(2*x1 + i))

plt.plot(x1, y1, 'blue')
plt.show()