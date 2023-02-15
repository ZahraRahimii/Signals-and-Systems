import matplotlib.pyplot as plt
import numpy as np

def impulse(n, x):
    return 1 * (x == 0)

n = np.arange(-6, 6, +1)
yn = np.heaviside(n-3, 0) - np.heaviside(3-n, 0) 

plt.plot(n, yn, 'o')
plt.show()

