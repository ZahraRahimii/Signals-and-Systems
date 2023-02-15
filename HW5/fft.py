import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import hilbert

def my_fft(x, Fs, M):
    signal = np.fft.fft(x, M)
    shift_fft = np.abs(np.fft.fftshift(signal))
    shift_t = np.linspace(-Fs/2, Fs/2, M)
    plt.plot(shift_t, shift_fft)

fs = 10000
fc = 1000
fm = 100
t0 = 0.1
M = 1024 * 1024

t = np.arange(0, t0, 1/fs)

x = np.sinc(100 * t)
# my_fft(x, fs, M)

x_mod = x * np.cos(2 * np.pi * fc * t) 
# my_fft(x_mod, fs, M)


x_demod = x_mod * np.cos(2*np.pi*fc*t)
# my_fft(x_demod, fs, M)


mat_contents = sio.loadmat('filter_coef.mat')['filter_coef']
mr = np.convolve(x_demod, np.reshape(mat_contents, mat_contents.size))
# my_fft(mr, fs, M)

plt.plot(mr)
plt.plot(2 * mr[int((len(mr) - len(x)) / 2) : int((len(mr) - len(x)) / 2) + len(x)])
plt.plot(x)

#   part 2
x_mod = x * np.cos(2 * np.pi * fc * t) - hilbert(x) * np.sin(2 * np.pi* fc * t)
x_mod = x * np.cos(2 * np.pi * fc * t) + hilbert(x) * np.sin(2 * np.pi* fc * t)
# plt.plot(x)
# plt.plot(hilbert(x))
# plt.plot(x_mod)
# my_fft(x, fs, M)

x_demod = x_mod * np.cos(2*np.pi*fc*t) 
# my_fft(x_demod, fs, M)

# my_fft(mr, fs, M)

x_mod = x * np.cos(2 * np.pi * fc * t)
x_demod = x_mod * np.cos(2*np.pi*fc*t) 
plt.plot(2 * x_demod[int((len(x_demod) - len(x)) / 2) : int((len(x_demod) - len(x)) / 2) + len(x)])
plt.plot(x)

plt.show()