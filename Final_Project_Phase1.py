import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


name = "phase1sample.wav"

sample_rate = 16000
dt = 0.00001
t = np.arange(0, 1, dt)
M = len(t)
freq = (1/(dt * M)) * np.arange(M)
first_half = np.arange(1, np.floor(M/2), dtype=np.int32)
shift_t = np.linspace(-sample_rate/2, sample_rate/2, M)

def audio(name):
    signal_wave = wave.open(name, 'r')
    sig = np.frombuffer(signal_wave.readframes(sample_rate), dtype=np.int16)
    sig = sig[1:250]
    return sig

def my_fft(x, M):
    signal = np.fft.fft(x, M)
    shift_fft = np.abs(np.fft.fftshift(signal))
    return shift_fft
    # return signal

def my_fft2(x, M):
    signal = np.fft.fft(x, M)
    arg = signal * np.conj(signal) / M

    shift_fft = np.abs(arg[first_half])
    return shift_fft


sig = audio(name)
ft_sig = my_fft(sig, M)


real_fft_x = np.fft.fft(sig, M)
arg = real_fft_x * np.conj(real_fft_x) / M
sort_fft = np.sort(ft_sig)[::-1]
treshold = sort_fft[99999]
denoised_ft =  (arg > treshold) * arg 

retrived_sig = denoised_ft
fitered_sig = np.fft.ifft(retrived_sig, M)

plt.plot(fitered_sig[1:250])
plt.plot(sig[1:250])

plt.show()
