# -*- coding: utf-8 -*-
"""
Created on Sat May 17 19:16:02 2025

@author: candell1
"""

#%% Importado de modulos

import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.io.wavfile import write
from scipy import signal
from scipy.signal import welch
from scipy.signal.windows import hamming, hann, blackman, kaiser, flattop, blackmanharris
from scipy.fft import fft, fftshift
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import os
from scipy.fft import fft
from scipy.signal import fftconvolve

def vertical_flaten(a):

    return a.reshape(a.shape[0],1)

#%%

# Parámetros
fs_ppg = 400  # Hz

# Cargar señal PPG
ppg = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)

# Normalizar señal completa
ppg = (ppg - np.mean(ppg)) / np.std(ppg)

# Selección de un segmento
start_sample = 2000
end_sample = 6000
ppg_segment = ppg[start_sample:end_sample]

N= end_sample - start_sample

# Eliminar tendencia lenta con media móvil
def detrend_signal(x, window_size):
    kernel = np.ones(window_size) / window_size
    trend = np.convolve(x, kernel, mode='same')
    return x - trend

ppg_segment_detrended = detrend_signal(ppg_segment, window_size=200)

# Re-normalización
ppg_segment_detrended = (ppg_segment_detrended - np.mean(ppg_segment_detrended)) / np.std(ppg_segment_detrended)

nperseg =  N // 8

# Estimar PSD con Welch 
frecs, psd = welch(ppg_segment_detrended,
                   fs=fs_ppg,
                   window='hamming',
                   nperseg = nperseg,
                   noverlap= nperseg / 2 )

# Calcular energía acumulada y anchos de banda
psd_norm = psd / np.sum(psd)
acumulada = np.cumsum(psd_norm)

bw_95 = frecs[np.where(acumulada >= 0.95)[0][0]]
bw_98 = frecs[np.where(acumulada >= 0.98)[0][0]]

print(f"Ancho de banda para 95% de la energía: {bw_95:.2f} Hz")
print(f"Ancho de banda para 98% de la energía: {bw_98:.2f} Hz")

# Gráficos
fig, axs = plt.subplots(3, 1, figsize=(12, 10))

t_segment = np.arange(start_sample, end_sample) / fs_ppg

# Señal original y detrended
axs[0].plot(t_segment, ppg_segment, label='Original')
axs[0].plot(t_segment, ppg_segment_detrended, label='Detrended')
axs[0].set_title("Segmento de señal PPG (Original vs Detrended)")
axs[0].set_xlabel("Tiempo [s]")
axs[0].set_ylabel("Amplitud")
axs[0].legend()
axs[0].grid(True)

# PSD
axs[1].plot(frecs, psd)
axs[1].set_title("PSD (Welch - Periodograma Modificado)")
axs[1].set_xlabel("Frecuencia [Hz]")
axs[1].set_ylabel("Densidad espectral de potencia")
axs[1].grid(True)

# PSD con anchos de banda
axs[2].plot(frecs, psd)
axs[2].axvline(bw_95, color='r', linestyle='--', label=f'BW 95%: {bw_95:.2f} Hz')
axs[2].axvline(bw_98, color='g', linestyle='--', label=f'BW 98%: {bw_98:.2f} Hz')
axs[2].set_title("PSD con Anchos de Banda 95% y 98%")
axs[2].set_xlabel("Frecuencia [Hz]")
axs[2].set_ylabel("Potencia")
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()
