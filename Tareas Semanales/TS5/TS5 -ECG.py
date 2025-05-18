# -*- coding: utf-8 -*-
"""
Created on Sat May 17 18:39:52 2025

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

#%% Genero mi funcion para implementar Blackmann-Tuckey

from scipy.signal import fftconvolve

def blackman_tukey_psd(x, window_len=None, window_func=blackman, fs=1.0):
    N = len(x)
    if window_len is None:
        window_len = N // 4

    # Autocorrelación estimada (eficiente)
    rxx = fftconvolve(x, x[::-1], mode='full') / N
    mid = len(rxx) // 2
    rxx = rxx[mid:mid + window_len]

    ventana = window_func(window_len)
    rxx_win = rxx * ventana

    Pxx = np.abs(fft(rxx_win, n=2 * window_len))
    f_Pxx = np.fft.fftfreq(2 * window_len, d=1/fs)
    mask = f_Pxx >= 0
    return f_Pxx[mask], Pxx[mask]

#%% Lectura del ECG

fs_ecg = 1000 # Hz

# para listar las variables que hay en el archivo
#io.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_one_lead = np.squeeze(mat_struct['ecg_lead']).astype(float)
ecg_one_lead = (ecg_one_lead - np.mean(ecg_one_lead)) / np.std(ecg_one_lead)

N = len(ecg_one_lead)

hb_1 = vertical_flaten(mat_struct['heartbeat_pattern1'])
hb_2 = vertical_flaten(mat_struct['heartbeat_pattern2'])


#%% Normalizar la señal
ecg_one_lead = (ecg_one_lead - np.mean(ecg_one_lead)) / np.std(ecg_one_lead)

ecg_segment = ecg_one_lead[5000:12000] # Me quedo solo con un segmento de la señal del ECG

#%% Calculo PSD con Blackman-Tukey
Fs_ECG_BT, Signal_ECG_Pxx = blackman_tukey_psd(x=ecg_segment,
                                               window_len=None,
                                               window_func=blackman,
                                               fs=fs_ecg)

#%% Calculo energía acumulada y anchos de banda
psd_norm = Signal_ECG_Pxx / np.sum(Signal_ECG_Pxx)
Acumulada = np.cumsum(psd_norm)

bw_95 = Fs_ECG_BT[np.where(Acumulada >= 0.95)[0][0]]
bw_98 = Fs_ECG_BT[np.where(Acumulada >= 0.98)[0][0]]

print(f"Ancho de banda para 95%: {bw_95:.2f} Hz")
print(f"Ancho de banda para 98%: {bw_98:.2f} Hz")

#%% Gráficos
fig, axs = plt.subplots(3, 1, figsize=(12, 10))

# Tiempo para el segmento
t_segment = np.linspace(5000/fs_ecg, 12000/fs_ecg, len(ecg_segment))

# Señal original (solo segmento)
axs[0].plot(t_segment, ecg_segment)
axs[0].set_title("Señal ECG (Segmento 5000 a 12000)")
axs[0].set_xlabel("Tiempo [s]")
axs[0].set_ylabel("Amplitud")
axs[0].grid(True)

# PSD del segmento
axs[1].plot(Fs_ECG_BT, Signal_ECG_Pxx)
axs[1].set_title("Densidad Espectral de Potencia (PSD) - Blackman-Tukey (Segmento)")
axs[1].set_xlabel("Frecuencia [Hz]")
axs[1].set_ylabel("Potencia")
axs[1].grid(True)

# PSD con anchos de banda del segmento
axs[2].plot(Fs_ECG_BT, Signal_ECG_Pxx)
axs[2].axvline(bw_95, color='r', linestyle='--', label=f'BW 95%: {bw_95:.2f} Hz')
axs[2].axvline(bw_98, color='g', linestyle='--', label=f'BW 98%: {bw_98:.2f} Hz')
axs[2].set_title("PSD con Ancho de Banda 95% y 98% (Segmento)")
axs[2].set_xlabel("Frecuencia [Hz]")
axs[2].set_ylabel("Potencia")
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()
