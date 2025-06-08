# -*- coding: utf-8 -*-
"""
Created on Wed May 28 20:49:40 2025

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

#%% Lectura del ECG

fs_ecg = 1000  # Hz

mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = np.squeeze(mat_struct['ecg_lead']).astype(float)

# Normalización
ecg_one_lead = (ecg_one_lead - np.mean(ecg_one_lead)) / np.std(ecg_one_lead)

#%% Segmento de interés
ecg_segment = ecg_one_lead
t_segment = np.linspace(0, len(ecg_segment) / fs_ecg, len(ecg_segment))

#%% Diseño del filtro FIR con firwin2

# Parametros de la plantilla del filtro 

fs = 1000 # Hz
fpass = np.array([0.5, 30])      # Banda de paso
fstop = np.array([0.2, 50])      # Banda de detención
ripple = 0.5                     # Rizado en banda de paso (dB)
attenuation = 40                # Atenuación en banda de detención (dB)

#%% Diseño de filtro con firwin2

cant_coef = 2500    # cantidad de coeficientes (orden + 1), ideal impar

nyq = fs / 2        # frecuencia de Nyquist

# Definimos los puntos de frecuencia y ganancia
# En Hz: queremos un filtro pasabanda entre 1 y 35 Hz

freq_hz = [0.0, 0.2, 0.5, 30.0, 50.0, nyq]   # en Hz
gain =    [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]     # ganancia deseada en cada punto

# Normalizar las frecuencias
freq_norm = [f / nyq for f in freq_hz]

# Diseñar el filtro con firwin2
mi_sos = sig.firwin2(
    numtaps=        cant_coef,
    freq=           freq_norm,
    gain=           gain,
    window=         ('kaiser', 10) )

#%% 

#%% Aplicación del filtro
#ecg_filtrada  = sig.lfilter(mi_sos, [1.0], ecg_segment)

ecg_filtrada = sig.filtfilt(mi_sos, [1.0], ecg_segment)


#%% Gráfico: señal original vs filtrada

# Crear 3 subplots (el tercero muestra ambas señales solapadas en toda la duración)
fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=False)

# Señal original
axs[0].plot(t_segment, ecg_segment, color='blue')
axs[0].set_title("ECG Original")
axs[0].set_ylabel("Amplitud")
axs[0].grid(True)

# Señal filtrada
axs[1].plot(t_segment, ecg_filtrada, color='green')
axs[1].set_title("ECG Filtrada con iirdesign + sosfiltfilt")
axs[1].set_ylabel("Amplitud")
axs[1].grid(True)

# Ambas señales superpuestas en toda la frecuencia
axs[2].plot(t_segment, ecg_segment, label="Original", color='blue', alpha=0.7)
axs[2].plot(t_segment, ecg_filtrada, label="Filtrada", color='green', alpha=0.7)
axs[2].set_title("ECG Original vs Filtrada (Toda la señal)")
axs[2].set_xlabel("Tiempo [s]")
axs[2].set_ylabel("Amplitud")
axs[2].grid(True)
axs[2].legend()

plt.tight_layout()
plt.show()