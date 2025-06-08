# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 20:24:50 2025

@author: candell1
"""


"""
Interpolacion y Diezmado

Diezmado con ECG e Interpolacion con Audio


"""
#%%
"""
Diezmado del ECG reduciendo el ancho de banda a 250 hz 

"""


#%% Importado de módulos

import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.interpolate import CubicSpline

def vertical_flaten(a):
    return a.reshape(a.shape[0], 1)

#%% Lectura del ECG

fs_ecg = 1000  # Hz

mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = np.squeeze(mat_struct['ecg_lead']).astype(float)

# Normalización
ecg_one_lead = (ecg_one_lead - np.mean(ecg_one_lead)) / np.std(ecg_one_lead)

# Tiempo total
t_ecg = np.arange(len(ecg_one_lead)) / fs_ecg

ecg_segment = ecg_one_lead
t_segment = np.linspace(0, len(ecg_segment) / fs_ecg, len(ecg_segment))

#%%

"""
Diseño del filtro FIR con firwin2

Quiero bajas la FS del ECG a 250 hz, por lo que vamos a hacer un diezmado

Hay que normalizar por nyquist

Primero quiero generar un LP cuya wc sea 1/M 

Y luego de eso, sacarle las muestras que correspondan

"""

#%% Plantilla del filtro

from scipy.signal import firwin2, freqz
import numpy as np
import matplotlib.pyplot as plt

fs = 1000  # Hz
M = 20  # factor de diezmado (ejemplo)
cant_coef = 501  # debe ser impar preferentemente para un retardo entero

# Frecuencia de corte deseada: hasta fs/(2M)
f_pass = (1 / M) * 0.9  # zona de paso (evitar dejarlo justo en fs/2M)
f_stop = (1 / M) * 1.1  # inicio zona de atenuación

# Normalizamos respecto a Nyquist
nyq = fs / 2
frecs = [0, f_pass, f_stop, nyq]
frecs_norm = [f / nyq for f in frecs]

# Ganancia en magnitud (lineal, no en dB)
gains = [1, 1, 0, 0]

# Filtro FIR
fir_lp = firwin2(cant_coef, frecs_norm, gains, window='hamming')


ecg_filtrada = sig.filtfilt(fir_lp, [1.0], ecg_segment)

ecg_filtrada = ecg_filtrada[::M]

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








