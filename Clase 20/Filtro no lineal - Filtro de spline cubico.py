# -*- coding: utf-8 -*-
"""
Created on Thu May 29 20:55:04 2025

@author: candell1
"""


#%% Importado de modulos

import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio

def vertical_flaten(a):

    return a.reshape(a.shape[0],1)

#%% Lectura del ECG

fs_ecg = 1000  # Hz

mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = np.squeeze(mat_struct['ecg_lead']).astype(float)

# Normalización
ecg_one_lead = (ecg_one_lead - np.mean(ecg_one_lead)) / np.std(ecg_one_lead)

#%% Segmento de interés
ecg_segment = ecg_one_lead[700000:745000]
t_segment = np.linspace(0, len(ecg_segment) / fs_ecg, len(ecg_segment))

# 700.000 - 745.000

#%% Filtro no lineal - Filtro de interpolacion con splines cubicos

from scipy.interpolate import CubicSpline

# Parámetros
step = 300

# Submuestreo de puntos
t_control = t_segment[::step]
ecg_control = ecg_segment[::step]

# Ajuste del spline cúbico
spline = CubicSpline(t_control, ecg_control)

# Estimación de la línea de base
baseline = spline(t_segment)

# Corrección de la señal
ecg_corrected = ecg_segment - baseline

# Gráfico en subplots
fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Señal original + línea base
axs[0].plot(t_segment, ecg_segment, label='ECG original', alpha=0.5)
axs[0].plot(t_segment, baseline, label='Línea base (Spline)', linewidth=2)
axs[0].set_ylabel('Amplitud')
axs[0].set_title('Señal original y línea de base')
axs[0].legend()
axs[0].grid(True)

# Señal corregida
axs[1].plot(t_segment, ecg_corrected, label='ECG corregido', color='green')
axs[1].set_xlabel('Tiempo [s]')
axs[1].set_ylabel('Amplitud')
axs[1].set_title('ECG corregido (sin línea de base)')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()



