# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 21:04:17 2025

@author: candell1

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

#%% Filtro la señal del ECG con spline cubicos

qrs_detections = np.squeeze(mat_struct['qrs_detections']).astype(int)

# Parámetros de ventana
N = 201
half_N = N // 2

ventanas = []

# Recorremos cada QRS detection
for idx in qrs_detections:
    if idx - half_N >= 0 and idx + half_N < len(ecg_one_lead):
        ventana = ecg_one_lead[idx - half_N : idx + half_N + 1]
        # Normalizamos cada ventana individualmente
        ventana_normalizada = (ventana - np.mean(ventana)) / np.std(ventana)
        ventanas.append(ventana_normalizada)

# Convertimos a matriz de R x 201
matriz_ventanas = np.array(ventanas)

# Tiempo relativo: de -0.1 a 0.1 segundos
t_ventana = np.linspace(-half_N, half_N, N) / fs_ecg

# Graficar todas las realizaciones normalizadas
plt.figure(figsize=(12, 6))
for i in range(matriz_ventanas.shape[0]):
    plt.plot(t_ventana, matriz_ventanas[i], color='steelblue', alpha=0.3)

plt.title('Todas las realizaciones del latido QRS (normalizadas)')
plt.xlabel('Tiempo relativo al QRS [s]')
plt.ylabel('Amplitud normalizada')
plt.grid(True)
plt.tight_layout()
plt.show()

# Promedio
latido_promedio = np.mean(matriz_ventanas, axis=0)

# Graficar promedio sobre todas las realizaciones
plt.figure(figsize=(12, 6))
for i in range(matriz_ventanas.shape[0]):
    plt.plot(t_ventana, matriz_ventanas[i], color='steelblue', alpha=0.2)

plt.plot(t_ventana, latido_promedio, color='red', linewidth=2, label='Latido promedio')

plt.title('Realizaciones del latido QRS normalizadas y su promedio')
plt.xlabel('Tiempo relativo al QRS [s]')
plt.ylabel('Amplitud normalizada')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
