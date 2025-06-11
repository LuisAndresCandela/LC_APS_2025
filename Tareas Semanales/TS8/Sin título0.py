# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 18:28:51 2025

@author: candell1
"""

#%% Importado de módulos
import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import medfilt
from scipy.interpolate import CubicSpline

#%% Función auxiliar
def vertical_flaten(a):
    return a.reshape(a.shape[0],1)

#%% Carga y preprocesamiento del ECG
fs_ecg = 1000  # Hz

# Carga de la señal ECG
mat_struct = sio.loadmat('./ecg.mat')  # Usamos el mismo archivo para ambos métodos
ecg_one_lead = np.squeeze(mat_struct['ecg_lead']).astype(float)

# Normalización
ecg_one_lead = (ecg_one_lead - np.mean(ecg_one_lead)) / np.std(ecg_one_lead)

# Tiempo total
t_ecg = np.arange(len(ecg_one_lead)) / fs_ecg

#%% --- Filtro de mediana (Baseline por mediana) ---
win1_samples = 200
win2_samples = 600

# Asegurar que sean impares
if win1_samples % 2 == 0:
    win1_samples += 1
if win2_samples % 2 == 0:
    win2_samples += 1

# Aplicación de los filtros
ecg_med1 = medfilt(ecg_one_lead, kernel_size=win1_samples)
baseline_median = medfilt(ecg_med1, kernel_size=win2_samples)

#%% --- Filtro por splines cúbicos (Baseline por spline) ---
qrs_detections = np.squeeze(mat_struct['qrs_detections']).astype(int)

# Punto de interés: 90 muestras antes del QRS
Point_of_Interest = qrs_detections - 90
Point_of_Interest = Point_of_Interest[(Point_of_Interest >= 0) & (Point_of_Interest + 20 < len(ecg_one_lead))]

# Cálculo del promedio en una ventana de 20 muestras
t_prom = []
val_prom = []

for idx in Point_of_Interest:
    window = ecg_one_lead[idx : idx + 20]
    prom = np.mean(window)
    t_prom.append(idx / fs_ecg)
    val_prom.append(prom)

t_prom = np.array(t_prom)
val_prom = np.array(val_prom)

# Spline cúbico
cs = CubicSpline(t_prom, val_prom)
baseline_spline = cs(t_ecg)

#%% --- Gráficos ---
plt.figure(figsize=(15, 8))

# Subplot 1 - Mediana
plt.subplot(2, 1, 1)
plt.plot(t_ecg, ecg_one_lead, label='ECG original', alpha=0.6)
plt.plot(t_ecg, baseline_median, label='Línea de base (Mediana)', linewidth=2)
plt.title('ECG con línea de base estimada - Filtrado de Mediana')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)

# Subplot 2 - Splines
plt.subplot(2, 1, 2)
plt.plot(t_ecg, ecg_one_lead, label='ECG original', alpha=0.6)
plt.plot(t_ecg, baseline_spline, label='Línea de base (Spline)', linewidth=2)
plt.title('ECG con línea de base estimada - Spline cúbico')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
