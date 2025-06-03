# -*- coding: utf-8 -*-
"""
Created on Thu May 29 21:08:45 2025

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

#%% Filtro con cubic spline

qrs_detections = np.squeeze(mat_struct['qrs_detections']).astype(int)

# Punto de interés: 90 muestras antes del QRS
Point_of_Interest = qrs_detections - 90

# Filtramos solo los puntos que estén dentro del rango válido
Point_of_Interest = Point_of_Interest[(Point_of_Interest >= 0) & (Point_of_Interest + 20 < len(ecg_one_lead))]

# Calculamos el promedio en una ventana de 20 muestras para cada punto
t_prom = []
val_prom = []

for idx in Point_of_Interest:
    window = ecg_one_lead[idx : idx + 20] # De mi punto de interes voy hasrta 20 muestras mas adelante
    prom = np.mean(window)
    t_prom.append(idx / fs_ecg) # Paso a indices lo encontrado
    val_prom.append(prom)

# Calculo los indices antes y con eso armo los vectores para pasar a spline
# AL pasar estos vectores con los indices ya tenemos los puntos en los cuales vamos 
# haces los splines

t_prom = np.array(t_prom)       # Genero los array con los indices
val_prom = np.array(val_prom)

#%% Spline cúbico
cs = CubicSpline(t_prom, val_prom)

# Evaluamos el spline sobre toda la señal ( Polinomio interpolado )
spline_total = cs(t_ecg)

# Señal filtrada (original menos la línea de base)
ecg_filtrada = ecg_one_lead - spline_total

#%% Gráfico 1: Señal original + spline (línea de base)

plt.figure(figsize=(15, 5))
plt.plot(t_ecg, ecg_one_lead, label='ECG original', linewidth=1)
plt.plot(t_ecg, spline_total, label='Línea de base (Spline)', linewidth=2)
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud (normalizada)')
plt.title('ECG completo vs Línea de base estimada (spline)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Gráfico 2: Señal filtrada (original - línea de base)

plt.figure(figsize=(15, 5))
plt.plot(t_ecg, ecg_one_lead, label='ECG original', linewidth=1)
plt.plot(t_ecg, ecg_filtrada, label='ECG filtrado (original - línea de base)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud (filtrada)')
plt.title('ECG Filtrado por resta de la línea de base')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# VER PROS Y CONTRAS DE ESTE TIPO DE FILTRADO 

# PRO: Mantiene la morfología de la señal mejor que los lineales