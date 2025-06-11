# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 19:44:39 2025

@author: candell1
"""

"""
Deteccion de patrones - Picos ECG

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

#%% AHora quiero ver la señal del qrs_pattern1

qrs_pattern = np.squeeze(mat_struct['qrs_pattern1']).astype(int)
    
# Normalizamos el patrón QRS
qrs_pattern_norm = (qrs_pattern - np.mean(qrs_pattern)) / np.std(qrs_pattern)

# Normalizamos la señal ECG filtrada
ecg_filtrada_norm = (ecg_filtrada - np.mean(ecg_filtrada)) / np.std(ecg_filtrada)

corr_norm = sig.correlate(ecg_filtrada_norm, qrs_pattern_norm, mode='same')

plt.figure(figsize=(12, 4))
plt.plot(t_ecg, corr_norm, label='Correlación normalizada', color='blue')
plt.title('Correlación normalizada entre ECG filtrado y patrón QRS')
plt.xlabel('Tiempo [s]')
plt.ylabel('Correlación (normalizada)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

from scipy.signal import find_peaks

# Paso 1: Normalización
ecg_filtrada_norm = (ecg_filtrada - np.mean(ecg_filtrada)) / np.std(ecg_filtrada)
qrs_pattern_norm = (qrs_pattern - np.mean(qrs_pattern)) / np.std(qrs_pattern)

# Paso 2: Correlación normalizada
corr_norm = sig.correlate(ecg_filtrada_norm, qrs_pattern_norm, mode='same')

# Paso 3: Reescalado
ecg_rescaled = ecg_filtrada_norm / np.max(np.abs(ecg_filtrada_norm))
corr_rescaled = corr_norm / np.max(np.abs(corr_norm))

# Paso 4: Detección de picos en la correlación reescalada
threshold = 0.15  # Umbral relativo al máximo
peaks, _ = find_peaks(corr_rescaled, height=threshold, distance=200)

# El umbral habia quedado muy alto en 0.5 y no detectabamos todo. Se bajo a 0.15
# Para ello nos pusimos en el pico mas bajo que observamos luego de inspeccionar la señal 
# Parandonos en esa situacion vimos que el pico mas bajo estbaa en 0.15

# Paso 5: Gráfico conjunto
plt.figure(figsize=(14, 5))
plt.plot(t_ecg, ecg_rescaled, label='ECG filtrado (normalizado)', alpha=0.7)
plt.plot(t_ecg, corr_rescaled, label='Correlación normalizada (reescalada)', alpha=0.7)
plt.plot(t_ecg[peaks], corr_rescaled[peaks], 'rx', label='Picos de correlación')
plt.title('ECG filtrado vs Correlación (con detección de picos)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud (escalada)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

"""
Guardar en un array todas las detecciones y compararlas con loque ya tenemos en el archivo

Hay que armar una tabla de " confusion "
       
                        q        No q               --- Detecciones ( Peak)
                Q
Detecciones
verdaderas
(Archivo)
                No Q
                

El objetivo es comparar la concondancia entre las 2 detecciones viendo 
si los eventos coiniciden dentro de ciertos margenes ( 100 ms aprox o 100 muestras )
Entonces, contamos cuandos de ellos coincidieron y cuantos no 

"""





