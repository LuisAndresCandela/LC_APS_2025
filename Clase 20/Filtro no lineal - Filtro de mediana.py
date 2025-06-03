# -*- coding: utf-8 -*-
"""
Created on Thu May 29 20:26:08 2025

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
ecg_segment = ecg_one_lead
t_segment = np.linspace(0, len(ecg_segment) / fs_ecg, len(ecg_segment))

# 700.000 - 745.000

#%% Filtro no Lineal - Filtro de mediana

from scipy.signal import medfilt

# Ventanas
win1_samples = 200
win2_samples = 1200

# Aseguro que sea impar, si es par le sumo 1
if win1_samples % 2 == 0:
    win1_samples += 1
if win2_samples % 2 == 0:
    win2_samples += 1

# Primer filtro de mediana (200 ms)
ecg_med1 = medfilt(ecg_segment, kernel_size=win1_samples)

# Segundo filtro de mediana (600 ms)
ecg_med2 = medfilt(ecg_med1, kernel_size=win2_samples)

# Visualización
plt.figure(figsize=(12, 5))
plt.plot(t_segment, ecg_segment, label='Señal original', alpha=0.6)
plt.plot(t_segment, ecg_med2, label='Linea de base - Filtrado (200 ms + 600 ms)', linewidth=2)
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('ECG filtrado con dos etapas de mediana')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Falta calcular la señal final restando ECG_med 2 a la original

# Ojo que esto agrega saltos y morfologias a la señal debido a las limitaciones del metodo
# 


