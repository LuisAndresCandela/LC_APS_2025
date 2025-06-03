# -*- coding: utf-8 -*-
"""
Created on Thu May 22 19:08:41 2025

@author: candell1
"""

#%% 

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

ecg_segment = ecg_one_lead #ecg_one_lead[5000:24000] # Me quedo solo con un segmento de la señal del ECG

t_segment = np.linspace(0, len(ecg_segment) / fs_ecg, len(ecg_segment))

#%% Filtrado de ECG

# Tipos de filtro que vamos a usar 

aprox_name = 'butter'
# aprox_name = 'cheby1'
# aprox_name = 'cheby2'
# aprox_name = 'ellip'

# Parametros de la plantilla del filtro 

fs = 1000 # Hz
fpass = np.array( [1, 35.0] )       # Lo definimos segun la plantilla. de 1 hz a 35 hz la banda de paso 
ripple = 1                          # Db -- Alpha max
fstop = np.array( [0.1, 50.] )      # y la banda de stop comienza en 50 hz
attenuation = 40                    # dB -- Alplha min

#%% Diseño de filtro con iirdesing

mi_sos = sig.iirdesign(
    wp=fpass,
    ws=fstop,
    gpass=ripple,
    gstop=attenuation,
    ftype=aprox_name,
    output='sos',
    fs=fs
)


#%% Aplicación del filtro a la señal ECG (segmento)
ecg_filtrada = sig.sosfiltfilt(mi_sos, ecg_segment)

from scipy.signal import sosfreqz, group_delay

# Calcular respuesta en frecuencia (frecuencia normalizada)
w, h = sosfreqz(mi_sos, worN=2048, fs=fs)

# Fase en radianes
fase = np.angle(h)

# Demora de grupo
# Convertimos el filtro SOS a formato (b, a) para usar `group_delay`
b, a = sig.sos2tf(mi_sos)
w_gd, gd = group_delay((b, a), fs=fs)

# Graficar
plt.figure(figsize=(14, 6))

# Fase
plt.subplot(2, 1, 1)
plt.plot(w, np.unwrap(fase), color='purple')
plt.title("Fase del Filtro")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Fase [radianes]")
plt.grid(True)

# Demora de grupo
plt.subplot(2, 1, 2)
plt.plot(w_gd, gd, color='darkred')
plt.title("Demora de Grupo del Filtro")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Demora [muestras]")
plt.grid(True)

plt.tight_layout()
plt.show()


#%%

demora = 0
fig_dpi = 50
fig_sz_x = 300
fig_sz_y = 120


regs_interes = ( 
        [4000, 5500], # muestras
        [10e3, 11e3], # muestras
        )

for ii in regs_interes:
    
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([N, ii[1]]), dtype='uint')
    
    plt.figure(figsize=(fig_sz_x, fig_sz_y), dpi= fig_dpi, facecolor='w', edgecolor='k')
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    #plt.plot(zoom_region, ECG_f_butt[zoom_region], label='Butter')
    plt.plot(zoom_region, ecg_filtrada[zoom_region + demora], label='Cheby')
    
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
    
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
            
    plt.show()
    
regs_interes = ( 
        np.array([5, 5.2]) *60*fs, # minutos a muestras
        np.array([12, 12.4]) *60*fs, # minutos a muestras
        np.array([15, 15.2]) *60*fs, # minutos a muestras
        )

for ii in regs_interes:
    
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([N, ii[1]]), dtype='uint')
    
    plt.figure(figsize=(fig_sz_x, fig_sz_y), dpi= fig_dpi, facecolor='w', edgecolor='k')
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    #plt.plot(zoom_region, ECG_f_butt[zoom_region], label='Butter')
    plt.plot(zoom_region, ecg_filtrada[zoom_region + demora], label='Cheby')
    
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
    
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
            
    plt.show()

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

