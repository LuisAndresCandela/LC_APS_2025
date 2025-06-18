# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 18:52:28 2025

@author: candell1
"""

"""

Hacer STFT y CWT de las señales dela TS5

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
import pandas as pd

def vertical_flaten(a):

    return a.reshape(a.shape[0],1)

####################
# Lectura de audio #
####################

# Cargar el archivo CSV como un array de NumPy

fs_audio, wav_data = sio.wavfile.read('la cucaracha.wav')

# FSAUDIO = 48.000 - Ir de 0 a 0.25

#wav_data = wav_data[0:8000]

# Calculo largo y defino solapamiento de Welch

N = len(wav_data)

nperseg = N // 6 


from scipy.signal import stft, welch

t = np.arange(len(wav_data)) / fs_audio

f_welch, Pxx = welch(wav_data, fs=fs_audio, nperseg=256)
    
    # STFT
f, t_stft, Zxx = stft(wav_data, fs=fs_audio, nperseg=2000)
    
    # Crear figura y ejes
fig = plt.figure(figsize=(12, 6))
ax1 = plt.subplot(3, 1, 1)
ax2 = plt.subplot(3, 1, 2)
ax3 = plt.subplot(3, 1, 3)
    
    # Señal
ax1.plot(t, wav_data)
ax1.set_title("Señal audio")
ax1.set_ylabel("Amplitud")
ax1.set_xlim(t[0], t[-1])
    
    # Subplot 2: Welch
ax2.semilogy(f_welch, Pxx)
ax2.set_title("Estimación espectral por Welch")
ax2.set_ylabel("PSD [V²/Hz]")
ax2.set_xlabel("Frecuencia [Hz]")
    
    # Espectrograma
# Espectrograma
pcm = ax3.pcolormesh(t_stft, f, np.abs(Zxx), shading='auto')
ax3.set_title("STFT (Espectrograma)")
ax3.set_ylabel("Frecuencia [Hz]")
ax3.set_xlabel("Tiempo [s]")
ax3.set_ylim(0, 4000)  # Limita el eje y a 0-4000 Hz

    # Colorbar en eje externo
cbar_ax = fig.add_axes([0.92, 0.11, 0.015, 0.35])  # [left, bottom, width, height]
fig.colorbar(pcm, cax=cbar_ax, label="Magnitud")
    
plt.tight_layout(rect=[0, 0, 0.9, 1])  # Dejar espacio para colorbar a la derecha
plt.show()

#% AHora CWT

import pywt

# Escalas y CWT
scales = np.logspace(0, np.log10(150), num=100)  # 1 a 100 en logscale, pero igual serán convertidas a Hz

# wavelet = pywt.ContinuousWavelet('cmor1.5-1.0')
# wavelet = pywt.ContinuousWavelet('mexh')
wavelet = pywt.ContinuousWavelet('gaus3')

f_c = pywt.central_frequency(wavelet)  # devuelve frecuencia normalizada
Δt = 1.0 / fs_audio
frequencies = f_c / (scales * Δt)

coefficients, frec = pywt.cwt(wav_data, scales, wavelet, sampling_period=Δt)


# Crear figura y ejes
fig = plt.figure(figsize=(12, 8))
ax1 = plt.subplot(3, 1, 1)
ax2 = plt.subplot(3, 1, 2)
ax3 = plt.subplot(3, 1, 3)

# Señal
ax1.plot(t, wav_data)
ax1.set_title("Señal")
ax1.set_ylabel("Amplitud")
ax1.set_xlim(t[0], t[-1])

# CWT
pcm = ax2.imshow(np.abs(coefficients),
                 extent=[t[0], t[-1], scales[-1], scales[0]],  # nota el orden invertido para eje Y
                 cmap='viridis', aspect='auto')
ax2.set_title("CWT con wavelet basada en $B_3(x)$")
ax2.set_xlabel("Tiempo")
ax2.set_ylabel("Escala")

# Agrega colorbar para CWT
cbar_ax = fig.add_axes([0.92, 0.43, 0.015, 0.35])  # Ajustado para nueva posición
fig.colorbar(pcm, cax=cbar_ax, label="Magnitud")

# Welch
ax3.semilogy(f_welch, Pxx)
ax3.set_title("Estimación espectral por Welch")
ax3.set_ylabel("PSD [V²/Hz]")
ax3.set_xlabel("Frecuencia [Hz]")

# Ajustar diseño general dejando espacio para colorbar a la derecha
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.show()

import scipy.io as sio


wav_data = wav_data[1000:8000]

# Cargar señal
fs = fs_audio # Hz (NNormalizamos a fs/2 = f_nyq)
nyq_frec = fs / 2


cant_muestras = len(wav_data)


t = np.arange(len(wav_data)) / fs

# DWT multiescala hasta nivel 4
wavelet = 'db4'
max_level = pywt.dwt_max_level(len(wav_data), pywt.Wavelet(wavelet).dec_len)

coeffs = pywt.wavedec(wav_data, wavelet, level=4)

# Separar coeficientes
cA4, cD4, cD3, cD2, cD1 = coeffs

# DWT multiescala hasta nivel 4
wavelet = 'db4'
max_level = pywt.dwt_max_level(len(wav_data), pywt.Wavelet(wavelet).dec_len)

coeffs = pywt.wavedec(wav_data, wavelet, level=4)

# Separar coeficientes
cA4, cD4, cD3, cD2, cD1 = coeffs

plt.figure(figsize=(12, 8))

plt.subplot(6, 1, 1)
plt.plot(t, wav_data)
plt.title("ECG original")
plt.grid()

plt.subplot(6, 1, 2)
plt.plot(cD1)
plt.title("Coeficiente de detalle (nivel 1)")

plt.subplot(6, 1, 3)
plt.plot(cD2)
plt.title("Coeficiente de detalle (nivel 2)")

plt.subplot(6, 1, 4)
plt.plot(cD3)
plt.title("Coeficiente de detalle (nivel 3)")

plt.subplot(6, 1, 5)
plt.plot(cD4)
plt.title("Coeficiente de detalle (nivel 4)")

plt.subplot(6, 1, 6)
plt.plot(cA4)
plt.title("Coeficiente de aproximación (nivel 4)")

plt.tight_layout()
plt.show()



#%%

mat_struct = sio.loadmat('ecg.mat')
fs=1000

ecg_signal = mat_struct['ecg_lead'].ravel()
cant_muestras = len(ecg_signal)


t = np.arange(cant_muestras) / fs

f_welch, Pxx = welch(ecg_signal, fs=fs, nperseg=100)
    
    # STFT
f, t_stft, Zxx = stft(ecg_signal, fs=fs, nperseg=100)
    
    # Crear figura y ejes
fig = plt.figure(figsize=(12, 6))
ax1 = plt.subplot(3, 1, 1)
ax2 = plt.subplot(3, 1, 2)
ax3 = plt.subplot(3, 1, 3)
    
    # Señal
ax1.plot(t, ecg_signal)
ax1.set_title("Señal ecg")
ax1.set_ylabel("Amplitud")
ax1.set_xlim(t[0], t[-1])
    
    # Subplot 2: Welch
ax2.semilogy(f_welch, Pxx)
ax2.set_title("Estimación espectral por Welch")
ax2.set_ylabel("PSD [V²/Hz]")
ax2.set_xlabel("Frecuencia [Hz]")
    
    # Espectrograma
# Espectrograma
pcm = ax3.pcolormesh(t_stft, f, np.abs(Zxx), shading='auto')
ax3.set_title("STFT (Espectrograma)")
ax3.set_ylabel("Frecuencia [Hz]")
ax3.set_xlabel("Tiempo [s]")
ax3.set_ylim(0, 15)  # Limita el eje y a 0-4000 Hz

    # Colorbar en eje externo
cbar_ax = fig.add_axes([0.92, 0.11, 0.015, 0.35])  # [left, bottom, width, height]
fig.colorbar(pcm, cax=cbar_ax, label="Magnitud")
    
plt.tight_layout(rect=[0, 0, 0.9, 1])  # Dejar espacio para colorbar a la derecha
plt.show()

#% AHora CWT

import pywt

# Escalas y CWT
scales = np.logspace(0, np.log10(150), num=100)  # 1 a 100 en logscale, pero igual serán convertidas a Hz

# wavelet = pywt.ContinuousWavelet('cmor1.5-1.0')
# wavelet = pywt.ContinuousWavelet('mexh')
wavelet = pywt.ContinuousWavelet('gaus3')

f_c = pywt.central_frequency(wavelet)  # devuelve frecuencia normalizada
Δt = 1.0 / fs
frequencies = f_c / (scales * Δt)

coefficients, frec = pywt.cwt(ecg_signal, scales, wavelet, sampling_period=Δt)

# Crear figura y ejes
fig = plt.figure(figsize=(12, 8))
ax1 = plt.subplot(3, 1, 1)
ax2 = plt.subplot(3, 1, 2)
ax3 = plt.subplot(3, 1, 3)

# Señal
ax1.plot(t, ecg_signal)
ax1.set_title("Señal")
ax1.set_ylabel("Amplitud")
ax1.set_xlim(t[0], t[-1])

# CWT
pcm = ax2.imshow(np.abs(coefficients),
                  extent=[t[0], t[-1], scales[-1], scales[0]],  # nota el orden invertido para eje Y
                  cmap='viridis', aspect='auto')
ax2.set_title("CWT con wavelet basada en $B_3(x)$")
ax2.set_xlabel("Tiempo")
ax2.set_ylabel("Escala")

# Agrega colorbar para CWT
cbar_ax = fig.add_axes([0.92, 0.43, 0.015, 0.35])  # Ajustado para nueva posición
fig.colorbar(pcm, cax=cbar_ax, label="Magnitud")

# Welch
ax3.semilogy(f_welch, Pxx)
ax3.set_title("Estimación espectral por Welch")
ax3.set_ylabel("PSD [V²/Hz]")
ax3.set_xlabel("Frecuencia [Hz]")

# Ajustar diseño general dejando espacio para colorbar a la derecha
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.show()

#%%

import scipy.io as sio

regs_interes = (
                np.array([4000, 5500]), # muestras
                np.array([10e3, 11e3]), # muestras
                np.array([5, 5.2]) *60*fs, # minutos a muestras
                np.array([12, 12.4]) *60*fs, # minutos a muestras
                np.array([15, 15.2]) *60*fs # minutos a muestras        
                )

roi = regs_interes[1].astype(int)

# Cargar señal
fs = 1000 # Hz (NNormalizamos a fs/2 = f_nyq)
nyq_frec = fs / 2

mat_struct = sio.loadmat('ecg.mat')

ecg_signal = mat_struct['ecg_lead']
ecg_signal = ecg_signal.flatten().astype(np.float64)
ecg_signal = ecg_signal[roi[0]:roi[1]]
cant_muestras = len(ecg_signal)


t = np.arange(len(ecg_signal)) / fs

# DWT multiescala hasta nivel 4
wavelet = 'db4'
max_level = pywt.dwt_max_level(len(ecg_signal), pywt.Wavelet(wavelet).dec_len)

coeffs = pywt.wavedec(ecg_signal, wavelet, level=4)

# Separar coeficientes
cA4, cD4, cD3, cD2, cD1 = coeffs

# DWT multiescala hasta nivel 4
wavelet = 'db4'
max_level = pywt.dwt_max_level(len(ecg_signal), pywt.Wavelet(wavelet).dec_len)

coeffs = pywt.wavedec(ecg_signal, wavelet, level=4)

# Separar coeficientes
cA4, cD4, cD3, cD2, cD1 = coeffs

plt.figure(figsize=(12, 8))

plt.subplot(6, 1, 1)
plt.plot(t, ecg_signal)
plt.title("ECG original")
plt.grid()

plt.subplot(6, 1, 2)
plt.plot(cD1)
plt.title("Coeficiente de detalle (nivel 1)")

plt.subplot(6, 1, 3)
plt.plot(cD2)
plt.title("Coeficiente de detalle (nivel 2)")

plt.subplot(6, 1, 4)
plt.plot(cD3)
plt.title("Coeficiente de detalle (nivel 3)")

plt.subplot(6, 1, 5)
plt.plot(cD4)
plt.title("Coeficiente de detalle (nivel 4)")

plt.subplot(6, 1, 6)
plt.plot(cA4)
plt.title("Coeficiente de aproximación (nivel 4)")

plt.tight_layout()
plt.show()