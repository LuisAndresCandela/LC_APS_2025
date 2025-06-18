# -*- coding: utf-8 -*-
"""
Created on Sat May 17 19:49:42 2025

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
import pandas as pd
#from tabulate import tabulate

def vertical_flaten(a):

    return a.reshape(a.shape[0],1)

#%%

####################
# Lectura de audio #
####################

# Cargar el archivo CSV como un array de NumPy

fs_audio, wav_data = sio.wavfile.read('la cucaracha.wav')
# fs_audio, wav_data = sio.wavfile.read('prueba psd.wav')
# fs_audio, wav_data = sio.wavfile.read('silbido.wav')

# Calculo largo y defino solapamiento de Welch

N = len(wav_data)
nperseg = N // 6 

# Calculo PDS con Welch

Fs_Audio_welch, Signal_Audio_Pxx = welch(x = wav_data , 
                                         fs = fs_audio, 
                                         window='hann', 
                                         nperseg = nperseg,
                                         noverlap = nperseg//2, 
                                         nfft=None, 
                                         detrend='linear', 
                                         return_onesided=True, 
                                         scaling='spectrum', 
                                         axis=0, 
                                         average='median')

# Convertir a decibeles
Signal_Audio_Pxx_mean_db = 10 * np.log10(Signal_Audio_Pxx)

Signal_Audio_Pxx_mean_db -= np.max(Signal_Audio_Pxx_mean_db )  

 
"""

Esto es normalizar el maximo a 0db. Esto tiene sentido porque tenemos distintos BW, 
para centrar todo en un mismo punto o que los picos esten en la misma escala

"""


#%% Calculo de ancho de banda

"""

Como tenemos una señal de audio, buscamos el pico maximo y a partir de alli nos movemos a los costados
de manera de ir avanzando hasta encontrar el 95 o 98 % del valor maximo de la señal. 

"""

# Paso 1: Potencia total
potencia_total = np.sum(Signal_Audio_Pxx)

# Paso 2: Índice del pico principal
peak_index = np.argmax(Signal_Audio_Pxx)

# Paso 3: Inicialización para acumulación centrada
acumulada = Signal_Audio_Pxx[peak_index]
i_low = peak_index
i_high = peak_index

# Umbrales deseados
umbral_95 = 0.95 * potencia_total
umbral_98 = 0.98 * potencia_total

# Guardamos BW cuando se cumplan los umbrales
bw_95_found = False
bw_98_found = False

# Paso 4: Expandir hacia los costados
while not bw_98_found and (i_low > 0 or i_high < len(Signal_Audio_Pxx) - 1):
    # Expandimos hacia el lado con mayor potencia siguiente
    if i_low > 0 and (i_high == len(Signal_Audio_Pxx) - 1 or Signal_Audio_Pxx[i_low - 1] >= Signal_Audio_Pxx[i_high + 1]):
        i_low -= 1
        acumulada += Signal_Audio_Pxx[i_low]
    elif i_high < len(Signal_Audio_Pxx) - 1:
        i_high += 1
        acumulada += Signal_Audio_Pxx[i_high]

    # Guardar anchos de banda cuando se alcanzan umbrales
    if not bw_95_found and acumulada >= umbral_95:
        bw_95_low = Fs_Audio_welch[i_low]
        bw_95_high = Fs_Audio_welch[i_high]
        bw_95_found = True

    if not bw_98_found and acumulada >= umbral_98:
        bw_98_low = Fs_Audio_welch[i_low]
        bw_98_high = Fs_Audio_welch[i_high]
        bw_98_found = True

# Ancho de banda centrado
bw_95 = bw_95_high - bw_95_low
bw_98 = bw_98_high - bw_98_low

print(f"BW centrado al 95% de la potencia: {bw_95:.2f} Hz (de {bw_95_low:.1f} Hz a {bw_95_high:.1f} Hz)")
print(f"BW centrado al 98% de la potencia: {bw_98:.2f} Hz (de {bw_98_low:.1f} Hz a {bw_98_high:.1f} Hz)")

#%% Subplots: Forma de onda + PSD dB + PSD Lineal con sombreado
#plt.style.use('seaborn-v0_8-darkgrid')
fig, axs = plt.subplots(3, 1, figsize=(14, 12))

# Anotar la frecuencia del pico principal
peak_idx = np.argmax(Signal_Audio_Pxx_mean_db)
peak_freq = Fs_Audio_welch[peak_idx]

# 1. Señal de audio
tiempo = np.arange(N) / fs_audio
axs[0].plot(tiempo, wav_data, color='tab:gray')
axs[0].set_title("Forma de Onda del Audio")
axs[0].set_xlabel("Tiempo [s]")
axs[0].set_ylabel("Amplitud")

# 2. PSD en dB normalizada
axs[1].plot(Fs_Audio_welch, Signal_Audio_Pxx_mean_db, color='tab:blue', lw=2)
axs[1].axvline(peak_freq, color='black', linestyle=':', label=f'Pico: {peak_freq:.1f} Hz')
axs[1].axvline(bw_95_low, color='green', linestyle='--', label=f'BW 95%')
axs[1].axvline(bw_95_high, color='green', linestyle='--')
axs[1].axvline(bw_98_low, color='red', linestyle='--', label=f'BW 98%')
axs[1].axvline(bw_98_high, color='red', linestyle='--')
axs[1].set_title("PSD (dB, Normalizada)")
axs[1].set_xlabel("Frecuencia [Hz]")
axs[1].set_ylabel("PSD [dB]")
axs[1].legend()

# 3. PSD lineal con banda del 98% sombreada
axs[2].plot(Fs_Audio_welch, Signal_Audio_Pxx, lw=2, color='tab:blue')
axs[2].fill_between(Fs_Audio_welch[i_low:i_high + 1], Signal_Audio_Pxx[i_low:i_high + 1],
                    color='orange', alpha=0.5, label='Potencia acumulada (98%)')
axs[2].axvline(peak_freq, color='black', linestyle=':', label='Pico principal')
axs[2].set_title("PSD Lineal con Zona de Potencia Acumulada (98%)")
axs[2].set_xlabel("Frecuencia [Hz]")
axs[2].set_ylabel("PSD [potencia/Hz]")
axs[2].legend()

plt.tight_layout()
plt.show()

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

ecg_segment = ecg_one_lead[5000:12000] # Me quedo solo con un segmento de la señal del ECG

#%% Calculo PSD con Blackman-Tukey
Fs_ECG_BT, Signal_ECG_Pxx = blackman_tukey_psd(x=ecg_segment,
                                               window_len=None,
                                               window_func=blackman,
                                               fs=fs_ecg)

#%% Calculo energía acumulada y anchos de banda
psd_norm = Signal_ECG_Pxx / np.sum(Signal_ECG_Pxx)
Acumulada = np.cumsum(psd_norm)

bw_95 = Fs_ECG_BT[np.where(Acumulada >= 0.95)[0][0]]
bw_98 = Fs_ECG_BT[np.where(Acumulada >= 0.98)[0][0]]

print(f"Ancho de banda para 95%: {bw_95:.2f} Hz")
print(f"Ancho de banda para 98%: {bw_98:.2f} Hz")

#%% Gráficos
fig, axs = plt.subplots(3, 1, figsize=(12, 10))

# Tiempo para el segmento
t_segment = np.linspace(5000/fs_ecg, 12000/fs_ecg, len(ecg_segment))

# Señal original (solo segmento)
axs[0].plot(t_segment, ecg_segment)
axs[0].set_title("Señal ECG (Segmento 5000 a 12000)")
axs[0].set_xlabel("Tiempo [s]")
axs[0].set_ylabel("Amplitud")
axs[0].grid(True)

# PSD del segmento
axs[1].plot(Fs_ECG_BT, Signal_ECG_Pxx)
axs[1].set_title("Densidad Espectral de Potencia (PSD) - Blackman-Tukey (Segmento)")
axs[1].set_xlabel("Frecuencia [Hz]")
axs[1].set_ylabel("Potencia")
axs[1].grid(True)

# PSD con anchos de banda del segmento
axs[2].plot(Fs_ECG_BT, Signal_ECG_Pxx)
axs[2].axvline(bw_95, color='r', linestyle='--', label=f'BW 95%: {bw_95:.2f} Hz')
axs[2].axvline(bw_98, color='g', linestyle='--', label=f'BW 98%: {bw_98:.2f} Hz')
axs[2].set_title("PSD con Ancho de Banda 95% y 98% (Segmento)")
axs[2].set_xlabel("Frecuencia [Hz]")
axs[2].set_ylabel("Potencia")
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()

#%%

# Parámetros
fs_ppg = 400  # Hz

# Cargar señal PPG
ppg = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)

# Normalizar señal completa
ppg = (ppg - np.mean(ppg)) / np.std(ppg)

# Selección de un segmento
start_sample = 2000
end_sample = 6000
ppg_segment = ppg[start_sample:end_sample]

N= end_sample - start_sample

# Eliminar tendencia lenta con media móvil
def detrend_signal(x, window_size):
    kernel = np.ones(window_size) / window_size
    trend = np.convolve(x, kernel, mode='same')
    return x - trend

ppg_segment_detrended = detrend_signal(ppg_segment, window_size=200)

# Re-normalización
ppg_segment_detrended = (ppg_segment_detrended - np.mean(ppg_segment_detrended)) / np.std(ppg_segment_detrended)

nperseg =  N // 8

# Estimar PSD con Welch 
frecs, psd = welch(ppg_segment_detrended,
                   fs=fs_ppg,
                   window='hamming',
                   nperseg = nperseg,
                   noverlap= nperseg / 2 )

# Calcular energía acumulada y anchos de banda
psd_norm = psd / np.sum(psd)
acumulada = np.cumsum(psd_norm)

bw_95 = frecs[np.where(acumulada >= 0.95)[0][0]]
bw_98 = frecs[np.where(acumulada >= 0.98)[0][0]]

print(f"Ancho de banda para 95% de la energía: {bw_95:.2f} Hz")
print(f"Ancho de banda para 98% de la energía: {bw_98:.2f} Hz")

# Gráficos
fig, axs = plt.subplots(3, 1, figsize=(12, 10))

t_segment = np.arange(start_sample, end_sample) / fs_ppg

# Señal original y detrended
axs[0].plot(t_segment, ppg_segment, label='Original')
axs[0].plot(t_segment, ppg_segment_detrended, label='Detrended')
axs[0].set_title("Segmento de señal PPG (Original vs Detrended)")
axs[0].set_xlabel("Tiempo [s]")
axs[0].set_ylabel("Amplitud")
axs[0].legend()
axs[0].grid(True)

# PSD
axs[1].plot(frecs, psd)
axs[1].set_title("PSD (Welch - Periodograma Modificado)")
axs[1].set_xlabel("Frecuencia [Hz]")
axs[1].set_ylabel("Densidad espectral de potencia")
axs[1].grid(True)

# PSD con anchos de banda
axs[2].plot(frecs, psd)
axs[2].axvline(bw_95, color='r', linestyle='--', label=f'BW 95%: {bw_95:.2f} Hz')
axs[2].axvline(bw_98, color='g', linestyle='--', label=f'BW 98%: {bw_98:.2f} Hz')
axs[2].set_title("PSD con Anchos de Banda 95% y 98%")
axs[2].set_xlabel("Frecuencia [Hz]")
axs[2].set_ylabel("Potencia")
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()

#%% Cuadro Comparativo de Anchos de Banda

data = [
    {
        'Señal': 'Audio',
        'Método PSD': 'Welch',
        'Segmento usado': 'Completo',
        'Fs [Hz]': fs_audio,
        'Ventana': 'Hann',
        'BW 95% [Hz]': round(bw_95, 2),
        'BW 98% [Hz]': round(bw_98, 2)
    },
    {
        'Señal': 'ECG',
        'Método PSD': 'Blackman-Tukey',
        'Segmento usado': '5000-12000',
        'Fs [Hz]': fs_ecg,
        'Ventana': 'Blackman',
        'BW 95% [Hz]': round(bw_95, 2),
        'BW 98% [Hz]': round(bw_98, 2)
    },
    {
        'Señal': 'PPG',
        'Método PSD': 'Welch',
        'Segmento usado': '2000-6000',
        'Fs [Hz]': fs_ppg,
        'Ventana': 'Hamming',
        'BW 95% [Hz]': round(bw_95, 2),
        'BW 98% [Hz]': round(bw_98, 2)
    }
]

# Crear DataFrame
df = pd.DataFrame(data)

# Mostrar como tabla bonita en consola
print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))
