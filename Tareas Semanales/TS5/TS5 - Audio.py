# -*- coding: utf-8 -*-
"""
Created on Sat May 17 18:33:56 2025

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
plt.style.use('seaborn-v0_8-darkgrid')
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

