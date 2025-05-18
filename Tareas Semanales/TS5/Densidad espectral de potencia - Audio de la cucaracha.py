# -*- coding: utf-8 -*-
"""
Created on Wed May  7 19:36:45 2025

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

#fs_audio, wav_data = sio.wavfile.read('la cucaracha.wav')
# fs_audio, wav_data = sio.wavfile.read('prueba psd.wav')
fs_audio, wav_data = sio.wavfile.read('silbido.wav')

# Calculo largo y defino solapamiento de Welch

N = len(wav_data)
nperseg = N // 6 

wav_data = wav_data / np.std(wav_data) # Normalizo dividiendo por STD

"""
Para poder normalizar podemos usar el metodo de dividir a la funcion por su 
desvio standar para así normalizarla en potencia 

"""

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


"""
IMPORTANTE

El Detrend, sirve para sacar tendencias que tenga la señal, al poner una lineal, esto hace que trate de sacarlo 
al poner una lineal que estime la señal. Entonces, este va a poner en cada uno de los bloques una lineal

De esta manera, si en dicho bloque tenemos mucha caida podemos acomodarlo si dividimos el N en una cantidad de bloques
que se acomode a nuestra necesidad

Si tenemos 40000 muestras y lo dividimos en 20 bloques, vamos a tener bloques de 2000 muestras, que si en cada bloque
podemos aislar las variaciones esto hace que mejore la estimacion

"""
# Convertir a decibeles
Signal_Audio_Pxx_mean_db = 10 * np.log10(Signal_Audio_Pxx)

#Signal_Audio_Pxx_mean_db -= np.max(Signal_Audio_Pxx_mean_db )  # Esto es normalizar el maximo a 0db 

# Esto tiene sentido porque tenemos distintos BW, para centrar todo en un mismo punto o que los picos esten en la misma escala

#%% Gracico PDS

# === Gráfica mejorada ===
plt.style.use('seaborn-v0_8-darkgrid')
plt.figure(figsize=(12, 6))
plt.plot(Fs_Audio_welch, Signal_Audio_Pxx_mean_db, color='tab:blue', lw=2, label='PSD (Welch)')

# Anotar la frecuencia del pico principal
peak_idx = np.argmax(Signal_Audio_Pxx_mean_db)
peak_freq = Fs_Audio_welch[peak_idx]
plt.annotate(f'Pico: {peak_freq:.1f} Hz',
             xy=(peak_freq, 0),
             xytext=(peak_freq + 100, -10),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             fontsize=10, backgroundcolor='white')

# Estética
plt.title(f'Densidad Espectral de Potencia de La Cucaracha', fontsize=14)
plt.xlabel('Frecuencia [Hz]', fontsize=12)
plt.ylabel('PSD [dB, normalizada]', fontsize=12)
#plt.ylim([-80, 5])
plt.xlim([0, fs_audio // 2])
plt.legend()
plt.tight_layout()
plt.show()

#%% Grafico de audio

plt.figure()
plt.plot(wav_data)

# si quieren oirlo, tienen que tener el siguiente módulo instalado
# pip install sounddevice
# import sounddevice as sd
# sd.play(wav_data, fs_audio)


#%% Comprobación de Parseval


# FFT directa de la señal completa
X = fft(wav_data)
N = len(wav_data)

# Energía en frecuencia con Parseval (fft normalizada)
energia_frecuencia = np.sum(np.abs(X)**2) / N

# Energía en tiempo 
energia_tiempo = np.sum(wav_data**2)

# Mostrar resultados
print("\n== Verificación de Parseval ==")
print(f"Energía (tiempo):     {energia_tiempo:.6f}")
print(f"Energía (frecuencia): {energia_frecuencia:.6f}")
print(f"Error relativo:       {100 * abs(energia_tiempo - energia_frecuencia) / energia_tiempo:.10f}%")

#%% Calculo de ancho de banda

# Voy a asumir que tiene un comportamiento Pasa Bajos para así tomar el acumulado desde 0 en adelante e ir 
# acumulando la potencia con Cumsum

#Potencia total (integral de la PSD)
potencia_total = np.sum(Signal_Audio_Pxx)

#Potencia acumulada (desde 0 Hz hacia adelante)
potencia_acumulada = np.cumsum(Signal_Audio_Pxx)

#Encontro la frecuencia donde se alcanza el 95% y 98%
umbral_95 = 0.95 * potencia_total
umbral_98 = 0.98 * potencia_total

# Buscar el índice donde se supera el umbral
idx_95 = np.where(potencia_acumulada >= umbral_95)[0][0]
idx_98 = np.where(potencia_acumulada >= umbral_98)[0][0]

# Ancho de banda
bw_95 = Fs_Audio_welch[idx_95]
bw_98 = Fs_Audio_welch[idx_98]

print(f"Ancho de banda para 95% de la potencia: {bw_95:.2f} Hz")
print(f"Ancho de banda para 98% de la potencia: {bw_98:.2f} Hz")

# === Graficar PSD con BW marcado ===
plt.figure(figsize=(12, 6))
plt.plot(Fs_Audio_welch, Signal_Audio_Pxx_mean_db, color='tab:blue', lw=2)
plt.axvline(bw_95, color='green', linestyle='--', label=f'BW 95% = {bw_95:.1f} Hz')
plt.axvline(bw_98, color='red', linestyle='--', label=f'BW 98% = {bw_98:.1f} Hz')
plt.title('Análisis de Ancho de Banda por Potencia Acumulada')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



"""
Como tenemos una señal de audio, buscamos el pico maximo y a partir de alli nos movemos a los costados
de manera de ir avanzando hasta encontrar el 95 o 98 % del valor maximo de la señal. Hay que hacerlo asi para la TS5
"""

# # Paso 1: Potencia total
# potencia_total = np.sum(Signal_Audio_Pxx)

# # Paso 2: Índice del pico principal
# peak_index = np.argmax(Signal_Audio_Pxx)

# # Paso 3: Inicialización para acumulación centrada
# acumulada = Signal_Audio_Pxx[peak_index]
# i_low = peak_index
# i_high = peak_index

# # Umbrales deseados
# umbral_95 = 0.95 * potencia_total
# umbral_98 = 0.98 * potencia_total

# # Guardamos BW cuando se cumplan los umbrales
# bw_95_found = False
# bw_98_found = False

# # Paso 4: Expandir hacia los costados
# while not bw_98_found and (i_low > 0 or i_high < len(Signal_Audio_Pxx) - 1):
#     # Expandimos hacia el lado con mayor potencia siguiente
#     if i_low > 0 and (i_high == len(Signal_Audio_Pxx) - 1 or Signal_Audio_Pxx[i_low - 1] >= Signal_Audio_Pxx[i_high + 1]):
#         i_low -= 1
#         acumulada += Signal_Audio_Pxx[i_low]
#     elif i_high < len(Signal_Audio_Pxx) - 1:
#         i_high += 1
#         acumulada += Signal_Audio_Pxx[i_high]

#     # Guardar anchos de banda cuando se alcanzan umbrales
#     if not bw_95_found and acumulada >= umbral_95:
#         bw_95_low = Fs_Audio_welch[i_low]
#         bw_95_high = Fs_Audio_welch[i_high]
#         bw_95_found = True

#     if not bw_98_found and acumulada >= umbral_98:
#         bw_98_low = Fs_Audio_welch[i_low]
#         bw_98_high = Fs_Audio_welch[i_high]
#         bw_98_found = True

# # Ancho de banda centrado
# bw_95 = bw_95_high - bw_95_low
# bw_98 = bw_98_high - bw_98_low

# print(f"BW centrado al 95% de la potencia: {bw_95:.2f} Hz (de {bw_95_low:.1f} Hz a {bw_95_high:.1f} Hz)")
# print(f"BW centrado al 98% de la potencia: {bw_98:.2f} Hz (de {bw_98_low:.1f} Hz a {bw_98_high:.1f} Hz)")

# # === Gráfico 1: PSD con líneas verticales que marcan los anchos de banda ===
# plt.figure(figsize=(12, 6))
# plt.plot(Fs_Audio_welch, Signal_Audio_Pxx_mean_db, color='tab:blue', lw=2, label='PSD (Welch)')

# # Marcar bandas de 95% y 98%
# plt.axvline(bw_95_low, color='green', linestyle='--', label=f'BW 95%: {bw_95_low:.1f}-{bw_95_high:.1f} Hz')
# plt.axvline(bw_95_high, color='green', linestyle='--')

# plt.axvline(bw_98_low, color='red', linestyle='--', label=f'BW 98%: {bw_98_low:.1f}-{bw_98_high:.1f} Hz')
# plt.axvline(bw_98_high, color='red', linestyle='--')

# # Anotar pico
# plt.axvline(Fs_Audio_welch[peak_index], color='black', linestyle=':', lw=1, label=f'Pico: {peak_freq:.1f} Hz')

# # Estética
# plt.title('Densidad Espectral de Potencia con Bandas de Potencia Centrada')
# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('PSD [dB, normalizada]')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # === Gráfico 2: Sombreado de bandas ===
# plt.figure(figsize=(12, 6))
# plt.plot(Fs_Audio_welch, Signal_Audio_Pxx, lw=2, color='tab:blue', label='PSD Lineal')

# # Rellenar bandas
# plt.fill_between(Fs_Audio_welch[i_low:i_high+1], Signal_Audio_Pxx[i_low:i_high+1], 
#                  color='orange', alpha=0.5, label='Potencia acumulada (98%)')

# plt.axvline(Fs_Audio_welch[peak_index], color='black', linestyle=':', lw=1, label='Pico')

# # Estética
# plt.title('Zonas de Potencia Acumulada en PSD')
# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('PSD [potencia/Hz]')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()



