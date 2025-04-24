# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 20:42:56 2025

@author: candell1
"""
#%% 

"""

Ahora queremos implementar Blackman Tuckey

"""


#%% Importo modulos

import numpy as np
from scipy import signal
from scipy.signal import welch
from scipy.signal import correlate
from scipy.signal.windows import hamming, hann, blackman, kaiser, flattop, blackmanharris
from scipy.fft import fft, fftshift
import matplotlib as mpl
import matplotlib.pyplot as plt

#%% Genero mi Omega 1 que va a ser una frecuencia original mas una inceretidumbre en la frecuencia
# Esto genera que no sepa la frecuencia que estamos, sino que solo sepamos entre que valores va a estar

#%% Datos de la simulacion

fs = 1000           # frecuencia de muestreo (Hz)
N = 1000            # cantidad de muestras
ts = 1/fs           # tiempo de muestreo
df = fs/N           # resolución espectral

N_Test = 200        # Numero de pruebas

SNR = 10            # Signal to Noise Ratio 

Sigma2 = 10**-1     # Despejando el PN de la ecuacion de SNR llego a este valor

Omega_0 = fs/4      # Nos ponemos a mitad de banda digital

#%% Genero mi vector de 1000x200 de la senoidal

A1 = np.sqrt(2)                                             # Genero la amplitud de manera que el seno quede normalizado

fr = np.random.uniform(-1/2,1/2,N_Test).reshape(1,N_Test)   # Declaro mi vector 1x200 al pasar 200 valores a la uniforme     
                                                            # Fuerzo las dimensiones
Omega_1 = Omega_0 + fr*df                                   # Genero mi Omega 1

# Genero vector de tiempo para poder meterlo como mi vector de 1000 x 200 en el seno 
tt = np.linspace(0, (N-1)*ts, N).reshape(N,1)

tt = np.tile(tt, (1, N_Test))                               # Genero la matriz de 1000 x 200

# Al mutiplicar con * hacemos el producto matricial para que quede de 1000x200
S = A1 * np.sin(2 * np.pi * Omega_1 * tt)

varianzas = np.var(S, axis=0)                               # Varianza por cada una de las 200 señales
varianza_promedio = np.mean(varianzas)                      # Compruebo varianza 1 


#%% Genereo el ruido para la señal

# Para poder general la señal de ruido, tenemos que tener una distribucion normal con un N(o,sigma)

Media = 0                   # Media
SD_Sigma = np.sqrt(Sigma2)  # Desvio standar 

#nn = np.random.normal(Media, SD_Sigma, N).reshape(N,1)              # Genero señal de ruido

nn = np.random.normal(Media, SD_Sigma, (N, N_Test))  # Directamente 1000 x 200


#nn = np.tile(nn, (1,N_Test))                                        # Ahora tengo que generaer mi matriz de ruido de 200x1000


#%% Ahora genero mi señal final sumando las matrices

# Esto seria mi x(k) = a1 * sen(W1 * k ) + n(k), siendo N(k) el ruido
Signal = S + nn 


#%% Desarrollo funcion de blackman tuckey

def blackman_tukey_psd(x, window_len=None, window_func=blackman, fs=1.0):
    
    N = len(x)
    if window_len is None:
        window_len = N // 4  # Longitud típica

    # Autocorrelación estimada
    rxx = np.correlate(x, x, mode='full') / N       # La autocorrelacion en modo full calcula todos los posibles desplazamientos
    mid = len(rxx) // 2                             # Punto central del Array
    rxx = rxx[mid:mid + window_len]                 # solo parte no negativa o derecha del espectro

    # Ventaneo
    ventana = window_func(window_len)
    rxx_win = rxx * ventana
    
    # FFT y espectro
    Pxx = np.abs(fft(rxx_win, n=2*window_len))          # Hago la fft solamente de del doble de la duracion de la ventana
    f_Pxx = np.fft.fftfreq(2 * window_len, d=1/fs)      # Busco frecuencias usadas en la fft 

    # Quedarse con parte positiva
    mask = f_Pxx >= 0
    return f_Pxx[mask], Pxx[mask]


#%% APlico funcion BT

Pxx_all = []

for i in range(Signal.shape[1]):
    _, Pxx_i = blackman_tukey_psd(x=Signal[:, i], window_len=None, window_func=blackman, fs=fs)
    Pxx_all.append(Pxx_i)

# Convertir la lista a array 2D (200 x freqs)
Signal_BT = np.array(Pxx_all)

# Frecuencia (eje x)
F_BT = np.linspace(0, fs/2, Signal_BT.shape[1])  # eje x para todos

# Graficar en dB
plt.figure(figsize=(10, 5))

for i in range(Signal_BT.shape[0]):
    plt.plot(F_BT, 10 * np.log10(Signal_BT[i, :] + 1e-12), alpha=0.3, color='steelblue')

plt.title('Espectros individuales en dB (Blackman-Tukey) de las 200 señales')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.grid(True)
plt.tight_layout()
plt.show()



# Inicializamos listas para los estimadores
a2_bt = []
w2_bt = []

for i in range(Signal.shape[1]):
    f, Pxx_i = blackman_tukey_psd(x=Signal[:, i], window_len=None, window_func=blackman, fs=fs)
    
    # Estimador de frecuencia: frecuencia donde el PSD tiene el máximo
    idx_max = np.argmax(Pxx_i)
    w2_bt.append(f[idx_max])
    
    # Estimador de amplitud: PSD en la frecuencia pico -> raíz para volver a amplitud
    a2_bt.append(np.sqrt(2 * Pxx_i[idx_max]))

# Convertimos a arrays
a2_bt = np.array(a2_bt)
w2_bt = np.array(w2_bt)

# Estadísticas
a2_media = np.mean(a2_bt)
a2_var = np.var(a2_bt)
a2_sesgo = a2_media - A1

w2_media = np.mean(w2_bt)
w2_var = np.var(w2_bt)
w2_sesgo = w2_media - Omega_0

# Tabla
print("-" * 65)
print(f"{'Estimador a2 (amplitud)':<25} Media: {a2_media:.4f}  Sesgo: {a2_sesgo:.4f}  Var: {a2_var:.4f}")
print(f"{'Estimador w2 (frecuencia)':<25} Media: {w2_media:.4f}  Sesgo: {w2_sesgo:.4f}  Var: {w2_var:.4f}")
print("-" * 65)

plt.figure(figsize=(12, 5))

# Amplitud
plt.subplot(1, 2, 1)
plt.hist(a2_bt, bins=30, color='skyblue', edgecolor='k', density=True)
plt.axvline(A1, color='red', linestyle='--', label='Valor real')
plt.axvline(a2_media, color='green', linestyle='--', label=f'Media: {a2_media:.2f}')
plt.title('Histograma de $\hat{a}_2$ (Blackman-Tukey)')
plt.xlabel('Amplitud estimada')
plt.ylabel('Densidad')
plt.grid(True)
plt.legend()

# Frecuencia
plt.subplot(1, 2, 2)
plt.hist(w2_bt, bins=30, color='salmon', edgecolor='k', density=True)
plt.axvline(Omega_0, color='red', linestyle='--', label='Valor real')
plt.axvline(w2_media, color='green', linestyle='--', label=f'Media: {w2_media:.2f} Hz')
plt.title('Histograma de $\hat{\omega}_2$ (Blackman-Tukey)')
plt.xlabel('Frecuencia estimada [Hz]')
plt.ylabel('Densidad')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

