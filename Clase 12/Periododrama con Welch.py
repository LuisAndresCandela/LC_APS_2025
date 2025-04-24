# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 18:26:38 2025

@author: candell1
"""

#%% Periodograma

#%% Como funciona la funcion periodograma ?

"""
periodogram(x, fs=1.0, window='boxcar', nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1)

--> Windows ;   Nos permite poner una ventana a placer, por defecto es una rectangular

--> nfft ;      Que hace ? - 

--> detren ;    Sacamos una tendencia en las muestras que por defecto es una costante que seria 
                    sacar el valor medio del espectro lo que genera que saquemosla energia
                Le podemos pasar constant o linear 

--> return_onesided;    Devolvemos la mitad del espectro lo cual tiene sentido ya que este es simetrico. 
                        Ojo, al hacer esto como al calcular la energia tenemos que hacerlo sobre todo el espectro
                            al hacer esto implicitamente esta multiplicando por 2 para compensar esto
                            
--> scaling ;   Usamos siempre density

Imp; Esto devuelve f, Pxx que serian las frecuencias de sampleo del periodrograma y el periodrograma propiamente dicho

"""

#%% Ahora veamos como funciona la funcion de Welch

"""
welch(x, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None, detrend='constant', 
      
                                                  return_onesided=True, scaling='density', axis=-1, average='mean')


    fs = 1 ;    implica que esta normalizada lo que hacer que el area coincida con el sigma cuadrado
    
   nprseg ;     Es la cantidad de bloques que vamos a tener, por defecto pone ninguno lo que equivale 
                   a un solo bloque del ancho del array.
                Un valor que tiene sentido para esto es que para 1000 muestras ponemos >= 5 para así promediar
                    5 o mas espectros. Por ejemplo N/4 o N/6
   
   noverlap ;   Cantidad de solapamiento entre bloques. Lo tipico es un 50% de solapamiento
   
   average ;    Como hace el promediado de los espectros. Por defecto usa el valor media o tambien la mediana
   

"""

#%% Importo la señal de la TS4 para ahora usar estas funciones

#%% Importo modulos

import numpy as np
from scipy import signal
from scipy.signal import welch
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

nn = np.random.normal(Media, SD_Sigma, (N, N_Test))           # Genero señal de ruido

#nn = np.tile(nn, (1,N_Test))                                        # Ahora tengo que generaer mi matriz de ruido de 200x1000


#%% Ahora genero mi señal final sumando las matrices

# Esto seria mi x(k) = a1 * sen(W1 * k ) + n(k), siendo N(k) el ruido
Signal = S + nn 

#%% Ahora vamos a implementar la funcion de Welch para calcular el espectro de Signal

nperseg = N // 6

Fs_welch, Signal_Pxx = welch(x = Signal , fs = fs, window='hann', nperseg = nperseg,
                             noverlap = nperseg//2, nfft=None, detrend='linear', 
                             return_onesided=True, scaling='density', 
                             axis=0, average='median')

Signal_Pxx_mean = Signal_Pxx # No promedio

#Signal_Pxx_mean = Signal_Pxx

# Convertir a decibeles
Signal_Pxx_mean_db = 10 * np.log10(Signal_Pxx_mean)
Signal_Pxx_mean_db -= np.max(Signal_Pxx_mean_db)  # Normalización para que el pico llegue a 0

# Graficar
plt.figure(figsize=(10, 5))
plt.plot(Fs_welch, Signal_Pxx_mean_db)
plt.title('PSD promedio (normalizada) en dB sobre {} señales'.format(N_Test))
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB, normalizada]')
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Calculo estimadores a2 y w2

# Índice correspondiente a Omega_0

# Buscamos la frecuencia más cercana a Omega_0 en el eje de frecuencia de Welch
index_f_target = np.argmin(np.abs(Fs_welch - Omega_0))

# PSD con Welch (ya calculada antes)
# Fs_welch, Signal_Pxx = welch(...)

# Estimador de amplitud a partir del pico en Omega_0 (densidad espectral en esa frecuencia)
a2_estimado = np.sqrt(2 * Signal_Pxx[index_f_target, :])  # Factor raíz(2) para normalizacion

# Estadísticas
a2_media = np.mean(a2_estimado)
a2_var = np.var(a2_estimado)
a2_sesgo = a2_media - A1

# Imprimir resultados
print(f"\nEstimador a2 (Welch):")
print(f"Esperanza E[â2]: {a2_media:.4f}")
print(f"Sesgo: {a2_sesgo:.4f}")
print(f"Varianza: {a2_var:.4f}")

# Graficar el histograma de los estimadores a2
plt.figure(figsize=(12, 7))
plt.hist(a2_estimado, bins=30, alpha=0.7, edgecolor='k', density=True, color='b', label=r'Est. $\hat{a}_2$ (Welch)')
plt.axvline(A1, color='red', linestyle='--', label=r'Valor real $a_1 = \sqrt{2}$')
plt.axvline(np.mean(a2_estimado), linestyle='--', color='g', label=f'Media: {np.mean(a2_estimado):.2f}')
plt.xlabel(r'Estimador $\hat{a}_2$')
plt.ylabel('Densidad')
plt.title('Distribución del estimador $\hat{a}_2$ (Amplitud con Welch)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Calculo w2 y grafico histogramas

# Índices de máximos en el eje de frecuencia (para cada realización)
idx_max_welch = np.argmax(Signal_Pxx, axis=0)
w2_estimado = Fs_welch[idx_max_welch]  # Convertimos a frecuencia en Hz

# Estadísticas
w2_media = np.mean(w2_estimado)
w2_var = np.var(w2_estimado)
w2_sesgo = w2_media - Omega_0

# Imprimir resultados
print(f"\nEstimador w2 (Welch):")
print(f"Esperanza E[ŵ2]: {w2_media:.4f} Hz")
print(f"Sesgo: {w2_sesgo:.4f} Hz")
print(f"Varianza: {w2_var:.4f} Hz²")

#Histogramas para w2
plt.figure(figsize=(12, 7))
plt.hist(w2_estimado, bins=30, alpha=0.7, edgecolor='k', density=True, color='r', label=r'Est. $\hat{w}_2$ (Welch)')
plt.axvline(Omega_0, color='red', linestyle='--', label=r'Valor real $\Omega_0$')
plt.axvline(np.mean(w2_estimado), linestyle='--', color='g', label=f'Media: {np.mean(w2_estimado):.2f} Hz')
plt.xlabel(r'Estimador $\hat{\omega}_2$ [Hz]')
plt.ylabel('Densidad')
plt.title('Distribución del estimador $\hat{\omega}_2$ (Frecuencia con Welch)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Algunas conclusiones 

"""
Algunas conclusiones

Si bien el grafico de la representacion del espectro con Welch observamos que tiene una menor varianza en general 
podemos apreciar que el pico donde esta la senoidal tiene una parte plana, esto ocurre debido a que al usar welch
y dividir en 6 grupos a nuestars 1000 muestras en definitiva estamos perdiendo resolucion espectral 

Esto se peude ver mas claramente al ver el estimador w2 el cual tiene una alta varianza. Es lo mismo que ocurria antes de 
que hicieramos el ZP en la TS4, en el cual tenemos menos ¨bines¨ en los cuales puede caer la frecuencia por lo que todos 
los valores intermedios de frecuencia los perdemos 

"""

#%% Hacemos el periodograma como lo haciamos antes solo con hann para comparar resultados


# Ventana Hann (normalizada en energía RMS)
ventana = hann(N).reshape(N, 1)
win_rms = np.sqrt(np.mean(ventana**2))
Signal_windowed = Signal * ventana / win_rms

# FFT
X_f = fft(Signal_windowed, axis=0)
X_f = X_f[:N//2, :]
frec_pos = np.linspace(0, fs/2, N//2)

# Magnitud promedio en dB y normalizada (pico en 0 dB)
mag_prom = np.mean(np.abs(X_f), axis=1)
mag_dB = 20 * np.log10(mag_prom + 1e-12)
mag_dB_norm = mag_dB - np.max(mag_dB)  # Normalización

# Plot
plt.figure(figsize=(10, 5))
plt.plot(frec_pos, mag_dB_norm, label='Ventana Hann', lw=2)
plt.axvline(Omega_0, color='red', linestyle=':', label=r'$\Omega_0$ Real')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.title('Espectro promedio normalizado (ventana Hann)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

index_f_target = int(Omega_0 / df)

X_f_a1 = fft(Signal_windowed, axis=0) / N
a1_estimado = np.abs(X_f_a1[index_f_target, :])
media_a1 = np.mean(a1_estimado)
sesgo_a1 = media_a1 - A1
varianza_a1 = np.var(a1_estimado)

idx_max = np.argmax(np.abs(X_f), axis=0)
omega1_estimado = idx_max * df
media_w1 = np.mean(omega1_estimado)
sesgo_w1 = media_w1 - Omega_0
varianza_w1 = np.var(omega1_estimado)

print(f"--------------------------")

print(f"Estimador a1 (Hann)")
print(f"  Media     : {media_a1:.4f}")
print(f"  Sesgo     : {sesgo_a1:.4f}")
print(f"  Varianza  : {varianza_a1:.4f}")

print(f"\nEstimador ω1 (Hann)")
print(f"  Media     : {media_w1:.4f} Hz")
print(f"  Sesgo     : {sesgo_w1:.4f} Hz")
print(f"  Varianza  : {varianza_w1:.4f} Hz²")
