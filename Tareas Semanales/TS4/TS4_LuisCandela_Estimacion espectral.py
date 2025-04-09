# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 19:16:10 2025

@author: candell1
"""

#%% 

""" 

TAREA SEMANAL 4 - Primeras nociones de la estimacion espectral

"""

#%% módulos y funciones a importar

import numpy as np
from scipy import signal
from scipy.signal.windows import hamming, hann, blackman, kaiser
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

fr = np.random.uniform(-1/2,1/2,N_Test).reshape(1,N_Test)      # Declaro mi vector 1x200 al pasar 200 valores a la uniforme     
                                                            # Fuerzo las dimensiones
Omega_1 = Omega_0 + fr*df                                   # Genero mi Omega 1

# Genero vector de tiempo para poder meterlo como mi vector de 1000 x 200 en el seno 
tt = np.linspace(0, (N-1)*ts, N).reshape(N,1)

tt = np.tile(tt, (1, N_Test))                               # Genero la matriz de 1000 x 200

# Al mutiplicar con * hacemos el producto matricial para que quede de 1000x200
S = A1 * np.sin(2 * np.pi * Omega_1 * tt)

#%% Genereo el ruido para la señal

# Para poder general la señal de ruido, tenemos que tener una distribucion normal con un N(o,sigma)

Media = 0                   # Media
SD_Sigma = np.sqrt(Sigma2)  # Desvio standar 

nn = np.random.normal(Media, SD_Sigma, N).reshape(N,1)              # Genero señal de ruido

nn = np.tile(nn, (1,N_Test))                                        # Ahora tengo que generaer mi matriz de ruido de 200x1000


#%% Ahora genero mi señal final sumando las matrices

# Esto seria mi x(k) = a1 * sen(W1 * k ) + n(k), siendo N(k) el ruido
Signal = S + nn

#%% Calcular la FFT de cada señal en la matriz Signal

X_f = fft(Signal, axis=0)  # FFT en cada columna (cada señal)
X_f = fftshift(X_f, axes=0)  # Centramos el espectro

# Generamos el eje de frecuencias
frec = np.linspace(-fs/2, fs/2, N)  # Eje de frecuencias

#%% Graficamos la magnitud de la FFT para algunas señales

num_senales = 10  # Número de señales a graficar
indices = np.linspace(0, N_Test-1, num_senales, dtype=int)  # Seleccionamos señales espaciadas

plt.figure(figsize=(10, 6))

for i in indices:
    plt.plot(frec, np.abs(X_f[:, i]), label=f'Señal {i+1}')  # Magnitud de la FFT

# Configuración de la gráfica
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud |X(f)|")
plt.title("Espectro de algunas señales en Signal")
plt.legend()
plt.grid(True)

# Mostrar la gráfica
plt.show()

#%% Estimadores a1 para 5 señales, cada una con una ventana distinta

np.random.seed(42)
random_indices = np.random.choice(N_Test, size=5, replace=False)

# Ventanas (una por señal seleccionada)
window_names = ['Rectangular', 'Hamming', 'Hann', 'Blackman', 'Kaiser (β=14)']
windows = [
    np.ones(N),
    hamming(N),
    hann(N),
    blackman(N),
    kaiser(N, beta=14)
]

# Almacenar estimadores
a1_estimados = []

# Eje de frecuencia
frec = fftshift(np.fft.fftfreq(N, ts))

# Subplots: comparar espectros original vs ventaneado (zoom cerca de 250 Hz)
fig, axs = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

for i, idx in enumerate(random_indices):
    xk = Signal[:, idx]
    win = windows[i]

    # Señal sin ventana (rectangular)
    Xk_orig = np.abs(fftshift(fft(xk)))
    # Señal con ventana
    xk_win = xk * win
    Xk_win = np.abs(fftshift(fft(xk_win)))

    # Estimador: valor pico del espectro (normalizado por N)
    a1_hat = np.max(Xk_win) / N
    a1_estimados.append(a1_hat)

    # Graficamos ambos espectros
    axs[i].plot(frec, Xk_orig, label='Original (Rectangular)', color='gray', alpha=0.6)
    axs[i].plot(frec, Xk_win, label=f'{window_names[i]}', linewidth=1.2)
    axs[i].set_title(f'Espectro Señal {idx} - Ventana: {window_names[i]}')
    axs[i].grid(True)
    axs[i].legend()
    axs[i].set_xlim(230, 270)  # ZOOM en la frecuencia de interés

axs[-1].set_xlabel('Frecuencia (Hz)')
fig.suptitle('Comparación del espectro original vs con ventana (Zoom en 250 Hz)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

#%% Gráfico comparando los estimadores a1 obtenidos

plt.figure(figsize=(8, 5))
plt.bar(window_names, a1_estimados, color='skyblue')
#plt.axhline(y=A1, color='red', linestyle='--', label=r'Valor real $a_1 = \sqrt{2}$')
plt.ylabel('Estimación de $a_1$')
plt.title('Comparación de estimadores $a_1$ con diferentes ventanas')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


