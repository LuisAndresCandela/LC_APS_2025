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

nn = np.random.normal(Media, SD_Sigma, N).reshape(N,1)              # Genero señal de ruido

nn = np.tile(nn, (1,N_Test))                                        # Ahora tengo que generaer mi matriz de ruido de 200x1000


#%% Ahora genero mi señal final sumando las matrices

# Esto seria mi x(k) = a1 * sen(W1 * k ) + n(k), siendo N(k) el ruido
Signal = S + nn

#%% Calculo las FFT de cada versión ( ventana, Hann, FlatTop, Blackman-Harris   )

#Declaro las ventanas que voy a usar y le doy N valores. Fuerzo dimensiones

ventanas = {
    'Rectangular': np.ones((N, 1)),
    'Hann': hann(N).reshape(N, 1),
    'FlatTop': flattop(N).reshape(N, 1),
    'Blackman-Harris': blackmanharris(N).reshape(N, 1)
}

num_senales = 200                                               # Cantidad de realizaciones
indices = np.linspace(0, N_Test - 1, num_senales, dtype=int)

# Eje de frecuencia solo parte positiva (0 a fs/2)
frec_pos = np.linspace(0, fs/2, N//2)

plt.figure(figsize=(12, 8))

# Ahora con mi vector ventana puedo ir generando cada una de las señales ventaneadas

for nombre, ventana in ventanas.items():
    
    S_windowed = Signal * ventana                   # Aplicamos la ventana

    X_f = fft(S_windowed, axis=0)                   # FFT y nos quedamos con la mitad positiva
    X_f = X_f[:N//2, :]  

    # Normalización por cada señal para llevar el pico a 0 dB
    X_f_norm = X_f / np.max(np.abs(X_f), axis=0)

    # Graficamos promedio de espectros en dB 
    espectro_promedio_db = 20 * np.log10(np.mean(np.abs(X_f_norm), axis=1) + 1e-12)
    
    plt.plot(frec_pos, espectro_promedio_db, label=nombre)

plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud |X(f)| [dB]")
plt.title("Comparación de ventanas - Parte positiva del espectro (Normalizado)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#%% Estimador a1

# Acá estamos calculando cada uno de los estimadores para el promedio de todas las señales que tenemos en N/4
# En esto queremos tener la menor dispersion posible, por lo que, queremos que nuestro histograma este lo mas comprimido posible
# Y si vemos el grafico, como todas estan masomenos centradas en le mismo lugar, implica que tenemos un sesgo similar
# entonces lo que nos va a determinar acá todo es la varianza o que tan comprimido esta

# Por otro lado, el sesgo ( donde esta centrado ) despues lo puedo modificar para que se acerque a mi valor real de a1
# En cambio, la varianza no. Por lo que, me va a interar tener la menor varianza posible, ya que esto no lo puedo calibrar o mejorar de ninguna otra manera


# Índice correspondiente nuestro Omega_0
f_target = Omega_0
index_f_target = int(f_target / df)             # Con esto obtengo 250 hz

# Estimadores por ventana
estimadores = {}

for nombre, ventana in ventanas.items():
    
    # Normalización de la energía de la ventana
    win_rms = np.sqrt(np.mean(ventana**2))  
    Signal_windowed = Signal * ventana / win_rms
    
    # FFT normalizada
    X_f = fft(Signal_windowed, axis=0) / N  
    
    # Calculo estimador en nuestro Omega_0
    a1_estimado = np.abs(X_f[index_f_target, :])
    
    # Almaceno los estimadores
    estimadores[nombre] = a1_estimado

# Un solo gráfico con todos los histogramas
plt.figure(figsize=(12, 7))

for nombre, est in estimadores.items():
    plt.hist(est, bins=30, alpha=0.5, label=nombre, edgecolor='k', density=True)

# Líneas verticales: valor real y referencias
plt.axvline(A1, color='red', linestyle='--', label=r'Valor real $a_1 = \sqrt{2}$')

# También mostramos la media de cada estimador con una línea punteada verde
for nombre, est in estimadores.items():
    plt.axvline(np.mean(est), linestyle='--', label=f'Media {nombre}: {np.mean(est):.2f}')

plt.xlabel(r'Estimador $\hat{a}_1$')
plt.ylabel('Densidad')
plt.title('Distribución del estimador $\hat{a}_1$ para distintas ventanas')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Calculo y hago tabla de sesgo y varianza de cada una de los ventaneos 

print("-" * 65)
print(f"{'Estimador a_1':<15}")
print(f"{'Ventana':<15}{'Esperanza E[â1]':>20}{'Sesgo':>15}{'Varianza':>15}")
print("-" * 65)

# Accedo al array de estimadores, cada posicion tiene los 200 valores de a1 para cada ventana
for nombre, est in estimadores.items():         
    media = np.mean(est)
    varianza = np.var(est)
    sesgo = media - A1
    print(f"{nombre:<15}{media:>20.4f}{sesgo:>15.4f}{varianza:>15.4f}")
    
#%% Estimador de frecuencia: omega1 = argmax |X(omega)|

estimadores_omega1 = {}

for nombre, ventana in ventanas.items():
    
    # Normalizamos energía RMS
    win_rms = np.sqrt(np.mean(ventana**2))
    Signal_windowed = Signal * ventana / win_rms

    # FFT (normalizada)
    X_f = fft(Signal_windowed, axis=0)
    X_f = X_f[:N//2, :]  # Parte positiva

    # Índice de máxima magnitud para cada realizacion
    idx_max = np.argmax(np.abs(X_f), axis=0)    

    # Convertimos índices a frecuencia
    omega1_estimado = idx_max * df
    
    # Almacenamos estimadores
    estimadores_omega1[nombre] = omega1_estimado

#%% Histograma de los estimadores omega1

plt.figure(figsize=(12, 7))

for nombre, est in estimadores_omega1.items():
    plt.hist(est, bins=30, alpha=0.5, label=nombre, edgecolor='k', density=True)

# Línea del valor real de Omega_1 (el centro del intervalo)
plt.axvline(Omega_0, color='red', linestyle='--', label=r'Valor real $\Omega_0$')

# También mostramos la media de cada estimador con una línea punteada verde
for nombre, est in estimadores_omega1.items():
    plt.axvline(np.mean(est), linestyle='--', label=f'Media {nombre}: {np.mean(est):.2f} Hz')

plt.xlabel(r'Estimador $\hat{\omega}_1$ [Hz]')
plt.ylabel('Densidad')
plt.title('Distribución del estimador $\hat{\omega}_1$ para distintas ventanas')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Tabla de sesgo y varianza para omega1

print("-" * 65)
print(f"{'Estimador Omega_1':<15}")
print(f"{'Ventana':<15}{'Esperanza E[ŵ1]':>20}{'Sesgo':>15}{'Varianza':>15}")
print("-" * 65)

for nombre, est in estimadores_omega1.items():
    media = np.mean(est)
    varianza = np.var(est)
    sesgo = media - Omega_0
    print(f"{nombre:<15}{media:>20.4f}{sesgo:>15.4f}{varianza:>15.4f}")


#%%

# El estimador Omega 1 va a ser el sobre el modulo de la FFT de la señal Omega1 = argmax|X(omega)|
# Donde esa operacioj devuelve el valor de omega que maximiza el modulo de la TF

# Vemos que obtenemos que caso todo esta al rededor de 250 hz que es donde esta el maximo, pero al tener una res. espectral regular
# A lo sumo nos vamos un poco del 250

#%% Hacemos Cero Padding para aumentar la resolucion espectral y asi vemos mejor como son las variaciones de Omega1

estimadores_omega1_CeroPadding = {}

for nombre, ventana in ventanas.items():
    # Normalizamos energía 
    win_rms = np.sqrt(np.mean(ventana**2))
    Signal_windowed = Signal * ventana / win_rms

    # Agrego cantidad de ceros a agregar
    N_ZP = 5*N

    # FFT (normalizada)
    X_f = fft(Signal_windowed,N_ZP, axis=0) # FFT Con cero padding
    X_f = X_f[:N_ZP//2, :]                  # Parte positiva teniendo en cuenta el cambio de N

    # Índice de máxima magnitud para cada prueba
    idx_max = np.argmax(np.abs(X_f), axis=0)    
    
    # Recalculo la resolucion espectral
    df_ZP = fs / N_ZP

    # Convertimos índices a frecuencia
    omega1_estimado_CeroPadding = idx_max * df_ZP
    
    # Almaceno estimadores
    estimadores_omega1_CeroPadding [nombre] = omega1_estimado_CeroPadding 

#%% Histograma de los estimadores  con ZP

plt.figure(figsize=(12, 7))

for nombre, est in estimadores_omega1_CeroPadding.items():
    plt.hist(est, bins=30, alpha=0.5, label=nombre, edgecolor='k', density=True)

# Línea del valor real de Omega_1 (el centro del intervalo)
plt.axvline(Omega_0, color='red', linestyle='--', label=r'Valor real $\Omega_0$')

# También mostramos la media de cada estimador con una línea punteada verde
for nombre, est in estimadores_omega1_CeroPadding.items():
    plt.axvline(np.mean(est), linestyle='--', label=f'Media {nombre}: {np.mean(est):.2f} Hz')

plt.xlabel(r'Estimador $\hat{\omega}_1$ [Hz]')
plt.ylabel('Densidad')
plt.title('Distribución del estimador $\hat{\omega}_1$ para distintas ventanas con Zero Padding de N_ZP ceros ')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Tabla de sesgo y varianza para omega1 con ZP

print("-" * 65)
print(f"{'Estimador Omega_1 con ZP':<15}")
print(f"{'Ventana con ZP':<15}{'Esperanza E[ŵ1]':>20}{'Sesgo':>15}{'Varianza':>15}")
print("-" * 65)

for nombre, est in estimadores_omega1_CeroPadding.items():
    media = np.mean(est)
    varianza = np.var(est)
    sesgo = media - Omega_0
    print(f"{nombre:<15}{media:>20.4f}{sesgo:>15.4f}{varianza:>15.4f}")

#%% Señal con ventana Hann (ya normalizada en energía RMS)
ventana = ventanas['Hann']
win_rms = np.sqrt(np.mean(ventana**2))
Signal_windowed = Signal * ventana / win_rms

# FFT sin zero padding
X_f_noZP = fft(Signal_windowed, axis=0)
X_f_noZP = X_f_noZP[:N//2, :]
f_noZP = np.linspace(0, fs/2, N//2)
mag_noZP_dB = 20 * np.log10(np.mean(np.abs(X_f_noZP), axis=1) + 1e-12)  # Evitar log(0)

# FFT con zero padding
N_ZP = 5 * N
X_f_ZP = fft(Signal_windowed, N_ZP, axis=0)
X_f_ZP = X_f_ZP[:N_ZP//2, :]
f_ZP = np.linspace(0, fs/2, N_ZP//2)
mag_ZP_dB = 20 * np.log10(np.mean(np.abs(X_f_ZP), axis=1) + 1e-12)

# Gráfico en dB
plt.figure(figsize=(12, 6))
plt.plot(f_noZP, mag_noZP_dB, label='Sin Zero Padding', lw=2)
plt.plot(f_ZP, mag_ZP_dB, label='Con Zero Padding (5N)', lw=2, linestyle='--')
plt.axvline(Omega_0, color='red', linestyle=':', label=r'$\Omega_0$ Real')

plt.title('Espectro promedio (ventana Hann) con y sin Zero Padding [dB]')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
    
#%%

""" 

Hacer para SNR = 3 y luego una tabla comparativa de las de los estimadores de cada ventana con las 2 SNR que calculamos

Calcular el sesgo y varianza para cada ventana y con eso tambien lo ponemos en la tabla

""" 

#%% Agregar, comparacion de una ventana elegida en base a la varianza y comparar los espectros con y sin zero padding

# Que implica agregar zeros a la señal ? O agregar informacion extra a la señal 

# Podemos tener ventanas que funcionen mejor en amplitud y otras que funcionan mejor en frecuencia analizar esto

# TS3 - Histograma // Es sobre los errores de cuantizacion que tenemos al muestrear la señal 
# El histograma es el conteo de esos errores y no podemos tener mas error que el paso de cuantizacion.
# Revisar escala para que el error no sea mayor que el paso de cuantizacion