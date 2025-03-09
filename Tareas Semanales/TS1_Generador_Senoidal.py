# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 20:28:48 2025

@author: candell1
"""

#%% Importacion de modulos a utilizar

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# import pdsmodulos as pds

#%% Algunas premisas a tener en cuenta

# Quiero generar un generador de senos, para lo cual vamos a hacer una 
# funcion, la cual admita los siguientes parametros:
    # Amplitud maxima de la senoidal ( Volts )
    # Valor medio ( Volts )
    # Frecuencia (Hz)
    # Fase ( Radianes )
    # Cantidad de muestras tomadas por el ADC ( # muestras o N )
    # Frecuencia de muestreo ( Fs )
    
#%% Definicion de funcion generadora de senos

# Vmax,             Amplitud maxima
# Vmed,             Valor medio
# f0,               Frecuencia de la señal
# Fase,             Fase de la señal
# Num_Muestras      Cantidad de muestras tomadas por el ADC
# Fs,               Frecuencia de muestreo

def Generador_Senoidal ( Vmax, Vmed, f0, Fase, Num_Muestras, fs ):
    
    # Calculamos el tiempo de sampleo
    ts = 1/fs
    
    # Grilla de sampleo temporal
    tt = np.linspace(0, (Num_Muestras-1)*ts, Num_Muestras)
    
    Seno_Generado = Vmax * np.sin( 2 * np.pi * f0 * tt + Fase ) + Vmed
    
    return tt, Seno_Generado

#%% Finalizacion de funcion 

#%% implementacion de funcion 

Vmax = 5
Vmed = 2
f0 = 100
Phi = np.pi / 2    # Fase en radianes
N = 1000
fs = 1000

tt, seno = Generador_Senoidal( Vmax, Vmed, f0, Phi, N, fs )

#%% Ploteo de la funcion

plt.figure(1)
line_hdls = plt.plot( tt, seno )

#%% Generador SINC

def Generador_Sinc ( Vmax, Vmed, f0, Fase, Num_Muestras, fs ):
    
    # Calculamos el tiempo de sampleo
    ts = 1/fs
    
    # Grilla de sampleo temporal
    tt = np.linspace(0, (Num_Muestras-1)*ts, Num_Muestras)
    
    Sinc_Generado = Vmax * np.sinc( 2 * np.pi * f0 * tt + Fase ) + Vmed

    # Devuelvo la grilla con todos los puntos que vamos a samplear y el seno que generamos con los parametros indicados
    
    return tt, Sinc_Generado

#%% Finalizacion de funcion 

#%% implementacion de funcion 

Vmax = 5
Vmed = 2
f0 = 100
Phi = 0    # Fase en radianes
N = 1000
fs = 1000

tt, sinc1 = Generador_Sinc( Vmax, Vmed, f0, Phi, N, fs )
    
#%% Ploteo de la funcion

plt.figure(1)
line_hdls = plt.plot( tt, sinc1 )