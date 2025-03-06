# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 19:48:14 2025

@author: candell1
"""

#%% importacion de modulos a utilizar

# Una vez invocadas estas funciones, podremos utilizar los módulos a través 
# del identificador que indicamos luego de "as".

# Por ejemplo np.linspace() -> función linspace dentro e NumPy

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#%% Declaramos nuestra funcion para la señal 

  
# Datos generales de la simulación
fs = 1000.0     # frecuencia de muestreo (Hz)
N = 1000        # cantidad de muestras
fo = 100        # frecuencia de la señal

ts = 1/fs       # tiempo de muestreo
df = fs/N       # resolución espectral

# grilla de sampleo temporal
tt = np.linspace(0, (N-1)*ts, N).flatten()

# linspace( Inicio, Paso, Final)
    
# grilla de sampleo frecuencial
ff = np.linspace(0, (N-1)*df, N).flatten()


# Declaro funcion senoidal 
Test_seno = np.sin( 2 * np.pi * fo * tt  )

plt.figure(1)
line_hdls = plt.plot(tt, Test_seno)

