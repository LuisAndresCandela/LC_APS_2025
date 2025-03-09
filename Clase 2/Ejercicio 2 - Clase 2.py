# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 21:11:40 2025

@author: candell1
"""

#%% Enunciado del ejercicio 

# Una forma de onda de una expresion matematica, va a ser muestreada periodicamente y 
# reproducida a partir de estas muestras. Halle el intervalo de tiempo maximo permitido 
# entre muestras para samplear estas señales

# 1) Cual es el ts ? 

# 2) Cuantas muestras tengo en 1 segundo de muestreo ?

# Señal: 20 + 20 * sin ( 500t + 30°)

#%% Ejercicio 

# 1) fs = 500 / Pi -> ts = 1 / fs // Porque la F de corte es Fs / 2

# Es importante entender que a esto no afecta la funcion lineal 20 y la fase del seno

# 2)  1 seg / tsampling (ts) 