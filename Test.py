# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 20:47:08 2025

@author: candell1
"""

#%% módulos y funciones a importar

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pdsmodulos as pds

#%% Datos de la simulación

fs = 1000 # frecuencia de muestreo (Hz)
N = 1000 # cantidad de muestras

# Al poner 1000 y 1000 logramos una resolucion espectral normalizada y que este 
# equiespaciada en una grilla de 1 hz 

# Datos del ADC 
B =  8 # bits
Vf = 0.001 # rango simétrico de +/- Vf Volts ; esto seria +- Vref
q = Vf / 2^7  # paso de cuantización de q Volts 2 a la b-1


# datos del ruido (potencia de la señal normalizada, es decir 1 W)
# Esta potencia la obtenemos al asegurar varianza 1

# pot_ruido_cuant = # Watts 
kn = 1. # escala de la potencia de ruido analógico
# pot_ruido_analog = pot_ruido_cuant * kn # 

ts = 1/fs  # tiempo de muestreo
df = fs/N # resolución espectral

#%% Genero mi señal senoidal

# Amplitud,             Amplitud maxima
# f0,               Frecuencia de la señal
# Fase,             Fase de la señal
# Num_Muestras      Cantidad de muestras tomadas por el ADC
# Fs,               Frecuencia de muestreo

def Generador_Senoidal ( Amplitud, f0, Fase, Num_Muestras, fs ):
    
    # Calculamos el tiempo de sampleo
    ts = 1/fs
    
    # Grilla de sampleo temporal
    tt = np.linspace(0, (Num_Muestras-1)*ts, Num_Muestras)
    
    Seno_Generado = Amplitud * np.sin( 2 * np.pi * f0 * tt + Fase )
    
    return tt, Seno_Generado

#%% Busco los valores para los cuales esta senoidal esta normalizada, es decir
# que la señal tiene una varianza = 1

# Como la varianza es una senoidal es x(t) = A sen ( wt + phi ), la varianza 
# va a estar dada por Var(x) = A^2 * 1/2 

A = np.sqrt(2)
Vmed = 0
f0 = 1
Phi = 0    
#N = 1000
#fs = 1000

tt,analog_sig = Generador_Senoidal(A, f0, Phi, N, fs )

# tt es la grilla temporal

varianza = np.var(analog_sig)

# Ya tengo mi señal normalizada en este punto


#%% Ahora, queremos sumar la señal de ruindo a esta senoidal

# Para poder general la señal de ruido, vamos a escalar el seno a un valor 
# que sea similar al indicado por el fabricante en el datasheet 50 nV / Raiz(hz)

Escala = 50*10^(-9) / np.sqrt(2)

nn = Escala * analog_sig

plt.figure(1)
line_hdls = plt.plot( tt, nn )

#%% Experimento: 
"""
   Se desea simular el efecto de la cuantización sobre una señal senoidal de 
   frecuencia 1 Hz. La señal "analógica" podría tener añadida una cantidad de 
   ruido gausiano e incorrelado.
   
   Se pide analizar el efecto del muestreo y cuantización sobre la señal 
   analógica. Para ello se proponen una serie de gráficas que tendrá que ayudar
   a construir para luego analizar los resultados.
   
"""

# np.random.normal
# np.random.uniform


# Señales

# analog_sig = # señal analógica sin ruido
# sr = # señal analógica de entrada al ADC (con ruido analógico)
# srq = # señal cuantizada

# nn =  # señal de ruido de analógico
# nq =  # señal de ruido de cuantización




#%% Visualización de resultados

# # cierro ventanas anteriores
# plt.close('all')

# ##################
# # Señal temporal
# ##################

# plt.figure(1)


# plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
# plt.xlabel('tiempo [segundos]')
# plt.ylabel('Amplitud [V]')
# axes_hdl = plt.gca()
# axes_hdl.legend()
# plt.show()

# #%% 

# plt.figure(2)
# ft_SR = 1/N*np.fft.fft( sr, axis = 0 )
# ft_Srq = 1/N*np.fft.fft( srq, axis = 0 )
# ft_As = 1/N*np.fft.fft( analog_sig, axis = 0)
# ft_Nq = 1/N*np.fft.fft( nq, axis = 0 )
# ft_Nn = 1/N*np.fft.fft( nn, axis = 0 )

# # grilla de sampleo frecuencial
# ff = np.linspace(0, (N-1)*df, N)

# bfrec = ff <= fs/2

# Nnq_mean = np.mean(np.abs(ft_Nq)**2)
# nNn_mean = np.mean(np.abs(ft_Nn)**2)

# plt.plot( ff_os[bfrec], 10* np.log10(2*np.abs(ft_As[bfrec])**2), color='orange', ls='dotted', label='$ s $ (sig.)' )
# plt.plot( np.array([ ff_os[bfrec][0], ff_os[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ), '--r', label= '$ \overline{n} = $' + '{:3.1f} dB'.format(10* np.log10(2* nNn_mean)) )
# plt.plot( ff_os[bfrec], 10* np.log10(2*np.mean(np.abs(ft_SR)**2, axis=1)[bfrec]), ':g', label='$ s_R = s + n $' )
# plt.plot( ff_os[bfrec], 10* np.log10(2*np.mean(np.abs(ft_Srq)**2, axis=1)[bfrec]), lw=2, label='$ s_Q = Q_{B,V_F}\{s_R\}$' )
# plt.plot( np.array([ ff_os[bfrec][0], ff_os[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--c', label='$ \overline{n_Q} = $' + '{:3.1f} dB'.format(10* np.log10(2* Nnq_mean)) )
# plt.plot( ff_os[bfrec], 10* np.log10(2*np.mean(np.abs(ft_Nn)**2, axis=1)[bfrec]), ':r')
# plt.plot( ff_os[bfrec], 10* np.log10(2*np.mean(np.abs(ft_Nq)**2, axis=1)[bfrec]), ':c')
# plt.plot( np.array([ ff[bfrec][-1], ff[bfrec][-1] ]), plt.ylim(), ':k', label='BW', lw = 0.5  )

# plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
# plt.ylabel('Densidad de Potencia [dB]')
# plt.xlabel('Frecuencia [Hz]')
# axes_hdl = plt.gca()
# axes_hdl.legend()

# #############
# # Histograma
# #############

# plt.figure()
# bins = 10
# # plt.hist(nq.flatten(), bins=2*bins)
# plt.hist(nqf.flatten()/(q/2), bins=2*bins)
# # plt.plot( np.array([-q/2, -q/2, q/2, q/2]), np.array([0, N*R/bins, N*R/bins, 0]), '--r' )
# plt.plot( np.array([-1/2, -1/2, 1/2, 1/2]), np.array([0, N*R/bins, N*R/bins, 0]), '--r' )
# plt.title( 'Ruido de cuantización para {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q))

# plt.xlabel('Pasos de cuantización (q) [V]')