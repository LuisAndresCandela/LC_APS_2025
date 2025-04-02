# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 18:12:44 2025

@author: candell1
"""

#%% módulos y funciones a importar

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#import pdsmodulos as pds

#%% Datos de la simulación

fs = 1000 # frecuencia de muestreo (Hz)
N = 1000 # cantidad de muestras

# Al poner 1000 y 1000 logramos una resolucion espectral normalizada y que este 
# equiespaciada en una grilla de 1 hz 

# Datos del ADC 
B =  8 # bits
Vf = 10 # rango simétrico de +/- Vf Volts ; esto seria +- Vref
q = Vf/2**(B-1)  # paso de cuantización de q Volts 2 a la b-1

# datos del ruido (potencia de la señal normalizada, es decir 1 W)
# Esta potencia la obtenemos al asegurar varianza 1

pot_ruido_cuant = q**2 / 12 # Watts, esto va a ser la varianza de mi señal de ruido 
kn = 1 # escala de la potencia de ruido analógico
pot_ruido_analog = pot_ruido_cuant * kn #Ahora, con esto podemos escalar el ruido para aumentar o disminuir

ts = 1/fs  # tiempo de muestreo
df = fs/N # resolución espectral



#%% Experimento: 


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
f0 = 35
Phi = 0    
#N = 1000
#fs = 1000

tt,analog_sig = Generador_Senoidal(A, f0, Phi, N, fs )

# tt es la grilla temporal

varianza = np.var(analog_sig)

#%% Ahora genero mi señal de entrada al ADC

# Señales

# analog_sig ,  señal analógica sin ruido
# nn ,          señal de ruido

# señal analógica de entrada al ADC (con ruido analógico)
sr = analog_sig

# Observacion; como tengo una señal normalizada a la que luego de sumo un ruido. Puedo hacer esto
# ya que, este ruido obedece una distribucion normal con el sigma cuadrado

#%% Ya con la señal de ruido, vamos a cuantizarla

# señal cuantizada sin escala original adimensional ( Si veo el grafico acá la escala esta en divisiones )
srq_1 = np.round(sr/q)

# Señal cuantizada llevada a escala original ( Con esto la vuelvo a llevar a +- 3v )
srq = srq_1 * q

nq = sr - srq


#%%

plt.figure(1)
ft_SR = 1/N*np.fft.fft( sr )
ft_Srq = 1/N*np.fft.fft( srq )
ft_As = 1/N*np.fft.fft( analog_sig )
#ft_Nq = 1/N*np.fft.fft( nq )
#ft_Nn = 1/N*np.fft.fft( nn )

# grilla de sampleo frecuencial
ff = np.linspace(0, (N-1)*df, N)

bfrec = ff <= fs/2 
# Con esto eliminamos la mitad del vector de la FFT asi eliminamos las redundancias
# Este es un vector booleano que tiene True hasta N/2 valores y luego son todos False. 
# Ya que, la otra parte es simetrica

#Nnq_mean = np.mean(np.abs(ft_Nq)**2)
#nNn_mean = np.mean(np.abs(ft_Nn)**2)

plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_As[bfrec])**2), color='orange', ls='dotted', label='$ s $ (sig.)' )
#plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ), '--r', label= '$ \overline{n} = $' + '{:3.1f} dB'.format(10* np.log10(2* nNn_mean)) )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_SR[bfrec])**2), ':g', label='$ s_R = s + n $' )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Srq[bfrec])**2), lw=2, label='$ s_Q = Q_{B,V_F}\{s_R\}$' )
#plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--c', label='$ \overline{n_Q} = $' + '{:3.1f} dB'.format(10* np.log10(2* Nnq_mean)) )
#plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nn[bfrec])**2), ':r')
#plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nq[bfrec])**2), ':c')
plt.plot( np.array([ ff[bfrec][-1], ff[bfrec][-1] ]), plt.ylim(), ':k', label='BW', lw = 0.5  )

plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
axes_hdl = plt.gca()
axes_hdl.legend()

