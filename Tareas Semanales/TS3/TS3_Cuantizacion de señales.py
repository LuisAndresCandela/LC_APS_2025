# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 11:41:45 2025

@author: candell1
"""

#%% 

""" 

TAREA SEMANAL 3 - Modelizado de un ADC

"""

#%% módulos y funciones a importar

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#%% Datos de la simulación

fs = 1000           # frecuencia de muestreo (Hz)
N = 1000            # cantidad de muestras

# Datos del ADC 
B =  4             # bits
Vf = 2             # rango simétrico de +/- Vf Volts ; esto seria +- Vref
q = Vf/2**(B-1)     # paso de cuantización de q Volts 2 a la b-1


# datos del ruido (potencia de la señal normalizada, es decir 1 W)
# Esta potencia la obtenemos al asegurar varianza 1

pot_ruido_cuant = q**2 / 12                 # Watts, esto va a ser la varianza de mi señal de ruido 
kn = 10                                     # Escala de la potencia de ruido analógico
pot_ruido_analog = pot_ruido_cuant * kn     # Ahora, con esto podemos escalar el ruido para aumentar o disminuir

ts = 1/fs           # tiempo de muestreo
df = fs/N           # resolución espectral

#%% Genero mi señal senoidal

# Amplitud,         Amplitud maxima
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

#%% Busco los valores para los cuales esta senoidal esta normalizada, es decir que la señal tiene una varianza = 1

# Como la varianza es una senoidal es x(t) = A sen ( wt + phi ), la varianza va a estar dada por Var(x) = A^2 * 1/2 

A = np.sqrt(2)
Vmed = 0
f0 = fs/N
Phi = 0    

tt,analog_sig = Generador_Senoidal( Amplitud= A, f0= f0, Fase= Phi, Num_Muestras= N, fs= fs )

# tt es la grilla temporal

varianza = np.var(analog_sig)

# Compruebo que la varianza es =1, entonces ya tengo mi señal normalizada en este punto

#%% Ahora, queremos sumar la señal de ruindo a esta senoidal

# Para poder general la señal de ruido, tenemos que tener una distribucion normal con un N(o,sigma)

Media = 0                   # Media
Sigma2 = pot_ruido_analog   # Varianza
SD_Sigma = np.sqrt(Sigma2)  # Desvio standar 

# Recuerdo que la pot del ruido es= q^2/12

# Genero señal de ruido
nn = np.random.normal(Media, SD_Sigma, N)

varianza_ruido = np.var(analog_sig)

# Compruebo que la varianza es =1, entonces ya tengo mi señal de ruido normalizada en este punto

#%% Ahora genero mi señal de entrada al ADC

# Señales

# analog_sig ,  señal analógica sin ruido
# nn ,          señal de ruido

# señal analógica de entrada al ADC (con ruido analógico)

sr = analog_sig + nn


# Observacion; como tengo una señal normalizada a la que luego de sumo un ruido. Puedo hacer esto
# ya que, este ruido obedece una distribucion normal con el sigma cuadrado

#%% Ya con la señal de ruido, vamos a cuantizarla

# Uso la funcion redondeo para aproximar el valor mas cercano de la funcion de ruido que estoy pasando 

# señal cuantizada sin escala original adimensional ( Si veo el grafico acá la escala esta en divisiones )
srq_1 = np.round(sr/q)

# Señal cuantizada llevada a escala original ( Con esto la vuelvo a llevar a +- 3v )
srq = srq_1 * q

# nn =  # señal de ruido de analógico
# nq =  # señal de ruido de cuantización

nq = sr - srq

#%% Visualización de resultados

# cierro ventanas anteriores
plt.close('all')

##################
# Señal temporal
##################

plt.figure(1)

plt.plot(tt, srq, lw=2, linestyle='', color='blue', marker='o', markersize=5, markerfacecolor='blue', markeredgecolor='blue', fillstyle='none', label='ADC out (diezmada)')
plt.plot(tt, sr, lw=1, color='black', marker='x', ls='dotted', label='$ s $ (analog)')

plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()


###########
# Espectro
###########

# En esta parte usamos la FFT para poder tranformar las señales usando la FFT
# Los numeros que vamos a tener en ft_SR son numeros complejos, ya que, estamos usando FFT

plt.figure(2)
ft_SR = 1/N*np.fft.fft( sr )
ft_Srq = 1/N*np.fft.fft( srq )
ft_As = 1/N*np.fft.fft( analog_sig )
ft_Nq = 1/N*np.fft.fft( nq )
ft_Nn = 1/N*np.fft.fft( nn )

# grilla de sampleo frecuencial
ff = np.linspace(0, (N-1)*df, N)

bfrec = ff <= fs/2 
# Con esto eliminamos la mitad del vector de la FFT asi eliminamos las redundancias
# Este es un vector booleano que tiene True hasta N/2 valores y luego son todos False. 
# Ya que, la otra parte es simetrica

Nnq_mean = np.mean(np.abs(ft_Nq)**2)
nNn_mean = np.mean(np.abs(ft_Nn)**2)

plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_As[bfrec])**2), color='orange', ls='dotted', label='$ s $ (sig.)' )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ), '--r', label= '$ \overline{n} = $' + '{:3.1f} dB'.format(10* np.log10(2* nNn_mean)) )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_SR[bfrec])**2), ':g', label='$ s_R = s + n $' )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Srq[bfrec])**2), lw=2, label='$ s_Q = Q_{B,V_F}\{s_R\}$' )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--c', label='$ \overline{n_Q} = $' + '{:3.1f} dB'.format(10* np.log10(2* Nnq_mean)) )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nn[bfrec])**2), ':r')
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nq[bfrec])**2), ':c')
plt.plot( np.array([ ff[bfrec][-1], ff[bfrec][-1] ]), plt.ylim(), ':k', label='BW', lw = 0.5  )

plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
axes_hdl = plt.gca()
axes_hdl.legend()

#############
# Histograma
#############

plt.figure(3)
bins = 10
plt.hist(nq.flatten()/(q), bins=bins)
plt.plot( np.array([-1/2, -1/2, 1/2, 1/2]), np.array([0, N/bins, N/bins, 0]), '--r' )
plt.title( 'Ruido de cuantización para {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q))

plt.xlabel('Pasos de cuantización (q) [V]')

