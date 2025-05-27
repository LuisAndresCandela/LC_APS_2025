# -*- coding: utf-8 -*-
"""
Created on Thu May 22 18:26:17 2025

@author: candell1
"""
#%% Importo modulos

import sympy as sp
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from pytc2.sistemas_lineales import plot_plantilla

#%% 

"""

Trabajo con filtros IIR

Diseño de filtros con plantilass ( IIR Design )

"""

#%% Vamos a implementar un filtro pasabanda para un ECG

# Tipos de filtro que vamos a usar 

aprox_name = 'butter'
# aprox_name = 'cheby1'
# aprox_name = 'cheby2'
# aprox_name = 'ellip'

# Parametros de la plantilla del filtro 

fs = 1000 # Hz
fpass = np.array( [1, 35.0] )       # Lo definimos segun la plantilla. de 1 hz a 35 hz la banda de paso 
ripple = 1                          # Db -- Alpha max
fstop = np.array( [0.1, 50.] )      # y la banda de stop comienza en 50 hz
attenuation = 40                    # dB -- Alplha min

#%% Filtro 

# Ver como llenar

mi_sos = sig.iirdesign(
    wp=fpass,
    ws=fstop,
    gpass=ripple,
    gstop=attenuation,
    ftype=aprox_name,
    output='sos',
    fs=fs
)


#%% ANalisis de filtro

fig = plt.figure(1)
plt.cla()

npoints = 1000

w, hh = sig.sosfreqz(mi_sos, worN=npoints)
plt.plot(w/np.pi, 20*np.log10(np.abs(hh)), label='Mi_sos')

plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')

ax = plt.gca()
ax.set_xlim([0, 1])
ax.set_ylim([-60, 1])

plot_plantilla(filter_type = aprox_name , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = attenuation, fs = fs)
plt.legend()
plt.show()
