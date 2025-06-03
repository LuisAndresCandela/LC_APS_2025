# -*- coding: utf-8 -*-
"""
Created on Thu May 22 19:08:41 2025

@author: candell1
"""

#%% 

"""

Ahora queremos implementar el filtro antes diseñado con iir design para filtrar la 
señal del ECG que habiamos trabajado anteriormente

Vamos a usar sosfiltfilt para filtrar la ECG una vez que tenemos un filtro diseñado 
y bien diseñado 

"""

#%% Importado de modulos

import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.io.wavfile import write
from scipy import signal
from scipy.signal import welch
from scipy.signal.windows import hamming, hann, blackman, kaiser, flattop, blackmanharris
from scipy.fft import fft, fftshift
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import os
from scipy.fft import fft
from scipy.signal import fftconvolve

def vertical_flaten(a):

    return a.reshape(a.shape[0],1)

#%% Genero mi funcion para implementar Blackmann-Tuckey

from scipy.signal import fftconvolve

def blackman_tukey_psd(x, window_len=None, window_func=blackman, fs=1.0):
    N = len(x)
    if window_len is None:
        window_len = N // 4

    # Autocorrelación estimada (eficiente)
    rxx = fftconvolve(x, x[::-1], mode='full') / N
    mid = len(rxx) // 2
    rxx = rxx[mid:mid + window_len]

    ventana = window_func(window_len)
    rxx_win = rxx * ventana

    Pxx = np.abs(fft(rxx_win, n=2 * window_len))
    f_Pxx = np.fft.fftfreq(2 * window_len, d=1/fs)
    mask = f_Pxx >= 0
    return f_Pxx[mask], Pxx[mask]

#%% Lectura del ECG

fs_ecg = 1000 # Hz

# para listar las variables que hay en el archivo
#io.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_one_lead = np.squeeze(mat_struct['ecg_lead']).astype(float)
ecg_one_lead = (ecg_one_lead - np.mean(ecg_one_lead)) / np.std(ecg_one_lead)

N = len(ecg_one_lead)

hb_1 = vertical_flaten(mat_struct['heartbeat_pattern1'])
hb_2 = vertical_flaten(mat_struct['heartbeat_pattern2'])


#%% Normalizar la señal
ecg_one_lead = (ecg_one_lead - np.mean(ecg_one_lead)) / np.std(ecg_one_lead)

ecg_segment = ecg_one_lead #ecg_one_lead[5000:24000] # Me quedo solo con un segmento de la señal del ECG

#%% Calculo PSD con Blackman-Tukey
Fs_ECG_BT, Signal_ECG_Pxx = blackman_tukey_psd(x=ecg_segment,
                                               window_len=None,
                                               window_func=blackman,
                                               fs=fs_ecg)

#%% Calculo energía acumulada y anchos de banda
psd_norm = Signal_ECG_Pxx / np.sum(Signal_ECG_Pxx)
Acumulada = np.cumsum(psd_norm)

bw_95 = Fs_ECG_BT[np.where(Acumulada >= 0.95)[0][0]]
bw_98 = Fs_ECG_BT[np.where(Acumulada >= 0.98)[0][0]]

print(f"Ancho de banda para 95%: {bw_95:.2f} Hz")
print(f"Ancho de banda para 98%: {bw_98:.2f} Hz")

#%% Gráficos
fig, axs = plt.subplots(3, 1, figsize=(12, 10))

# Tiempo para el segmento
#t_segment = np.linspace(5000/fs_ecg, 24000/fs_ecg, len(ecg_segment))

t_segment = np.linspace(0, len(ecg_segment) / fs_ecg, len(ecg_segment))

# Señal original (solo segmento)
axs[0].plot(t_segment, ecg_segment)
axs[0].set_title("Señal ECG ")
axs[0].set_xlabel("Tiempo [s]")
axs[0].set_ylabel("Amplitud")
axs[0].grid(True)

# PSD del segmento
axs[1].plot(Fs_ECG_BT, Signal_ECG_Pxx)
axs[1].set_title("Densidad Espectral de Potencia (PSD) - Blackman-Tukey (Segmento)")
axs[1].set_xlabel("Frecuencia [Hz]")
axs[1].set_ylabel("Potencia")
axs[1].grid(True)

# PSD con anchos de banda del segmento
axs[2].plot(Fs_ECG_BT, Signal_ECG_Pxx)
axs[2].axvline(bw_95, color='r', linestyle='--', label=f'BW 95%: {bw_95:.2f} Hz')
axs[2].axvline(bw_98, color='g', linestyle='--', label=f'BW 98%: {bw_98:.2f} Hz')
axs[2].set_title("PSD con Ancho de Banda 95% y 98% (Segmento)")
axs[2].set_xlabel("Frecuencia [Hz]")
axs[2].set_ylabel("Potencia")
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()

#%% Filtrado de ECG

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

#%% Diseño de filtro con iirdesing

mi_sos = sig.iirdesign(
    wp=fpass,
    ws=fstop,
    gpass=ripple,
    gstop=attenuation,
    ftype=aprox_name,
    output='sos',
    fs=fs
)

#%% Funcion para plantilla de filtros

def plot_plantilla(filter_type, fpass, ripple, fstop, attenuation, fs):
    nyq = fs / 2  # Frecuencia de Nyquist
    
    # Zonas de paso y stop normalizadas
    wp_norm = fpass / nyq
    ws_norm = fstop / nyq

    ax = plt.gca()

    # Banda de paso permitida (amplitud dentro de -ripple dB)
    ax.fill_between(wp_norm, -ripple, 0, color='green', alpha=0.2, label='Banda de paso')

    # Banda de stop (amplitud por debajo de -attenuation dB)
    ax.fill_betweenx([-attenuation, -60], 0, ws_norm[0], color='red', alpha=0.2, label='Stop bajo')
    ax.fill_betweenx([-attenuation, -60], ws_norm[1], 1.0, color='red', alpha=0.2, label='Stop alto')

    # Líneas guías horizontales para ripple y attenuation
    ax.axhline(-ripple, color='gray', linestyle='--', linewidth=1)
    ax.axhline(-attenuation, color='gray', linestyle='--', linewidth=1)


#%% Analisis de filtro para ver que cumpla con las condiciones propuestas

fig = plt.figure(2)
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


#%% Aplicación del filtro a la señal ECG (segmento)
ecg_filtrada = sig.sosfiltfilt(mi_sos, ecg_segment)

#%% Gráfico: señal original vs filtrada

# Crear 3 subplots (uno adicional para el zoom)
#fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=False)

# Señal original
axs[0].plot(t_segment, ecg_segment, color='blue')
axs[0].set_title("ECG Original")
axs[0].set_ylabel("Amplitud")
axs[0].grid(True)

# Señal filtrada
axs[1].plot(t_segment, ecg_filtrada, color='green')
axs[1].set_title("ECG Filtrada con iirdesign + sosfiltfilt")
axs[1].set_ylabel("Amplitud")
axs[1].grid(True)

# --- ZOOM ---
# Definir índice de zoom
idx_start = 1000
idx_end = 5000

# Tiempo para la zona de zoom
t_zoom = t_segment[idx_start:idx_end]

# Señales en la zona de zoom
ecg_orig_zoom = ecg_segment[idx_start:idx_end]
ecg_filt_zoom = ecg_filtrada[idx_start:idx_end]

# Graficar ambas señales en la zona de zoom
axs[2].plot(t_zoom, ecg_orig_zoom, label="Original", color='blue', alpha=0.7)
axs[2].plot(t_zoom, ecg_filt_zoom, label="Filtrada", color='green', alpha=0.7)
axs[2].set_title("Zoom: ECG Original vs Filtrada (índices 1000-5000)")
axs[2].set_xlabel("Tiempo [s]")
axs[2].set_ylabel("Amplitud")
axs[2].grid(True)
axs[2].legend()

plt.tight_layout()
plt.show()


"""

Podemos ver como el filtro estabiliza a la señal del ECG, elimina los movimientos
generados por las bajas frecuencias y a su vez los movimientos pequeños de las altas frec

Estos nos dice que el filtrado funciona bien tanto para las bajas como las altas, tal 
y como lo habiamos diseñado 

Si bien, ahora comprobamos que el filtrado funciona, como podemos ver que la banda de paso 
funciona del filtro este funcionando correctamente ?

Tenemos que tomar una referencia, para lo cual en una ECG tenemos uqe tomar la señal al principio de todo 
que va a ser cuando mas limpia va a estar. Entonces, en este punto podemos comparar las 
amplitudes, morfoligias, etc. POr lo que, si acá la señal filtrada se parece a la original entonces
la banda de paso de nuestro filtro estaria funcionando correctamente

Un filtro tiene que ser inocuo en su banda de paso: Tiene que pasar sin afectar la
morfologia de la señal 


OBS PARA TS7 

Modificar la plantilla para que sea mas inocuo el filtro, y por lo que se mantenga mas la 
mofologia de la señal

Usar filt filt y comparar con filt.

Recordar que el filt filt lo hace 2 veces, por lo que, duplicamos la plantilla. En cambio
con el filt lo hacemos una vez 

Con el filt filt mitigamos distorsion de fase 
con el filt no 



"""


