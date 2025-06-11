# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 18:51:38 2025

@author: candell1
"""

#%% Importado de módulos

import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.interpolate import CubicSpline
from pytc2.sistemas_lineales import bodePlot, plot_plantilla
from scipy.signal import sosfreqz



def vertical_flaten(a):
    return a.reshape(a.shape[0], 1)

#%% Lectura del ECG

fs = 1000  # Hz

mat_struct = sio.loadmat('./ecg.mat')
ecg_one_lead = np.squeeze(mat_struct['ecg_lead']).astype(float)

# Normalización
ecg_one_lead = (ecg_one_lead - np.mean(ecg_one_lead)) / np.std(ecg_one_lead)

# Tiempo total
t_ecg = np.arange(len(ecg_one_lead)) / fs

nyq_frec = fs / 2

#%% Diseño de filtro IIR - Plantilla de filtro - Butter

aprox_name = 'butter'
# aprox_name = 'cheby1'
# aprox_name = 'cheby2'
# aprox_name = 'ellip'

# Parametros de la plantilla del filtro 


fpass = np.array([0.5, 30])      # Banda de paso
fstop = np.array([0.2, 50])      # Banda de detención
ripple = 0.5                     # Rizado en banda de paso (dB)
attenuation = 40                # Atenuación en banda de detención (dB)

#%% Diseño de filtro con iirdesing

Butter_iir = sig.iirdesign(
    wp=     fpass,
    ws=     fstop,
    gpass=  ripple,
    gstop=  attenuation,
    ftype=  aprox_name,
    output='sos',
    fs=     fs
)

f_low = np.linspace(0.01, 0.4, 300)
f_fine = np.linspace(0.4, 0.6, 500)
f_high = np.linspace(0.6, nyq_frec, 700)
f_total = np.concatenate((f_low, f_fine, f_high))

w_rad = f_total / nyq_frec * np.pi
w, hh = sosfreqz(Butter_iir, worN=w_rad)

plt.figure()
plt.plot(w / np.pi * nyq_frec, 20 * np.log10(np.abs(hh) + 1e-15),
         label=f'Respuesta del filtro ({aprox_name})')
plt.title(f'Plantilla del filtro digital para ECG ({aprox_name})')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')
plt.grid(True)
plt.ylim([-80, 5])

fpass = np.array([0.5, 29]) 

plot_plantilla(
    filter_type='bandpass',
    fpass=fpass,
    ripple=ripple,
    fstop=fstop,
    attenuation=attenuation,
    fs=fs
)

plt.legend()
plt.show()

#%% Diseño de filtro IIR - Plantilla de filtro - Cheby

# aprox_name = 'butter'
# aprox_name = 'cheby1'
aprox_name = 'cheby2'
# aprox_name = 'ellip'

# Parametros de la plantilla del filtro 

fpass = np.array([0.5, 30])      # Banda de paso
fstop = np.array([0.2, 50])      # Banda de detención
ripple = 0.5                     # Rizado en banda de paso (dB)
attenuation = 40                # Atenuación en banda de detención (dB)

#%% Diseño de filtro con iirdesing

Cheby_iir = sig.iirdesign(
    wp=     fpass,
    ws=     fstop,
    gpass=  ripple,
    gstop=  attenuation,
    ftype=  aprox_name,
    output='sos',
    fs=     fs
)

f_low = np.linspace(0.01, 0.4, 300)
f_fine = np.linspace(0.4, 0.6, 500)
f_high = np.linspace(0.6, nyq_frec, 700)
f_total = np.concatenate((f_low, f_fine, f_high))

w_rad = f_total / nyq_frec * np.pi
w, hh = sosfreqz(Cheby_iir, worN=w_rad)

plt.figure()
plt.plot(w / np.pi * nyq_frec, 20 * np.log10(np.abs(hh) + 1e-15),
         label=f'Respuesta del filtro ({aprox_name})')
plt.title(f'Plantilla del filtro digital para ECG ({aprox_name})')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')
plt.grid(True)
plt.ylim([-80, 5])

fpass = np.array([0.5, 29]) 

plot_plantilla(
    filter_type='bandpass',
    fpass=fpass,
    ripple=ripple,
    fstop=fstop,
    attenuation=attenuation,
    fs=fs
)

plt.legend()
plt.show()

#%% Diseño de filtros con FIR - Ventanas



# Parametros de la plantilla del filtro 

fpass = np.array([1.3, 35])      # Banda de paso
fstop = np.array([0.1, 50])      # Banda de detención
ripple = 0.5                     # Rizado en banda de paso (dB)
attenuation = 40                # Atenuación en banda de detención (dB)

#%% Diseño de filtro con firwin2

cant_coef = 5001    # cantidad de coeficientes (orden + 1), ideal impar

nyq = fs / 2        # frecuencia de Nyquist

# Definimos los puntos de frecuencia y ganancia
# En Hz: queremos un filtro pasabanda entre 1 y 35 Hz

freq_hz = [0.0, 0.1, 1, 35.0, 50.0, nyq]   # en Hz
gain =    [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]     # ganancia deseada en cada punto

# Normalizar las frecuencias
freq_norm = [f / nyq for f in freq_hz]

# Diseñar el filtro con firwin2
Windows_fir = sig.firwin2(
    numtaps=        cant_coef,
    freq=           freq_norm,
    gain=           gain,
    window=         ('kaiser', 10) )

aprox_name = 'FIR-Kaiser' 

f_low = np.linspace(0.01, 0.4, 300)
f_fine = np.linspace(0.4, 0.6, 500)
f_high = np.linspace(0.6, nyq_frec, 700)
f_total = np.concatenate((f_low, f_fine, f_high))

w_rad = f_total / nyq_frec * np.pi
w, hh = sig.freqz(Windows_fir, worN=w_rad)

plt.figure()
plt.plot(w / np.pi * nyq_frec, 20 * np.log10(np.abs(hh) + 1e-15),
         label=f'Respuesta del filtro ({aprox_name})')
plt.title(f'Plantilla del filtro digital para ECG ({aprox_name})')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')
plt.grid(True)
plt.ylim([-80, 5])

plot_plantilla(
    filter_type='bandpass',
    fpass=fpass,
    ripple=ripple,
    fstop=fstop,
    attenuation=attenuation,
    fs=fs
)

plt.legend()
plt.show()

#%% Diseño de FIR con cuadrados minimos

from pytc2.filtros_digitales import fir_design_ls

# Parámetros

fstop = np.array([0.5, 50])    # Banda de detención
fpass = np.array([2, 35])      # Banda de paso
ripple = 2                     # dB (banda de paso)
attenuation = 30              # dB (banda de detención)

# Frecuencias normalizadas (por Nyquist = fs/2)
fn = fs / 2
Be = [
    0.0, fstop[0]/fn,           # detención baja
    fpass[0]/fn, fpass[1]/fn,   # paso
    fstop[1]/fn, 1.0            # detención alta
]

# Respuesta deseada en cada banda
D = [0, 0, 1, 1, 0, 0]

# Peso relativo (convertido de dB aproximado)
W = [10**(attenuation/20), 1, 10**(attenuation/20)]  # enfatiza la banda de paso

# Estimamos orden (puedes refinar esto)
N = 250  # orden del filtro (ajustable)

# Diseño del filtro
lsq_fir = fir_design_ls(order=N, band_edges=Be, desired=D, weight=W, filter_type='m', grid_density=16)

# Evaluamos FFT
fft_sz = 4096
H = np.fft.fft(lsq_fir, fft_sz)
frecuencias = np.linspace(0, fn, fft_sz//2)

# Graficar
plt.figure(figsize=(10, 5))
plt.plot(frecuencias, 20*np.log10(np.abs(H[:fft_sz//2]) + 1e-8), label='Filtro FIR LS')
plt.title("Respuesta en Frecuencia del Filtro FIR Pasabanda")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.ylim([-80, 5])
plt.grid(True)
plt.legend()
plt.tight_layout()


# Plantilla
plot_plantilla(filter_type='bandpass',
               fpass=fpass,
               ripple=ripple,
               fstop=fstop,
               attenuation=attenuation,
               fs=fs)

plt.title(f"Filtro FIR Pasa Banda - Orden {N}")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

#%%

# PARTO EL DISEÑO PARA MEJORAR LA PERFORMANCE

import numpy as np
import matplotlib.pyplot as plt
from pytc2.filtros_digitales import fir_design_ls

# Parámetros
fs = 1000  # Asegúrate de definir fs apropiadamente
fstop = np.array([0.5, 50])
fpass = np.array([1, 35])
ripple = 2
attenuation = 40
fn = fs / 2  # Frecuencia de Nyquist

N = 750  # orden del filtro
fft_sz = 4096

# ----------------------------
# Filtro Pasa Altos (corte en fpass[0])
# ----------------------------
Be_hp = [
    0.0, fstop[0]/fn,     # detención
    fpass[0]/fn, 1.0      # paso
]
D_hp = [0, 0, 1, 1]
W_hp = [10**(attenuation/20), 1]

fir_hp = fir_design_ls(order=N, band_edges=Be_hp, desired=D_hp, weight=W_hp, filter_type='m', grid_density=16)

# ----------------------------
# Filtro Pasa Bajos (corte en fpass[1])
# ----------------------------
Be_lp = [
    0.0, fpass[1]/fn,      # paso
    fstop[1]/fn, 1.0       # detención
]
D_lp = [1, 1, 0, 0]
W_lp = [1, 10**(attenuation/20)]

fir_lp = fir_design_ls(order=N, band_edges=Be_lp, desired=D_lp, weight=W_lp, filter_type='m', grid_density=16)

# ----------------------------
# Convolución para obtener el filtro Pasabanda
# ----------------------------
fir_bp = np.convolve(fir_hp, fir_lp)

# ----------------------------
# Evaluación en frecuencia
# ----------------------------
H = np.fft.fft(fir_bp, fft_sz)
frecuencias = np.linspace(0, fn, fft_sz//2)

# ----------------------------
# Graficar
# ----------------------------
plt.figure(figsize=(10, 5))
plt.plot(frecuencias, 20*np.log10(np.abs(H[:fft_sz//2]) + 1e-8), label='Filtro FIR Pasabanda (HP * LP)')
plt.title("Respuesta en Frecuencia del Filtro FIR Pasabanda")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.ylim([-80, 5])
plt.grid(True)
plt.legend()
plt.tight_layout()

# Plantilla de referencia
plot_plantilla(filter_type='bandpass',
               fpass=fpass,
               ripple=ripple,
               fstop=fstop,
               attenuation=attenuation,
               fs=fs)

plt.title(f"Filtro FIR Pasabanda (HP * LP) - Orden efectivo {len(fir_bp)-1}")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.grid()
plt.tight_layout()
plt.show()


#%% Aplicación de los 4 filtros y graficado

# Aplicación de los filtros
ecg_butter = sig.sosfiltfilt(Butter_iir, ecg_one_lead)
ecg_cheby  = sig.sosfiltfilt(Cheby_iir, ecg_one_lead)
ecg_winfir = sig.filtfilt(Windows_fir, [1], ecg_one_lead)
ecg_lsqfir = sig.filtfilt(lsq_fir, [1], ecg_one_lead)

# Seleccionamos una ventana de tiempo para visualizar
seg_inicio = 2  # segundo inicial
seg_dur = 200     # duración de la ventana
idx_inicio = int(seg_inicio * fs)
idx_fin = int((seg_inicio + seg_dur) * fs)

t_zoom = t_ecg[idx_inicio:idx_fin]
ecg_orig_zoom = ecg_one_lead[idx_inicio:idx_fin]

# Subplots
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(t_zoom, ecg_orig_zoom, label='Original', alpha=0.6)
plt.plot(t_zoom, ecg_butter[idx_inicio:idx_fin], label='Butterworth', linewidth=1.2)
plt.title('Filtro IIR - Butterworth')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(t_zoom, ecg_orig_zoom, label='Original', alpha=0.6)
plt.plot(t_zoom, ecg_cheby[idx_inicio:idx_fin], label='Chebyshev II', linewidth=1.2)
plt.title('Filtro IIR - Chebyshev II')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(t_zoom, ecg_orig_zoom, label='Original', alpha=0.6)
plt.plot(t_zoom, ecg_winfir[idx_inicio:idx_fin], label='FIR - Ventana Kaiser', linewidth=1.2)
plt.title('Filtro FIR - Ventana Kaiser')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(t_zoom, ecg_orig_zoom, label='Original', alpha=0.6)
plt.plot(t_zoom, ecg_lsqfir[idx_inicio:idx_fin], label='FIR - Cuadrados Mínimos', linewidth=1.2)
plt.title('Filtro FIR - Mínimos Cuadrados')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

#%% Analizamos regiones de interes

cant_muestras = len(ecg_one_lead)

# Regiones de interés con ruido (en muestras)
regs_ruido = (
    [4000, 5500],
    [10_000, 11_000],
)

# Regiones sin ruido (en minutos convertidos a muestras)
regs_sin_ruido = (
    np.array([5, 5.2]) * 60 * fs,
    np.array([12, 12.4]) * 60 * fs,
    np.array([15, 15.2]) * 60 * fs,
)

# Juntamos todo en una lista con etiquetas
regiones = [("Con Ruido", regs_ruido), ("Sin Ruido", regs_sin_ruido)]

for tipo_region, lista_regiones in regiones:
    for i, reg in enumerate(lista_regiones):
        # Convertimos a enteros por seguridad
        reg = np.array(reg, dtype=int)
        # Limitamos a rango válido
        zoom_region = np.arange(np.max([0, reg[0]]), np.min([cant_muestras, reg[1]]), dtype='uint')

        # Tiempo correspondiente a la región
        t_zoom = t_ecg[zoom_region]

        # Subplots
        plt.figure(figsize=(12, 10))
        plt.suptitle(f'{tipo_region} - Región {i+1}: muestras {reg[0]} a {reg[1]}', fontsize=14)

        plt.subplot(4, 1, 1)
        plt.plot(t_zoom, ecg_one_lead[zoom_region], label='Original', alpha=0.6)
        plt.plot(t_zoom, ecg_butter[zoom_region], label='Butterworth', linewidth=1.2)
        plt.title('Filtro IIR - Butterworth')
        plt.ylabel('Amplitud')
        plt.grid(True)
        plt.legend()

        plt.subplot(4, 1, 2)
        plt.plot(t_zoom, ecg_one_lead[zoom_region], label='Original', alpha=0.6)
        plt.plot(t_zoom, ecg_cheby[zoom_region], label='Chebyshev II', linewidth=1.2)
        plt.title('Filtro IIR - Chebyshev II')
        plt.ylabel('Amplitud')
        plt.grid(True)
        plt.legend()

        plt.subplot(4, 1, 3)
        plt.plot(t_zoom, ecg_one_lead[zoom_region], label='Original', alpha=0.6)
        plt.plot(t_zoom, ecg_winfir[zoom_region], label='FIR - Ventana Kaiser', linewidth=1.2)
        plt.title('Filtro FIR - Ventana Kaiser')
        plt.ylabel('Amplitud')
        plt.grid(True)
        plt.legend()

        plt.subplot(4, 1, 4)
        plt.plot(t_zoom, ecg_one_lead[zoom_region], label='Original', alpha=0.6)
        plt.plot(t_zoom, ecg_lsqfir[zoom_region], label='FIR - Mínimos Cuadrados', linewidth=1.2)
        plt.title('Filtro FIR - Mínimos Cuadrados')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud')
        plt.grid(True)
        plt.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
