# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 18:13:11 2025

@author: candell1
"""

#%% Importacion de modulos

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
from scipy.signal import sosfreqz
from pytc2.sistemas_lineales import plot_plantilla

#%% Cargar archivo CSV con datos de respiración
csv_path = "slp01a_respiration.csv"
df = pd.read_csv(csv_path)

#%% Seleccionar primeros 300 segundos
df_300 = df[df["Time [s]"] <= 300]
time = df_300["Time [s]"]
resp_signal = df_300.iloc[:, 1].values  # Asumimos que la señal de respiración está en la segunda columna

#%% Definir parámetros
fs = 125
nyq = fs / 2

#%% Filtro IIR Chebyshev II
fpass = 0.3     # Hz
fstop = 0.6     # Hz
ripple = 0.5    # dB
attenuation = 40  # dB

cheby2_iir = sig.iirdesign(
    wp=fpass,
    ws=fstop,
    gpass=ripple,
    gstop=attenuation,
    ftype='cheby2',
    output='sos',
    fs=fs
)

# Evaluar respuesta en frecuencia
w, h = sosfreqz(cheby2_iir, worN=2048, fs=fs)

# Graficar respuesta
plt.figure()
plt.plot(w, 20 * np.log10(np.abs(h) + 1e-10), label='Chebyshev II')
plt.title("Respuesta en frecuencia - IIR Chebyshev II")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.ylim([-80, 5])
plt.grid(True)
plot_plantilla('lowpass', fpass=fpass, fstop=fstop, ripple=ripple, attenuation=attenuation, fs=fs)
plt.legend()
plt.show()

#%% Filtro FIR con ventanas (Kaiser)
cant_coef = 8001  # número de coeficientes (orden + 1)
freq_hz = [0.0, 0.4, 0.8, nyq]
gain =    [1.0, 1.0, 0.0, 0.0]

fpass = 0.4     # Hz
fstop = 0.85     # Hz

freq_norm = [f / nyq for f in freq_hz]

fir_window = sig.firwin2(
    numtaps=cant_coef,
    freq=freq_norm,
    gain=gain,
    window=('kaiser', 10)
)

# Evaluar respuesta
w_fir, h_fir = sig.freqz(fir_window, worN=2048, fs=fs)

# Graficar respuesta
plt.figure()
plt.plot(w_fir, 20 * np.log10(np.abs(h_fir) + 1e-10), label='FIR Ventana Kaiser')
plt.title("Respuesta en frecuencia - FIR (ventana Kaiser)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.ylim([-80, 5])
plt.grid(True)
plot_plantilla('lowpass', fpass=fpass, fstop=fstop, ripple=ripple, attenuation=attenuation, fs=fs)
plt.legend()
plt.show()

#%% Aplicar filtros a la señal de respiración
resp_cheby = sig.sosfiltfilt(cheby2_iir, resp_signal)
resp_fir = sig.filtfilt(fir_window, [1], resp_signal)

#%% Graficar resultados
plt.figure(figsize=(12, 6))
plt.plot(time, resp_signal, label='Original', alpha=0.6)
plt.plot(time, resp_cheby, label='Filtrado Chebyshev II', linewidth=1.5)
plt.plot(time, resp_fir, label='Filtrado FIR Ventana Kaiser', linewidth=1.5)
plt.title('Comparación de filtrado - Señal de Respiración')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Definir regiones de interés para señal de respiración
cant_muestras = len(resp_signal)

# Regiones (en segundos) convertidas a muestras
regs_ruido = (
    np.array([200, 220]) * fs,  # Con ruido
)
regs_sin_ruido = (
    np.array([10, 30]) * fs,    # Sin ruido
)

# Juntamos todo en una lista con etiquetas
regiones = [("Con Ruido", regs_ruido), ("Sin Ruido", regs_sin_ruido)]

#%% Gráficos por regiones
for tipo_region, lista_regiones in regiones:
    for i, reg in enumerate(lista_regiones):
        reg = np.array(reg, dtype=int)
        zoom_region = np.arange(np.max([0, reg[0]]), np.min([cant_muestras, reg[1]]), dtype='uint')

        # Tiempo correspondiente a la región
        t_zoom = time[zoom_region]

        plt.figure(figsize=(12, 6))
        plt.suptitle(f'{tipo_region} - Región {i+1}: muestras {reg[0]} a {reg[1]}', fontsize=14)

        plt.subplot(2, 1, 1)
        plt.plot(t_zoom, resp_signal[zoom_region], label='Original', alpha=0.6)
        plt.plot(t_zoom, resp_cheby[zoom_region], label='Filtrado Chebyshev II', linewidth=1.2)
        plt.title('Filtro IIR - Chebyshev II')
        plt.ylabel('Amplitud')
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(t_zoom, resp_signal[zoom_region], label='Original', alpha=0.6)
        plt.plot(t_zoom, resp_fir[zoom_region], label='Filtrado FIR - Ventana Kaiser', linewidth=1.2)
        plt.title('Filtro FIR - Ventana Kaiser')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud')
        plt.grid(True)
        plt.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()




