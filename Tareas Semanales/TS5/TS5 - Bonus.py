# -*- coding: utf-8 -*-
"""
Created on Sun May 18 11:42:07 2025

@author: candell1
"""
#%% Carga de nmodulos

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.integrate import cumulative_trapezoid

#%% Cargo archivo CSV

csv_path = "slp01a_respiration.csv"  # Asegurate de tenerlo en el mismo directorio
df = pd.read_csv(csv_path)

#%% Me quedo con los primeros 300 segundos de datos

df_300 = df[df["Time [s]"] <= 300]
time = df_300["Time [s]"]
signal_columns = df_300.columns[1:]  # Elijo todas las columnas menos la primera que contiene los indices de tiempo


fs = 125  # Frecuencia de muestreo en Hz - Tome una fs acorde a como es la respiracion

#%% Procesamiento de la señal

for col in signal_columns:
    raw_signal = df_300[col].values

    # --- Normalización ---
    normalized_signal = (raw_signal - np.mean(raw_signal)) / np.std(raw_signal)

    # --- Cálculo de PSD usando Welch ---
    f, psd = welch(normalized_signal, fs=fs, nperseg=1024)

    # --- Calcular bandas de frecuencia (95% y 98%) ---
    cumulative_psd = cumulative_trapezoid(psd, f, initial=0)
    total_power = cumulative_psd[-1]

    f_95 = f[np.where(cumulative_psd >= 0.95 * total_power)[0][0]]
    f_98 = f[np.where(cumulative_psd >= 0.98 * total_power)[0][0]]

    print(f"Señal: {col}")
    print(f"  BW 95%: {f_95:.3f} Hz")
    print(f"  BW 98%: {f_98:.3f} Hz")

    # === 5. Graficar ===
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle(f"Análisis de señal de respiración: {col}", fontsize=14)

    # Señal original
    axs[0].plot(time, raw_signal, color='steelblue')
    axs[0].set_title("Señal Original (sin normalizar)")
    axs[0].set_xlabel("Tiempo [s]")
    axs[0].set_ylabel("Amplitud")
    axs[0].grid(True)

    # PSD
    axs[1].semilogy(f, psd, color='darkgreen')
    axs[1].set_title("Densidad espectral de potencia (PSD)")
    axs[1].set_xlabel("Frecuencia [Hz]")
    axs[1].set_ylabel("PSD [1/Hz]")
    axs[1].grid(True)

    # PSD con BW
    axs[2].semilogy(f, psd, color='purple')
    axs[2].axvline(f_95, color='orange', linestyle='--', label='95% BW')
    axs[2].axvline(f_98, color='red', linestyle='--', label='98% BW')
    axs[2].set_title("PSD con límites de BW (95% y 98%)")
    axs[2].set_xlabel("Frecuencia [Hz]")
    axs[2].set_ylabel("PSD [1/Hz]")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

