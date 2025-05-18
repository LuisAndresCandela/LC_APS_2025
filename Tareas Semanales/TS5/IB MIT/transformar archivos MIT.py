# -*- coding: utf-8 -*-
"""
Created on Sun May 18 10:12:59 2025

@author: candell1
"""
import wfdb
import pandas as pd
import numpy as np

# === 1. Configuración ===
record_name = 'slp01a'         # Registro de ejemplo
database = 'slpdb'
signal_keywords = ['resp', 'thoracic', 'abdominal']  # Palabras clave para detectar señales respiratorias

# === 2. Descargar registro (si no lo tenés) ===
wfdb.dl_database(database, dl_dir='slpdb_data', records=[record_name])

# === 3. Leer registro
record = wfdb.rdrecord(f'slpdb_data/{record_name}')
fs = record.fs  # Frecuencia de muestreo
signal_names = record.sig_name
signals = record.p_signal

# === 4. Filtrar señales de respiración
resp_indices = [i for i, name in enumerate(signal_names) if any(k in name.lower() for k in signal_keywords)]

if not resp_indices:
    raise ValueError("❌ No se encontraron señales respiratorias en este registro.")

# Extraer las señales de respiración
resp_signals = signals[:, resp_indices]
resp_names = [signal_names[i] for i in resp_indices]

# === 5. Crear DataFrame
time = np.arange(signals.shape[0]) / fs
df_resp = pd.DataFrame(resp_signals, columns=resp_names)
df_resp.insert(0, "Time [s]", time)

# === 6. Guardar CSV
output_file = f"{record_name}_respiration.csv"
df_resp.to_csv(output_file, index=False)

print(f"✅ Señales respiratorias guardadas en: {output_file}")
