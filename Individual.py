import h5py
import numpy as np
import matplotlib.pyplot as plt
from pytc2.sistemas_lineales import plot_plantilla
import scipy.signal as sig

# === Filtrado de EEG ===
fs = 250
fpass = np.array([0.5, 45])
fstop = np.array([0.1, 50])
ripple = 1
attenuation = 40

sos_eeg = sig.iirdesign(
    wp=fpass,
    ws=fstop,
    gpass=ripple,
    gstop=attenuation,
    ftype='butter',
    output='sos',
    fs=fs
)

w_Hz_raw = np.append(np.logspace(-2, 0.8, 250), np.logspace(0.9, 1.75, 250))
w_Hz_raw = np.append(w_Hz_raw, np.linspace(57, fs / 2, 500))
w_rad = w_Hz_raw / (fs / 2) * np.pi
w_Hz = w_rad / np.pi * (fs / 2)

_, h = sig.sosfreqz(sos_eeg, worN=w_rad)
h_db = 20 * np.log10(np.maximum(np.abs(h), 1e-10))

fig = plt.figure(figsize=(25, 6))
plt.subplot(1, 2, 1)
plt.title('Plantilla y Filtro - EEG')
plt.grid(which='both', axis='both')
plt.xlim([0, 60])
plt.ylim([-50, 5])
plot_plantilla(filter_type='bandpass', fpass=fpass, ripple=ripple, fstop=fstop, attenuation=attenuation, fs=fs)
plt.plot(w_Hz, h_db, label='Butterworth (EEG)', color='lightseagreen', linewidth=1)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.legend()

# === Filtrado de EOG ===
fs_eog = 250
fpass = np.array([0.3, 45])
fstop = np.array([0.1, 50])
ripple = 1
attenuation = 40

sos_eog = sig.iirdesign(
    wp=fpass,
    ws=fstop,
    gpass=ripple,
    gstop=attenuation,
    ftype='butter',
    output='sos',
    fs=fs_eog
)

w_Hz_raw = np.concatenate([
    np.linspace(0.01, 3, 400),
    np.linspace(3, 40, 300),
    np.linspace(40, 50, 300),
    np.linspace(55, fs_eog / 2, 200)
])
w_rad = w_Hz_raw / (fs_eog / 2) * np.pi
w_Hz = w_rad / np.pi * (fs_eog / 2)

_, h = sig.sosfreqz(sos_eog, worN=w_rad)
h_db = 20 * np.log10(np.maximum(np.abs(h), 1e-10))

plt.subplot(1, 2, 2)
plt.title('Plantilla y Filtro - EOG')
plt.grid(which='both', axis='both')
plt.xlim([0, 60])
plt.ylim([-50, 5])
plot_plantilla(filter_type='bandpass', fpass=fpass, ripple=ripple, fstop=fstop, attenuation=attenuation, fs=fs_eog)
plt.plot(w_Hz, h_db, label='Butterworth (EOG)', color='lightseagreen', linewidth=1)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.legend()
plt.tight_layout()
plt.show()

#%%
# === Aplicar filtro a EEG ===
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
archivo_h5 = os.path.join(script_dir, 'DOD-H', '7d778801-88e7-5086-ad1d-70f31a371876.h5')


if not os.path.exists(archivo_h5):
    raise FileNotFoundError(f"No se encontró el archivo: {archivo_h5}")
print("Ruta seleccionada:", archivo_h5)
ruta = archivo_h5

#%%
with h5py.File(ruta, 'r') as f:
    grupo_eeg = f['signals']['eeg']
    nombres_eeg = list(grupo_eeg.keys())
    print("Canales EEG disponibles:")
    for nombre in nombres_eeg:
        canal = grupo_eeg[nombre]
        print(f" - {nombre}: {len(canal)} muestras")
    fs = grupo_eeg.attrs['fs']
    print(f"\nFrecuencia de muestreo (fs): {fs} Hz")
    print(f"Cantidad de canales EEG: {len(nombres_eeg)}")
    print("\nAtributos del grupo EEG:")
    for clave, valor in grupo_eeg.attrs.items():
        print(f" - {clave}: {valor}")

eeg_data = {}
with h5py.File(ruta, 'r') as f:
    for nombre in nombres_eeg:
        eeg_data[nombre] = np.array(f['signals']['eeg'][nombre])

eeg_filtradas = {}
for canal in eeg_data:
    señal = eeg_data[canal]
    señal_centrada = señal - np.mean(señal)
    señal_filtrada = sig.sosfiltfilt(sos_eeg, señal_centrada)
    eeg_filtradas[canal] = señal_filtrada

# === Aplicar filtro a EOG ===
with h5py.File(ruta, 'r') as f:
    grupo_eog = f['signals']['eog']
    nombres_eog = list(grupo_eog.keys())
    print("\nCanales EOG disponibles:")
    for nombre in nombres_eog:
        canal = grupo_eog[nombre]
        print(f" - {nombre}: {len(canal)} muestras")
    fs_eog = grupo_eog.attrs['fs']
    print(f"\nFrecuencia de muestreo (fs_eog): {fs_eog} Hz")
    print(f"Cantidad de canales EOG: {len(nombres_eog)}")
    print("\nAtributos del grupo EOG:")
    for clave, valor in grupo_eog.attrs.items():
        print(f" - {clave}: {valor}")

eog_data = {}
with h5py.File(ruta, 'r') as f:
    for nombre in nombres_eog:
        eog_data[nombre] = np.array(f['signals']['eog'][nombre])

eog_filtradas = {}
for canal in eog_data:
    señal = eog_data[canal]
    señal_centrada = señal - np.mean(señal)
    señal_filtrada = sig.sosfiltfilt(sos_eog, señal_centrada)
    eog_filtradas[canal] = señal_filtrada



#%% HIPNOGRAMA:
import numpy as np
import matplotlib.pyplot as plt
import h5py
import json

# === Cargar hipnograma ===
with h5py.File(ruta, 'r') as f:
    hipnograma = f['hypnogram'][:]

# === Diccionario de clases ===
clases_dict = {
    0: 'Wake',
    1: 'N1',
    2: 'N2',
    3: 'N3',
    4: 'REM'
}

# === Ocurrencia deseada por clase ===
oc_0 =26  # Wake
oc_1 = 2  # N1
oc_2 = 6 # N2
oc_3 = 10  # N3
oc_4 = 9  # REM

oc_dict = {
    0: oc_0,
    1: oc_1,
    2: oc_2,
    3: oc_3,
    4: oc_4
}

# === Buscar índices para cada clase según ocurrencia deseada ===
indices_por_clase = {}
for clase in np.unique(hipnograma):
    ocurrencias = np.where(hipnograma == clase)[0]
    oc = oc_dict[clase]
    if len(ocurrencias) > oc:
        indices_por_clase[clase] = ocurrencias[oc]

# === Plot del hipnograma con ventanas destacadas ===
tiempo_horas = np.arange(len(hipnograma)) * 30 / 3600  # 30 s por época

plt.figure(figsize=(18, 4))
plt.step(tiempo_horas, hipnograma, where='post', color='black')

for clase, idx in indices_por_clase.items():
    t_ini = idx * 30 / 3600
    t_fin = (idx + 1) * 30 / 3600
    #plt.axvspan(t_ini, t_fin, color='C1', alpha=0.5)

plt.yticks(ticks=list(clases_dict.keys()), labels=list(clases_dict.values()))
plt.gca().invert_yaxis()
plt.xlabel('Tiempo [horas]')
plt.ylabel('Etapa')
plt.xlim([0, len(hipnograma) * 30 / 3600])
plt.title('Hipnograma')
plt.grid(True, axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# === Parámetros generales ===
fs = 250
epoca_dur = 30
epoca_muestras = epoca_dur * fs
t = np.arange(epoca_muestras) / fs
espaciado_eeg = 150


#%%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# === Seleccionar clase y época ===
clase = 4  # Por ejemplo, REM
idx_epoca = indices_por_clase[clase]
inicio = idx_epoca * epoca_muestras
fin = (idx_epoca + 1) * epoca_muestras
t = np.arange(epoca_muestras) / fs  # tiempo en segundos

# === EEG y EOG ===
canales_eeg = list(eeg_filtradas.keys())
datos_eeg = np.array([eeg_filtradas[c][inicio:fin] for c in canales_eeg])

canales_eog = list(eog_filtradas.keys())
datos_eog = np.array([eog_filtradas[c][inicio:fin] for c in canales_eog])

# === Crear figura con GridSpec ===
fig = plt.figure(figsize=(18, 13))
etiquetas = ["Wake", "N1", "N2", "N3", "REM"]
fig.suptitle(f"Etapa: {etiquetas[clase]}", fontsize=8)
fig.subplots_adjust(top=0.95)

gs = GridSpec(7, 2, figure=fig, height_ratios=[1]*6 + [1], hspace=0.15)

# === Plot EEG (6x2) ===
for i in range(12):
    fila = i // 2
    col = i % 2
    ax = fig.add_subplot(gs[fila, col])
    ax.plot(t, datos_eeg[i], linewidth=0.5, color='black')
    ax.set_ylabel(canales_eeg[i], rotation=0, labelpad=20, fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.tick_params(axis='y', labelsize=6)
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.set_xlim([0, epoca_dur])
    ax.set_ylim([-100, 100])

# === Plot EOG (fila 6, columna 0 y 1) ===
for i in range(2):
    ax = fig.add_subplot(gs[6, i])
    ax.plot(t, datos_eog[i], linewidth=0.5, color='black')
    ax.set_ylabel(canales_eog[i], rotation=0, labelpad=20, fontsize=8, color='grey')
    ax.set_xlabel("Tiempo [s]", fontsize=6)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.tick_params(axis='y', labelsize=6)
    ax.tick_params(axis='x', labelsize=6)
    ax.set_xlim([0, epoca_dur])
    ax.set_ylim([-100, 100])

plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.signal as sig

# === Parámetros ===
canal_eeg = 'F4_M1'   # Cambiá esto por el canal que quieras analizar
fs = 250           # Frecuencia de muestreo en Hz
nperseg = 5 * fs   # Ventana de 5 segundos
noverlap = 2.5 * fs  # 50% de solapamiento

# === Cargar señal completa del canal EEG ===
with h5py.File(ruta, 'r') as f:
    datos = np.array(f['signals']['eeg'][canal_eeg])

# === Aplicar filtro ===
datos_centrada = datos - np.mean(datos)
datos_filtrada = sig.sosfiltfilt(sos_eeg, datos_centrada)

# === Calcular espectrograma ===
f, t_seg, Sxx = sig.spectrogram(datos_filtrada, fs=fs, nperseg=int(nperseg), noverlap=int(noverlap))

# === Convertir tiempo a horas ===
t_horas = t_seg / 3600
#%%
# === Plot ===
plt.figure(figsize=(16, 5))
plt.pcolormesh(t_horas, f, 10 * np.log10(Sxx), shading='gouraud', cmap='Spectral', vmin=-40, vmax=20)
plt.ylabel('Frecuencia [Hz]')
plt.xlabel('Tiempo [horas]')
plt.colorbar(label='PSD [dB]')
plt.title(f"Espectrograma – Canal {canal_eeg}")
plt.ylim([0, 45])  # Máximo razonable para EEG
plt.tight_layout()
plt.show()

#%%
plt.figure(figsize=(16, 5))
plt.pcolormesh(t_horas, f, Sxx, shading='gouraud', cmap='Spectral', vmin=0, vmax=10)
plt.ylabel('Frecuencia [Hz]')
plt.xlabel('Tiempo [horas]')
plt.colorbar(label='Potencia [uV²/Hz]')
plt.title(f"Espectrograma – Canal {canal_eeg}")
plt.ylim([0, 40])  # Mostrar solo hasta 40 Hz
plt.tight_layout()
plt.show()

#%%
from utils_info import detector_k_complex
from collections import defaultdict

# === Inicializar contador por canal y clase ===
conteo_k_complexes = defaultdict(lambda: defaultdict(int))

# Recorrer todas las épocas
for idx, clase in enumerate(hipnograma):
    inicio = idx * epoca_muestras
    fin = (idx + 1) * epoca_muestras
    for canal in eeg_filtradas:
        segmento = eeg_filtradas[canal][inicio:fin]
        k_detectados = detector_k_complex(fs, segmento)
        conteo_k_complexes[canal][clase] += k_detectados

# === Mostrar resultados ===
clases_dict = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
print("Cantidad total de K-complexes detectados por canal y etapa:\n")
for canal in sorted(conteo_k_complexes):
    print(f"{canal}:")
    for clase in range(5):
        nombre_clase = clases_dict[clase]
        cantidad = conteo_k_complexes[canal][clase]
        print(f"  {nombre_clase}: {cantidad}")

#%%
from utils_info import detector_k_complex
from scipy.signal import find_peaks
import pywt
import matplotlib.pyplot as plt
import numpy as np

# === Buscar primer segmento N2 con K-complex detectable ===
canal_objetivo = None
indice_objetivo = None
for canal in eeg_filtradas:
    for idx, clase in enumerate(hipnograma):
        if clase != 2:
            continue
        inicio = idx * epoca_muestras
        fin = (idx + 1) * epoca_muestras
        segmento = eeg_filtradas[canal][inicio:fin]
        if detector_k_complex(fs, segmento) > 0:
            canal_objetivo = canal
            indice_objetivo = idx
            break
    if canal_objetivo:
        break

if canal_objetivo and indice_objetivo is not None:
    print(f"Canal seleccionado: {canal_objetivo} — Época N2: {indice_objetivo}")
    segmento = eeg_filtradas[canal_objetivo][indice_objetivo * epoca_muestras : (indice_objetivo + 1) * epoca_muestras]
    inicio = indice_objetivo * epoca_muestras
    centro_seg = inicio / fs
    t = np.arange(len(segmento)) / fs + centro_seg

    # Wavelet
    coeffs = pywt.wavedec(segmento, 'sym6', level=6)
    a6 = coeffs[0]

    # TEO
    def teo(x):
        return x[2:-2]**2 - x[1:-3]*x[3:-1] - x[0:-4]*x[4:]
    teo_a6 = teo(a6)
    teo_a6_norm = (teo_a6 - np.mean(teo_a6)) / np.std(teo_a6)

    t_a6 = np.arange(len(a6)) * (2**6) / fs + centro_seg
    t_teo = np.arange(len(teo_a6)) * (2**6) / fs + centro_seg + (2**6 / fs)
    t_ini = centro_seg
    t_fin = t_ini + 1.4

    # === Figura multietapa ===
    fig = plt.figure(figsize=(14, 20))
    labels = ['Original', 'D6', 'D5', 'D4', 'D3', 'D2', 'D1', 'A6', 'TEO A6']

    plt.subplot(9, 1, 1)
    plt.plot(t, segmento, color='black')
    #plt.axvline(t_ini, color='red', linestyle='--')
    #plt.axvline(t_fin, color='red', linestyle='--')
    plt.title(labels[0])
    plt.ylabel("µV")

    for i in range(1, 7):
        coef = coeffs[i]
        factor = 2**(7 - i)
        t_sub = np.arange(len(coef)) * factor / fs + centro_seg
        plt.subplot(9, 1, i + 1)
        plt.plot(t_sub, coef, color='black')
        plt.title(labels[i])
        plt.ylabel("Coef")

    plt.subplot(9, 1, 8)
    plt.plot(t_a6, a6, color='black')
    plt.title(labels[7])
    plt.ylabel("Coef")

    plt.subplot(9, 1, 9)
    plt.plot(t_teo, teo_a6, color='black')
    plt.title(labels[8])
    plt.xlabel("Tiempo [s]")
    plt.ylabel("TEO")

    plt.tight_layout()
    plt.show()

    # === Detección de picos ===
    fs_down = fs / 2**6
    peaks, props = find_peaks(teo_a6_norm, prominence=3, wlen=int(5 * fs_down), distance=int(5 * fs_down), height=3)

    # === Visualización
    plt.figure(figsize=(12, 3))
    plt.plot(t_teo, teo_a6_norm, label='TEO norm', color = 'black')
    plt.plot(t_teo[peaks], teo_a6_norm[peaks], 'rx', label='k-complex')
    plt.title("Picos detectados")
    plt.xlabel("Tiempo [s]")
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No se encontró un segmento N2 con complejos K detectables.")
