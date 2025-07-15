''' DESCRIPCION:
Este script fue utilizado para realizar trabajo exploratorio sobre un único archivo de EEG del dataset DOD-H. 
Permite visualizar las señales EEG y EOG, aplicar filtros IIR Butterworth (en forma SOS y con filtrado bidireccional), 
crear plantillas de diseño, graficar espectrogramas, estimar densidades espectrales de potencia (PSD) 
y analizar morfología de las señales a nivel de segmentos.

También se implementaron y probaron diferentes técnicas de detección de complejos K utilizando transformada wavelet 
y energía de Teager. Este entorno permitió ajustar visualmente umbrales, evaluar diferentes niveles de descomposición 
y comparar el comportamiento del detector sobre canales y etapas específicas.

El código simula para un solo sujeto lo que en `extraccion_feat.py` se realiza automáticamente sobre múltiples archivos,
permitiendo verificar visualmente la calidad del filtrado y la utilidad de cada característica extraída.
'''
#%% LIBRERIAS REQUERIDAS
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pytc2.sistemas_lineales import plot_plantilla
import scipy.signal as sig
import os
from matplotlib.gridspec import GridSpec
from utils_info import detector_k_complex
from scipy.signal import find_peaks
import pywt
from scipy.signal import welch
from collections import defaultdict
def teo(x): # es un teo modificado, en vez de 1 muestra son 2 las que se mueve. NO esta copado
    return x[2:-2]**2 - x[1:-3]*x[3:-1] - x[0:-4]*x[4:]
def teo_clasico(x):
    return x[1:-1]**2 - x[0:-2] * x[2:]

#%% FILTRADO EEG Y EOG + DISEÑO DE PLANTILLA

# EEG
fs = 250
fpass = np.array([0.5, 45])
fstop = np.array([0.1, 50])
ripple = 0.5
attenuation = 20

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

fig = plt.figure(figsize=(14, 8))
plt.subplot(3, 1, 1)
plt.title('EEG - Banda de transicion baja')
plt.grid(which='both', axis='both')
plt.xlim([0, 5])
plt.ylim([-30, 1])
plot_plantilla(filter_type='bandpass', fpass=fpass, ripple=ripple, fstop=fstop, attenuation=attenuation, fs=fs)
plt.plot(w_Hz, h_db, label='Butterworth (EEG)', color='lightseagreen', linewidth=1)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.legend()

plt.subplot(3, 1, 2)
plt.title('EEG & EOG - Banda de transicion alta')
plt.grid(which='both', axis='both')
plt.xlim([40, 55])
plt.ylim([-30, 1])
plot_plantilla(filter_type='bandpass', fpass=fpass, ripple=ripple, fstop=fstop, attenuation=attenuation, fs=fs)
plt.plot(w_Hz, h_db, label='Butterworth (EEG)', color='lightseagreen', linewidth=1)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.legend()

# EOG
fs_eog = 250
fpass = np.array([0.3, 45])
fstop = np.array([0.1, 50])
ripple = 0.5
attenuation = 20

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

plt.subplot(3, 1, 3)
plt.title('EOG - Banda de transicion baja')
plt.grid(which='both', axis='both')
plt.xlim([0, 5])
plt.ylim([-30, 1])
plot_plantilla(filter_type='bandpass', fpass=fpass, ripple=ripple, fstop=fstop, attenuation=attenuation, fs=fs_eog)
plt.plot(w_Hz, h_db, label='Butterworth (EOG)', color='lightseagreen', linewidth=1)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.legend()

plt.tight_layout()
plt.show()

#%% SELECCION DE ARCHIVO (SUJETO)

script_dir = os.path.dirname(os.path.abspath(__file__))
archivo_h5 = os.path.join(script_dir, 'DOD-H', '7d778801-88e7-5086-ad1d-70f31a371876.h5')

if not os.path.exists(archivo_h5):
    raise FileNotFoundError(f"No se encontró el archivo: {archivo_h5}")
else:
    print("Ruta seleccionada:", archivo_h5)
    ruta = archivo_h5

#%% APERTURA DE ARCHIVO + PRINT DE INFORMACION
print('INFORMACION EEG:')
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
        
print('INFORMACION EOG:')
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

#%% FILTRADO DE EEG Y EOG
eeg_filtradas = {}
for canal in eeg_data:
    señal = eeg_data[canal]
    señal_centrada = señal - np.mean(señal)
    señal_filtrada = sig.sosfiltfilt(sos_eeg, señal_centrada)
    eeg_filtradas[canal] = señal_filtrada


eog_filtradas = {}
for canal in eog_data:
    señal = eog_data[canal]
    señal_centrada = señal - np.mean(señal)
    señal_filtrada = sig.sosfiltfilt(sos_eog, señal_centrada)
    eog_filtradas[canal] = señal_filtrada


#%% CARGA DEL HIPNOGRAMA: El hipnograma indica a que etapa de sueño corresponde cada ventana de 30seg

with h5py.File(ruta, 'r') as f:
    hipnograma = f['hypnogram'][:]


clases_dict = {
    0: 'Wake',
    1: 'N1',
    2: 'N2',
    3: 'N3',
    4: 'REM'
}

#%% PLOTEO DEL HIPNOGRAMA
# Seleccion de num de ocurrencia a plotear
oc_0 =26  
oc_1 = 2  
oc_2 = 6 
oc_3 = 10  
oc_4 = 9  

oc_dict = {
    0: oc_0,
    1: oc_1,
    2: oc_2,
    3: oc_3,
    4: oc_4
}

# A partir de las ocurrencias se buscan los indices
indices_por_clase = {}
for clase in np.unique(hipnograma):
    ocurrencias = np.where(hipnograma == clase)[0]
    oc = oc_dict[clase]
    if len(ocurrencias) > oc:
        indices_por_clase[clase] = ocurrencias[oc]



mapa_clase_plot = {0: 5, 4: 4, 1: 3, 2: 2, 3: 1}
hipnograma_plot = np.vectorize(mapa_clase_plot.get)(hipnograma)


tiempo_horas = np.arange(len(hipnograma)) * 30 / 3600  

plt.figure(figsize=(18, 4))
plt.step(tiempo_horas, hipnograma_plot, where='post', color='black')

for clase, idx in indices_por_clase.items():
    t_ini = idx * 30 / 3600
    t_fin = (idx + 1) * 30 / 3600
    #plt.axvspan(t_ini, t_fin, color='C1', alpha=0.5)

# Etiquetas para el nuevo eje Y
ticks_y = [5, 4, 3, 2, 1]
labels_y = ['Wake', 'REM', 'N1', 'N2', 'N3']
plt.yticks(ticks=ticks_y, labels=labels_y)

plt.xlabel('Tiempo [horas]')
plt.ylabel('Etapa')
plt.xlim([0, len(hipnograma) * 30 / 3600])
plt.title('Hipnograma')
plt.grid(True, axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()


#%% PLOTEO DE TODOS LOS CANALES DE 1 EPOCA (30seg) DE LA ETAPA SELECCIONADA

epoca_dur = 30
epoca_muestras = epoca_dur * fs
t = np.arange(epoca_muestras) / fs
espaciado_eeg = 150

clase = 4  
idx_epoca = indices_por_clase[clase]
inicio = idx_epoca * epoca_muestras
fin = (idx_epoca + 1) * epoca_muestras
t = np.arange(epoca_muestras) / fs  


canales_eeg = list(eeg_filtradas.keys())
datos_eeg = np.array([eeg_filtradas[c][inicio:fin] for c in canales_eeg])

canales_eog = list(eog_filtradas.keys())
datos_eog = np.array([eog_filtradas[c][inicio:fin] for c in canales_eog])


fig = plt.figure(figsize=(18, 13))
etiquetas = ["Wake", "N1", "N2", "N3", "REM"]
fig.suptitle(f"Etapa: {etiquetas[clase]}", fontsize=8)
fig.subplots_adjust(top=0.95)

gs = GridSpec(7, 2, figure=fig, height_ratios=[1]*6 + [1], hspace=0.15)


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

#%% PLOTEO PSD con bandas coloreadas - lineal y dB
# SE PLOTEA LA PSD DE LAS OCURRENCIAS ELEJIDAS ANTES
# SE PLOTEA UNA PSD DE CADA EPOCA

'''
Canales EEG disponibles:
 - C3_M2
 - F3_F4
 - F3_M2
 - F3_O1
 - F4_M1
 - F4_O2
 - FP1_F3
 - FP1_M2
 - FP1_O1
 - FP2_F4
 - FP2_M1
 - FP2_O2
'''
# Parámetros
canal_eeg = 'F4_M1'
epoca_dur = 30
epoca_muestras = fs * epoca_dur

clases_dict = {
    0: 'Wake',
    1: 'N1',
    2: 'N2',
    3: 'N3',
    4: 'REM'
}

colores = {
    0: 'darkorange',
    1: 'slateblue',
    2: 'forestgreen',
    3: 'crimson',
    4: 'gold'
}


bandas = [
    ('Delta', 0.5, 4, '#d0f0c0'),
    ('Theta', 4, 8, '#c0d8f0'),
    ('Alfa', 8, 13, '#f0e0c0'),
    ('Beta', 13, 30, '#f0c0c0'),
    ('Gamma', 30, 45, '#e0c0f0')
]

# ---------- VERSION 1: PSD en ESCALA LINEAL ----------
plt.figure(figsize=(10, 6))
for nombre, f_low, f_high, color in bandas:
    plt.axvspan(f_low, f_high, color=color, alpha=0.6, label=nombre)

for clase in sorted(indices_por_clase.keys()):
    idx = indices_por_clase[clase]
    inicio = idx * epoca_muestras
    fin = (idx + 1) * epoca_muestras
    segmento = eeg_filtradas[canal_eeg][inicio:fin]
    #segmento = (segmento - np.mean(segmento)) / np.std(segmento)

    f, Pxx = welch(segmento, fs=fs, nperseg=min(len(segmento), fs*4), noverlap=fs*2)
    plt.plot(f, Pxx, label=clases_dict[clase], color=colores[clase], linewidth=1.5)

plt.title(f'PSD (escala lineal) - Canal {canal_eeg}')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD')
plt.xlim(0, 50)
xticks_band_edges = sorted({0.5, 4, 8, 13, 30, 35, 45, 50, fs/2})
plt.xticks(
    xticks_band_edges,
    [f"{x:.1f}" if x == 0.5 else f"{int(x)}" for x in xticks_band_edges]
)

plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(loc='upper right', ncol=2)
plt.tight_layout()
plt.show()


# ---------- VERSION 2: PSD en dB ----------
plt.figure(figsize=(10, 6))
for nombre, f_low, f_high, color in bandas:
    plt.axvspan(f_low, f_high, color=color, alpha=0.8, label=nombre)

for clase in sorted(indices_por_clase.keys()):
    idx = indices_por_clase[clase]
    inicio = idx * epoca_muestras
    fin = (idx + 1) * epoca_muestras
    segmento = eeg_filtradas[canal_eeg][inicio:fin]
    #segmento = (segmento - np.mean(segmento)) / np.std(segmento)

    f, Pxx = welch(segmento, fs=fs, nperseg=min(len(segmento), fs*4), noverlap=fs*2)
    Pxx_dB = 10 * np.log10(Pxx + 1e-12)  # protección para evitar log(0)
    plt.plot(f, Pxx_dB, label=clases_dict[clase], color=colores[clase], linewidth=1.5)

plt.title(f'PSD - Canal {canal_eeg}')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.xlim(0, fs/2)
#lt.ylim(-40,35)
plt.grid(True, linestyle='--', alpha=0.3)
xticks_band_edges = sorted({0.5, 4, 8, 13, 30, 35, 45, 50, fs/2})
plt.xticks(
    xticks_band_edges,
    [f"{x:.1f}" if x == 0.5 else f"{int(x)}" for x in xticks_band_edges]
)


plt.legend(loc='upper right', ncol=2)
plt.tight_layout()
plt.show()

# ---------- VERSION 3: Señales temporales asociadas a cada PSD ----------
fig, axs = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
fig.suptitle(f'Señales temporales - Canal {canal_eeg}', fontsize=14)

for i, clase in enumerate(sorted(indices_por_clase.keys())):
    idx = indices_por_clase[clase]
    inicio = idx * epoca_muestras
    fin = (idx + 1) * epoca_muestras
    segmento = eeg_filtradas[canal_eeg][inicio:fin]
    t_seg = np.arange(len(segmento)) / fs

    axs[i].plot(t_seg, segmento, color=colores[clase], linewidth=0.8)
    axs[i].set_ylabel(clases_dict[clase], rotation=0, labelpad=30, fontsize=10)
    axs[i].grid(True, linestyle='--', alpha=0.3)
    axs[i].set_xlim(0, epoca_dur)
    axs[i].set_ylim(-120,120)

axs[-1].set_xlabel("Tiempo [s]", fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# ---------- VERSION 4: Señal temporal + espectro en dB por etapa ----------
fig, axs = plt.subplots(5, 2, figsize=(14, 10), gridspec_kw={'width_ratios': [1, 1]})
fig.suptitle(f'Señal temporal y PSD en dB por etapa - Canal {canal_eeg}', fontsize=14)

for i, clase in enumerate(sorted(indices_por_clase.keys())):
    idx = indices_por_clase[clase]
    inicio = idx * epoca_muestras
    fin = (idx + 1) * epoca_muestras
    segmento = eeg_filtradas[canal_eeg][inicio:fin]
    t_seg = np.arange(len(segmento)) / fs

    # Señal temporal (columna izquierda)
    axs[i, 0].plot(t_seg, segmento, color=colores[clase], linewidth=0.8)
    axs[i, 0].set_ylabel(clases_dict[clase], rotation=0, labelpad=30, fontsize=10)
    axs[i, 0].grid(True, linestyle='--', alpha=0.3)
    axs[i, 0].set_xlim(0, 30)
    axs[i, 0].set_ylim(-120, 120)
    if i == 4:
        axs[i, 0].set_xlabel("Tiempo [s]", fontsize=10)

    # PSD en dB (columna derecha)
    f, Pxx = welch(segmento, fs=fs, nperseg=min(len(segmento), fs*4), noverlap=fs*2)
    Pxx_dB = 10 * np.log10(Pxx + 1e-12)
    axs[i, 1].plot(f, Pxx_dB, color=colores[clase], linewidth=1)
    axs[i, 1].set_xlim(0, 45)
    axs[i, 1].set_ylim(-40, 40)
    axs[i, 1].grid(True, linestyle='--', alpha=0.3)
    if i == 4:
        axs[i, 1].set_xlabel("Frecuencia [Hz]", fontsize=10)
    axs[i, 1].set_ylabel("PSD [dB]", fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.align_xlabels()
plt.show()


# ---------- VERSION 5: Señal y espectro de una sola época en negro ----------

clase = 3  
idx = indices_por_clase[clase]
inicio = idx * epoca_muestras
fin = (idx + 1) * epoca_muestras
segmento = eeg_filtradas[canal_eeg][inicio:fin]
t_seg = np.arange(len(segmento)) / fs

f, Pxx = welch(segmento, fs=fs, nperseg=min(len(segmento), fs*4), noverlap=fs*2)
Pxx_dB = 10 * np.log10(Pxx + 1e-12)

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle(f"Señal y PSD - Canal {canal_eeg} - Etapa {clases_dict[clase]}", fontsize=14)

# Señal temporal
axs[0].plot(t_seg, segmento, color='black', linewidth=0.8)
axs[0].set_title("Señal temporal")
axs[0].set_xlabel("Tiempo [s]")
axs[0].set_ylabel("Amplitud [µV]")
axs[0].grid(True, linestyle='--', alpha=0.3)
axs[0].set_xlim(0, 30)
axs[0].set_ylim(-150, 150)


axs[1].plot(f, Pxx_dB, color='black', linewidth=1)
axs[1].set_title("PSD (Welch, en dB)")
axs[1].set_xlabel("Frecuencia [Hz]")
axs[1].set_ylabel("PSD [dB]")
axs[1].grid(True, linestyle='--', alpha=0.3)
axs[1].set_xlim(0,fs/2)

ymin, ymax = axs[1].get_ylim()

axs[1].fill_between(f, ymin, ymax, where=(f >= 0.5) & (f <= 45), color='green', alpha=0.1, label='Banda de paso')
axs[1].fill_between(f, ymin, ymax, where=(f > 45) & (f <= 50), color='orange', alpha=0.2, label='Banda de transición')
axs[1].fill_between(f, ymin, ymax, where=(f > 50), color='red', alpha=0.1, label='Banda de stop')


axs[1].legend()
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()


#%% VISUALIZACION DE ESPECTOGRAMA
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.signal as sig

# parametros
canal_eeg = 'F4_M1'   
         
nperseg = 5 * fs   # tamaño de ventana del espectograma
noverlap = 2.5 * fs  # solapamiento (seria el 50%)

# carga de la señal completa
with h5py.File(ruta, 'r') as f:
    datos = np.array(f['signals']['eeg'][canal_eeg])

# filtrado
datos_centrada = datos - np.mean(datos)
datos_filtrada = sig.sosfiltfilt(sos_eeg, datos_centrada)

# espectograma
f, t_seg, Sxx = sig.spectrogram(datos_filtrada, fs=fs, nperseg=int(nperseg), noverlap=int(noverlap))


t_horas = t_seg / 3600

#EN DB 
plt.figure(figsize=(16, 5))
plt.pcolormesh(t_horas, f, 10 * np.log10(Sxx), shading='gouraud', cmap='Spectral', vmin=-40, vmax=20)
plt.ylabel('Frecuencia [Hz]')
plt.xlabel('Tiempo [horas]')
plt.colorbar(label='PSD [dB]')
plt.title(f"Espectrograma – Canal {canal_eeg}")
plt.ylim([0, 60]) 
plt.tight_layout()
plt.show()

#%% NO DB
plt.figure(figsize=(16, 5))
plt.pcolormesh(t_horas, f, Sxx, shading='gouraud', cmap='Spectral', vmin=0, vmax=10)
plt.ylabel('Frecuencia [Hz]')
plt.xlabel('Tiempo [horas]')
plt.colorbar(label='Potencia [uV²/Hz]')
plt.title(f"Espectrograma – Canal {canal_eeg}")
plt.ylim([0, 40])  
plt.tight_layout()
plt.show()

    
#%% 
'''
DESCOMPOSICION CON TRANSFORMADA WAVELET DISCRETA. 
Se usa la familia Sym6. 
Se elige la ocurrencia que quiera Y canal
Se prueba en epocas N2 (DONDE HAY K-COMPLEX)
'''

n_ocurrencia = 14  # se puede cambiar
contador_n2 = 0
indice_objetivo = None
canal_objetivo = 'C3_M2'  # es el mejor para complejos K

for idx, clase in enumerate(hipnograma):
    if clase == 2:  # N2
        contador_n2 += 1
        if contador_n2 == n_ocurrencia:
            indice_objetivo = idx
            break

if indice_objetivo is not None:
    print(f"Canal seleccionado: {canal_objetivo} — Época N2 número {n_ocurrencia}: {indice_objetivo}")
    
    inicio = indice_objetivo * epoca_muestras
    fin = (indice_objetivo + 1) * epoca_muestras
    segmento = eeg_filtradas[canal_objetivo][inicio:fin]
    centro_seg = inicio / fs
    t = np.arange(len(segmento)) / fs + centro_seg

    # Wavelet
    coeffs = pywt.wavedec(segmento, 'sym4', level=6)
    a6 = coeffs[0]

    teo_a6 = teo_clasico(a6)
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

    # === Detecn de picos ===
    fs_down = fs / 2**6
    peaks, props = find_peaks(teo_a6_norm, prominence=3, wlen=int(5 * fs_down), distance=int(5 * fs_down), height=3.2)

    # === Visualizacion
    plt.figure(figsize=(12, 3))
    plt.plot(t_teo, teo_a6_norm, label='TEO norm', color='black')
    plt.plot(t_teo[peaks], teo_a6_norm[peaks], 'rx', label='k-complex')
    plt.title("Picos detectados")
    plt.xlabel("Tiempo [s]")
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print(f"No se encontró la ocurrencia número {n_ocurrencia} de N2.")

  
#%% CANTIDAD DE COMPLEJOS K ENCONTRADOS EN TODA LA NOCHE X CANAL


print('Version 6 nieveles')

conteo_k_complexes = defaultdict(lambda: defaultdict(int))


for idx, clase in enumerate(hipnograma):
    inicio = idx * epoca_muestras
    fin = (idx + 1) * epoca_muestras
    for canal in eeg_filtradas:
        segmento = eeg_filtradas[canal][inicio:fin]
        k_detectados = detector_k_complex(fs, segmento)
        conteo_k_complexes[canal][clase] += k_detectados


clases_dict = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
print("Cantidad total de K-complexes detectados por canal y etapa:\n")
for canal in sorted(conteo_k_complexes):
    print(f"{canal}:")
    for clase in range(5):
        nombre_clase = clases_dict[clase]
        cantidad = conteo_k_complexes[canal][clase]
        print(f"  {nombre_clase}: {cantidad}")
        
