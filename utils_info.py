import scipy.signal as sig
import numpy as np
import pywt
from scipy.signal import find_peaks
 

def filtrado(señal, fs, fpass, ripple, fstop, attenuation):
    """
    Aplica un filtro Butterworth a la señal dada.
    
    Parámetros:
    - señal: array_like, la señal a filtrar.
    - fs: int, frecuencia de muestreo.
    - fpass: tuple, frecuencias de paso (bajo y alto).
    - ripple: float, ripple en la banda de paso.
    - fstop: tuple, frecuencias de parada (bajo y alto).
    - attenuation: float, atenuación en la banda de parada.
    
    Retorna:
    - señal_filtrada: array_like, señal filtrada.
    """
    sos = sig.iirdesign(
        wp=fpass,
        ws=fstop,
        gpass=ripple,
        gstop=attenuation,
        ftype='butter',
        output='sos',
        fs=fs
    )
    señal_centrada = señal - np.mean(señal)
    señal_filtrada = sig.sosfiltfilt(sos, señal_centrada)
    
    return señal_filtrada

def segmentar(señal, fs, duracion):
    """
    Segmenta la señal en segmentos de duración especificada.
    
    Parámetros:
    - señal: array_like, la señal a segmentar.
    - fs: int, frecuencia de muestreo.
    - duracion: float, duración de cada segmento en segundos.
    
    Retorna:
    - segmentos: list, lista de segmentos de la señal.
    """
    n_muestras = int(fs * duracion)
    segmentos = [señal[i:i + n_muestras] for i in range(0, len(señal), n_muestras) if i + n_muestras <= len(señal)]
    
    return segmentos

def features_temporales(segmento,fs):
    """
    Extrae características temporales de los segmentos.
    
    Parámetros:
    - segmento: 1 segmento.
    
    Retorna:
    - features: list, lista de características extraídas.
    """
    features = []
    varianza = np.var(segmento)
    cruces = np.sum(np.diff(np.sign(segmento)) != 0)
    
    return varianza, cruces

def features_espectrales(segmento, fs):
    f, Pxx = sig.welch(segmento, fs=fs, nperseg=fs*2, noverlap=fs, window='hann')
    energia_total = np.trapz(Pxx, f)

    bandas = {
        'delta': (0.5, 4),
        'theta_baja': (4, 6),
        'theta_alta': (6, 8),
        'alfa': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }

    proporciones = {}
    for nombre, (fmin, fmax) in bandas.items():
        mask = (f >= fmin) & (f <= fmax)
        energia_banda = np.trapz(Pxx[mask], f[mask])
        proporciones[nombre] = energia_banda / energia_total if energia_total > 0 else 0

    potencia_media = np.mean(Pxx)
    freq_pico = f[np.argmax(Pxx)]

    # Banda dominante = frecuencia con mayor energía
    banda_dominante = freq_pico

    # Relación alfa/delta
    rel_alpha_delta = proporciones['alfa'] / (proporciones['delta'] + 1e-10)

    return (
        potencia_media,
        freq_pico,
        proporciones['delta'],
        proporciones['theta_baja'],
        proporciones['theta_alta'],
        proporciones['alfa'],
        proporciones['beta'],
        proporciones['gamma'],
        rel_alpha_delta,
        banda_dominante
    )




def es_candidato(segmento, fs, umbral=100, ventana_s=2, paso_s=0.25):
    ventana = int(ventana_s * fs)
    paso = int(paso_s * fs)
    candidatos = []

    for i in range(0, len(segmento) - ventana + 1, paso):
        ventana_i = segmento[i:i + ventana]
        idx_min = np.argmin(ventana_i)
        idx_max = np.argmax(ventana_i)
        amp_pp = ventana_i[idx_max] - ventana_i[idx_min]
        dur = abs(idx_max - idx_min) / fs

        if amp_pp >= umbral and idx_min < idx_max and 0.5 <= dur <= 1.0:
            t_rel = i / fs
            candidatos.append(t_rel)

    if candidatos:
        return True
    else:
        return False
 

def teo(x):
    return x[2:-2]**2 - x[1:-3] * x[3:-1] - x[0:-4] * x[4:]
    
def detector_k_complex(fs, segmento):
    
    if es_candidato(segmento, fs):
        coeffs = pywt.wavedec(segmento, 'sym6', level=6)
        a6 = coeffs[0]
        fs_down = fs / 2**6 
        teo_a6 = teo(a6)
        teo_a6_norm = (teo_a6 - np.mean(teo_a6)) / np.std(teo_a6)
        peaks, props = find_peaks(teo_a6_norm,
                                  prominence=4.5,  # cuanto debe destacar
                                  wlen = int(2 * fs_down), distance = 2 * fs_down, height=3.2)
        cant_k_complex = len(peaks)
        return cant_k_complex
    else:
        cant_k_complex = 0
        return cant_k_complex
        



from scipy.signal import welch


def features_eog(segmento, fs):
    """
    Extrae features de un segmento EOG filtrado.
    
    Parámetros:
        segmento : array_like
            Señal EOG (filtrada) del segmento (1D).
        fs : int or float
            Frecuencia de muestreo (Hz).
    
    Devuelve:
        - n_picos : int
        - media_picos_pos : float
        - media_picos_neg : float
        - varianza : float
        - cruces_cero : int
        - energia_rel_0_1 : float
        - energia_rel_1_4 : float
        - energia_rel_4_8 : float
    """
    x = np.asarray(segmento)
    N = len(x)

    # --- Varianza
    varianza = np.var(x)

    # --- Cruces por cero
    cruces_cero = np.sum((x[:-1] * x[1:]) < 0)

    # --- Detección de picos con std local y distancia mínima
    std = np.std(x)
    dist_min = int(0.1 * fs)
    picos_pos, _ = find_peaks(x, height=std, distance=dist_min)
    picos_neg, _ = find_peaks(-x, height=std, distance=dist_min)

    n_picos = len(picos_pos) + len(picos_neg)
    media_picos_pos = np.mean(x[picos_pos]) if len(picos_pos) > 0 else np.nan
    media_picos_neg = np.mean(x[picos_neg]) if len(picos_neg) > 0 else np.nan

    # --- Espectro con Welch
    f, Pxx = welch(x, fs=fs, nperseg=min(N, fs*4), noverlap=fs*2 if N > fs*4 else fs//2)
    energia_total = np.trapz(Pxx, f)

    def energia_rel(f_low, f_high):
        idx = (f >= f_low) & (f < f_high)
        return np.trapz(Pxx[idx], f[idx]) / energia_total if energia_total > 0 else np.nan

    energia_rel_0_1 = energia_rel(0.1, 1)
    energia_rel_1_4 = energia_rel(1, 4)
    energia_rel_4_8 = energia_rel(4, 8)

    return n_picos, media_picos_pos, media_picos_neg, varianza, cruces_cero, energia_rel_0_1, energia_rel_1_4, energia_rel_4_8




from scipy.stats import entropy as shannon_entropy

def features_eeg(segmento, fs):
    """
    Extrae features espectrales, temporales y de entropía de un segmento EEG.

    Parámetros:
        segmento : array_like
            Señal EEG (1D) ya filtrada.
        fs : int
            Frecuencia de muestreo.

    Devuelve:
        varianza, cruces_cero, proporciones, relaciones, entropia_shannon
    """
    x = np.asarray(segmento)
    N = len(x)

    # --- Temporales
    varianza = np.var(x)
    cruces_cero = np.sum((x[:-1] * x[1:]) < 0)

    # --- Welch
    f, Pxx = welch(x, fs=fs, nperseg=min(N, fs*4), noverlap=fs//2)
    energia_total = np.trapz(Pxx, f)

    bandas = {
        'delta': (0.5, 4),
        'theta_baja': (4, 6),
        'theta_alta': (6, 8),
        'alfa': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45),
        'theta_total': (4, 8)
    }

    proporciones = {}
    for nombre, (f_low, f_high) in bandas.items():
        idx = (f >= f_low) & (f < f_high)
        energia_banda = np.trapz(Pxx[idx], f[idx])
        proporciones[nombre] = energia_banda / energia_total if energia_total > 0 else np.nan

    # --- Relaciones espectrales
    rel_alfa_theta = proporciones['alfa'] / proporciones['theta_total'] if proporciones['theta_total'] > 0 else np.nan
    rel_theta_delta = proporciones['theta_total'] / proporciones['delta'] if proporciones['delta'] > 0 else np.nan
    rel_beta_alfa = proporciones['beta'] / proporciones['alfa'] if proporciones['alfa'] > 0 else np.nan
    rel_alfa_lento = proporciones['alfa'] / (proporciones['delta'] + proporciones['theta_total']) if (proporciones['delta'] + proporciones['theta_total']) > 0 else np.nan

    relaciones = {
        'alfa_theta': rel_alfa_theta,
        'theta_delta': rel_theta_delta,
        'beta_alfa': rel_beta_alfa,
        'alfa_lento': rel_alfa_lento
    }

    # --- Entropía de Shannon
    hist, bin_edges = np.histogram(x, bins=100, density=True)
    hist = hist[hist > 0]  # evitamos log(0)
    entropia_shannon = shannon_entropy(hist, base=2)  # bits

    return varianza, cruces_cero, proporciones, relaciones, entropia_shannon

