import h5py
import os
import pandas as pd
import json
from utils_info import (
    filtrado, segmentar, features_eeg, features_eog, detector_k_complex
)




from pathlib import Path
ruta_base = Path(__file__).parent.resolve()

ruta_datos = ruta_base / "DOD-H"
ruta_scores = ruta_base / "scores" / "dodh"
ruta_salida = ruta_base / "DOD-H-features"


ruta_salida.mkdir(parents=True, exist_ok=True)


dur_seg = 30
fpass_eeg = (0.5, 45)
fstop_eeg = (0.1, 50)
fpass_eog = (0.3, 40)
fstop_eog = (0.1, 50)
ripple = 0.5
attenuation = 20

# === PROCESAMIENTO DE TODOS LOS ARCHIVOS H5 ===
for archivo in sorted(os.listdir(ruta_datos)):
    if not archivo.endswith(".h5"):
        continue

    path_archivo = os.path.join(ruta_datos, archivo)
    sujeto = archivo.replace(".h5", "")
    nombre_salida = f"features_{sujeto}.csv"
    df_total = pd.DataFrame()

    print(f"\nProcesando archivo: {sujeto}")

    with h5py.File(path_archivo, 'r') as f:
        fs = f['signals']['eeg'].attrs['fs']
        try:
            hipnograma = f['hypnogram'][:]
        except KeyError:
            print(f" No se encontró el hipnograma en {archivo}")
            continue

        # EEG
        grupo_eeg = f['signals']['eeg']
        nombres_canales = list(grupo_eeg.keys())

        señales_filtradas = {}
        segmentos_por_canal = {}
        for canal in nombres_canales:
            señal = grupo_eeg[canal][()]
            señal_filtrada = filtrado(señal, fs, fpass_eeg, ripple, fstop_eeg, attenuation)
            señales_filtradas[canal] = señal_filtrada
            segmentos_por_canal[canal] = segmentar(señal_filtrada, fs, dur_seg)

        # EOG
        grupo_eog = f['signals']['eog']
        nombres_canales_eog = list(grupo_eog.keys())

        señales_filtradas_eog = {}
        segmentos_por_canal_eog = {}
        for canal in nombres_canales_eog:
            señal = grupo_eog[canal][()]
            señal_filtrada = filtrado(señal, fs, fpass_eog, ripple, fstop_eog, attenuation)
            señales_filtradas_eog[canal] = señal_filtrada
            segmentos_por_canal_eog[canal] = segmentar(señal_filtrada, fs, dur_seg)

        # Número válido de segmentos
        num_segmentos = min(len(v) for v in segmentos_por_canal.values())
        num_segmentos = min(num_segmentos, len(hipnograma))

        # Cargar etiquetas de los scorers
        etiquetas_scorers = {}
        for i in range(1, 6):
            path_json = os.path.join(ruta_scores, f"scorer_{i}", archivo.replace(".h5", ".json"))
            with open(path_json, 'r') as f_json:
                etiquetas = json.load(f_json)
                if len(etiquetas) < num_segmentos:
                    raise ValueError(f"Scorer {i} tiene menos etiquetas ({len(etiquetas)}) que los segmentos disponibles ({num_segmentos}) en {archivo}")
                etiquetas_scorers[f"etiqueta_{i}"] = etiquetas

        # Info general
        for i in range(num_segmentos):
            t_ini = i * dur_seg
            t_fin = t_ini + dur_seg

            fila = {
                "sujeto": sujeto,
                "t_inicio": t_ini,
                "t_fin": t_fin,
                "etiqueta_0": int(hipnograma[i])
            } # Scores
            for j in range(1, 6):
                fila[f"etiqueta_{j}"] = int(etiquetas_scorers[f"etiqueta_{j}"][i])

            # === EEG ===
            for canal in nombres_canales:
                segmento = segmentos_por_canal[canal][i]
                varianza, cruces, proporciones, relaciones, entropia = features_eeg(segmento, fs)

                fila[f"{canal}_varianza"] = varianza
                fila[f"{canal}_cruces_cero"] = cruces
                fila[f"{canal}_p_delta"] = proporciones["delta"]
                fila[f"{canal}_p_theta_baja"] = proporciones["theta_baja"]
                fila[f"{canal}_p_theta_alta"] = proporciones["theta_alta"]
                fila[f"{canal}_p_theta_total"] = proporciones["theta_total"]
                fila[f"{canal}_p_alfa"] = proporciones["alfa"]
                fila[f"{canal}_p_beta"] = proporciones["beta"]
                fila[f"{canal}_p_gamma"] = proporciones["gamma"]
                fila[f"{canal}_rel_alfa_theta"] = relaciones["alfa_theta"]
                fila[f"{canal}_rel_theta_delta"] = relaciones["theta_delta"]
                fila[f"{canal}_rel_beta_alfa"] = relaciones["beta_alfa"]
                fila[f"{canal}_rel_alfa_lento"] = relaciones["alfa_lento"]
                fila[f"{canal}_entropia_shannon"] = entropia
                
                ## COMPLEJOS K
                cant_k = detector_k_complex(fs, segmento)
                fila[f"{canal}_kcomplexes"] = cant_k

            # === EOG ===
            for canal in nombres_canales_eog:
                segmento = segmentos_por_canal_eog[canal][i]
                n_picos, media_pos, media_neg, varianza, cruces, p01, p14, p48 = features_eog(segmento, fs)

                fila[f"{canal}_varianza"] = varianza
                fila[f"{canal}_cruces_cero"] = cruces
                fila[f"{canal}_num_picos"] = n_picos
                fila[f"{canal}_media_picos_pos"] = media_pos
                fila[f"{canal}_media_picos_neg"] = media_neg
                fila[f"{canal}_p_0_1Hz"] = p01
                fila[f"{canal}_p_1_4Hz"] = p14
                fila[f"{canal}_p_4_8Hz"] = p48

            df_total = pd.concat([df_total, pd.DataFrame([fila])], ignore_index=True)

    # Guardar CSV por archivo
    
    df_total.to_csv(os.path.join(ruta_salida, nombre_salida), index=False)
    print(" Guardado:", nombre_salida)


