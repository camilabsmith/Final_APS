import os
import json
import random
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    classification_report, confusion_matrix
)

# === Configuración ===
random_seed = 15
ruta_base = Path(__file__).parent.resolve()
ruta_csvs = ruta_base / "DOD-H-features"
ruta_resultados = ruta_base / "Resultados_Folds_balanced"
ruta_resultados.mkdir(parents=True, exist_ok=True)

# === Leer y dividir archivos ===
archivos = sorted([a for a in os.listdir(ruta_csvs) if a.endswith(".csv")])
random.Random(random_seed).shuffle(archivos)
grupos = [archivos[i::5] for i in range(5)]

# === Cargar datos ===
def cargar_datos(lista_archivos):
    dfs = []
    for archivo in lista_archivos:
        df = pd.read_csv(ruta_csvs / archivo)
        df = df[df["etiqueta_0"] != -1].copy()
        df["etiqueta_final"] = df["etiqueta_0"]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# === Loop por fold ===
etiquetas_dict = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
colores_modelos = {'RandomForest': 'red', 'LogisticRegression': 'green', 'RidgeClassifier': 'blue'}

for i, test_files in enumerate(grupos):
    print(f"Procesando FOLD {i + 1}...")
    fold_dir = ruta_resultados / f"FOLD_{i+1}"
    fold_dir.mkdir(exist_ok=True)

    train_files = [f for j, g in enumerate(grupos) if j != i for f in g]
    df_train = cargar_datos(train_files)
    df_test = cargar_datos(test_files)

    columnas_excluir = ['sujeto', 't_inicio', 't_fin'] + [f'etiqueta_{j}' for j in range(6)] + ['consenso', 'etiqueta_final']
    features = [c for c in df_train.columns if c not in columnas_excluir and df_train[c].dtype != 'object']

    X_train = df_train[features]
    y_train = df_train["etiqueta_final"]
    X_test = df_test[features]
    y_test = df_test["etiqueta_final"]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    modelos = {
        "RandomForest": RandomForestClassifier(random_state=0, class_weight='balanced'),
        "LogisticRegression": LogisticRegression(max_iter=500, class_weight='balanced'),
        "RidgeClassifier": RidgeClassifier(class_weight='balanced')
    }

    pred_df = pd.DataFrame(index=df_test.index)
    pred_df['sujeto'] = df_test['sujeto']
    pred_df['etiqueta'] = y_test.values
    metrics_json = {'test_files': test_files}

    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        pred_df[nombre] = y_pred

        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred, average='macro')
        pre = precision_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        metrics_json[nombre] = {
            'accuracy': acc,
            'recall': rec,
            'precision': pre,
            'f1': f1
        }

        # === Importancia de features ===
        if nombre == "RandomForest":
            importancias = modelo.feature_importances_
            df_feat = pd.DataFrame({'feature': features, 'importance': importancias})
            df_feat.to_csv(fold_dir / f"importancia_{nombre}.csv", index=False)
        elif nombre == "RidgeClassifier":
            importancias = np.abs(modelo.coef_).mean(axis=0)
            df_feat = pd.DataFrame({'feature': features, 'importance': importancias})
            df_feat.to_csv(fold_dir / f"importancia_{nombre}.csv", index=False)
        elif nombre == "LogisticRegression":
            for clase_idx, clase_nombre in etiquetas_dict.items():
                importancias = modelo.coef_[clase_idx]
                df_feat = pd.DataFrame({'feature': features, 'importance': importancias})
                df_feat.to_csv(fold_dir / f"importancia_{nombre}_{clase_nombre}.csv", index=False)

        # === Matriz de confusión general ===
        cm = confusion_matrix(y_test, y_pred)
        cm_norm = confusion_matrix(y_test, y_pred, normalize='true')

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Spectral', xticklabels=etiquetas_dict.values(), yticklabels=etiquetas_dict.values(), ax=ax)
        ax.set_title(f"Confusion Matrix - {nombre}")
        plt.savefig(fold_dir / f"{nombre}_conf_matrix.png")
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Spectral', xticklabels=etiquetas_dict.values(), yticklabels=etiquetas_dict.values(), ax=ax)
        ax.set_title(f"Normalized Confusion Matrix - {nombre}")
        plt.savefig(fold_dir / f"{nombre}_conf_matrix_normalized.png")
        plt.close()

        # === Matrices por sujeto ===
        sujetos_unicos = df_test['sujeto'].unique()
        fig, axs = plt.subplots(1, len(sujetos_unicos), figsize=(5 * len(sujetos_unicos), 5))
        fig_norm, axs_norm = plt.subplots(1, len(sujetos_unicos), figsize=(5 * len(sujetos_unicos), 5))

        for j, sujeto in enumerate(sujetos_unicos):
            idx = df_test['sujeto'] == sujeto
            y_real = y_test[idx]
            y_p = y_pred[idx]
            cm_sub = confusion_matrix(y_real, y_p)
            cm_sub_norm = confusion_matrix(y_real, y_p, normalize='true')

            sns.heatmap(cm_sub, annot=True, fmt='d', cmap='Spectral', xticklabels=etiquetas_dict.values(), yticklabels=etiquetas_dict.values(), ax=axs[j])
            axs[j].set_title(sujeto)

            sns.heatmap(cm_sub_norm, annot=True, fmt='.2f', cmap='Spectral', xticklabels=etiquetas_dict.values(), yticklabels=etiquetas_dict.values(), ax=axs_norm[j])
            axs_norm[j].set_title(sujeto)

        fig.suptitle(f"Confusion Matrices - {nombre}")
        fig_norm.suptitle(f"Normalized Confusion Matrices - {nombre}")
        fig.tight_layout()
        fig.savefig(fold_dir / f"{nombre}_matrices_por_sujeto.png")
        plt.close(fig)
        fig_norm.tight_layout()
        fig_norm.savefig(fold_dir / f"{nombre}_matrices_por_sujeto_norm.png")
        plt.close(fig_norm)

        # === Hipnogramas ===
        # === Hipnogramas ===
        for sujeto in sujetos_unicos:
            df_s = df_test[df_test['sujeto'] == sujeto]
            t = np.arange(len(df_s)) * 30 / 3600  # tiempo en horas
        
            plt.figure(figsize=(14, 5))
            plt.plot(t, df_s["etiqueta_final"], color='black', label='Real')
            plt.plot(t, pred_df.loc[df_s.index, nombre], label=nombre, color=colores_modelos[nombre], alpha=0.6)
            plt.yticks(ticks=range(5), labels=[etiquetas_dict[i] for i in range(5)])
            plt.gca().invert_yaxis()  # <- esto invierte el eje Y
            plt.xlabel("Tiempo [h]")
            plt.title(f"Hipnograma - {nombre} - {sujeto}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fold_dir / f"hipnograma_{nombre}_{sujeto}.png")
            plt.close()



    pred_df.to_csv(fold_dir / "predicciones.csv", index=False)
    with open(fold_dir / "resultados.json", 'w') as f:
        json.dump(metrics_json, f, indent=4)
