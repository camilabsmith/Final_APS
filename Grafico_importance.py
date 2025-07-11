import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# === Rutas base ===
base_balanced = Path("Resultados_Folds_balanced/FOLD_5")
base_nobal = Path("Resultados_Folds_no_balance/FOLD_5")

# === Etiquetas de sueño (clases de logistic regression)
etiquetas = ['Wake', 'N1', 'N2', 'N3', 'REM']

# === Función para obtener tipo de feature correctamente
def obtener_tipo_feature(nombre_feature):
    if nombre_feature.startswith("EOG"):
        return "EOG_" + nombre_feature.split("_", 1)[1]
    else:
        partes = nombre_feature.split("_")
        if len(partes) >= 3:
            return "_".join(partes[2:])
        else:
            return "otro"

# === RANDOM FOREST ===
df_rf = pd.read_csv(base_nobal / "importancia_RandomForest.csv")
df_rf["tipo"] = df_rf["feature"].apply(obtener_tipo_feature)
df_rf["importance_abs"] = df_rf["importance"].abs()

df_tipo_rf = df_rf.groupby("tipo")["importance_abs"].sum().reset_index()
df_tipo_rf = df_tipo_rf.sort_values("importance_abs", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=df_tipo_rf, y="tipo", x="importance_abs", palette="Reds")
plt.title("Importancia por tipo de feature - Random Forest")
plt.xlabel("Importancia absoluta total")
plt.ylabel("Tipo de feature")
plt.tight_layout()
plt.savefig("importancia_por_tipo_randomforest.png")
plt.close()

# === RIDGE CLASSIFIER ===
df_ridge = pd.read_csv(base_balanced / "importancia_RidgeClassifier.csv")
df_ridge["tipo"] = df_ridge["feature"].apply(obtener_tipo_feature)
df_ridge["importance_abs"] = df_ridge["importance"].abs()

df_tipo_ridge = df_ridge.groupby("tipo")["importance_abs"].sum().reset_index()
df_tipo_ridge = df_tipo_ridge.sort_values("importance_abs", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=df_tipo_ridge, y="tipo", x="importance_abs", palette="Blues")
plt.title("Importancia por tipo de feature - Ridge Classifier")
plt.xlabel("Importancia absoluta total")
plt.ylabel("Tipo de feature")
plt.tight_layout()
plt.savefig("importancia_por_tipo_ridgeclassifier.png")
plt.close()

# === LOGISTIC REGRESSION: por clase ===
for clase in etiquetas:
    archivo = base_balanced / f"importancia_LogisticRegression_{clase}.csv"
    if archivo.exists():
        df_log = pd.read_csv(archivo)
        df_log["importance_abs"] = df_log["importance"].abs()
        df_log["tipo"] = df_log["feature"].apply(obtener_tipo_feature)

        df_tipo = df_log.groupby("tipo")["importance_abs"].sum().reset_index()
        df_tipo = df_tipo.sort_values("importance_abs", ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_tipo, y="tipo", x="importance_abs", palette="Greens")
        plt.title(f"Importancia por tipo de feature - Regresión Logística - Clase {clase}")
        plt.xlabel("Importancia absoluta total")
        plt.ylabel("Tipo de feature")
        plt.tight_layout()
        plt.savefig(f"importancia_por_tipo_logreg_{clase}.png")
        plt.close()
