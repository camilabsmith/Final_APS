"""

El script asume que se ejecuta **dentro de la carpeta “Final APS”**.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


script_dir = Path(__file__).resolve().parent        
fold_dir   = script_dir / "Resultados_Folds_no_balance" / "FOLD_5"
csv_path   = fold_dir / "predicciones.csv"
out_dir    = fold_dir / "hipnogramas_por_sujeto"
out_dir.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(csv_path)

# remapeo para el eje
label_to_y = {0: 5, 4: 4, 1: 3, 2: 2, 3: 1}

# --- Hipnograma por sujeto ---
for sujeto, g in df.groupby("sujeto", sort=False):
    y_true = g["etiqueta"].map(label_to_y).reset_index(drop=True)
    y_pred = g["RandomForest"].map(label_to_y).reset_index(drop=True)
    x      = range(len(g))

    plt.figure(figsize=(12, 3))
    plt.step(x, y_true, where="post", color="k", linewidth=1, label="Etiqueta")
    plt.step(x, y_pred, where="post", color="r", linewidth=1, label="RandomForest", alpha=0.5)

    plt.yticks([5, 4, 3, 2, 1], ["Wake", "REM", "N1", "N2", "N3"])
    plt.ylim(0.5, 5.5)
    plt.xlabel("Época (30 s)")
    plt.title(f"Hipnograma – Sujeto {sujeto}")
    plt.legend(loc="upper right")
    plt.tight_layout()

    # Guardar imagen
    plt.savefig(out_dir / f"hipnograma_{sujeto}.png", dpi=300)
    plt.close()
