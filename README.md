# Clasificación automática de etapas del sueño a partir de EEG y EOG

Repositorio para el trabajo final de "Análisis y Procesamiento de Señales" – UNSAM  

Autora: 🌸🌼🌺🌻🌷🌸🌼 Camila B. Smith  🌸🌼🌺🌻🌷🌸🌼🌺🌻 

GitHub: [https://github.com/camilabsmith/Final_APS](https://github.com/camilabsmith/Final_APS)

---

## Descripción general

Este repositorio implementa un pipeline completo para la **clasificación automática de las etapas del sueño** usando señales EEG y EOG del dataset abierto DOD-H (Dreem Organization).  
Incluye preprocesamiento, extracción de características, entrenamiento y evaluación de modelos supervisados, y análisis visual exploratorio.

---

## Requisitos de librerías

```bash
pip install numpy pandas matplotlib seaborn scikit-learn h5py pywt boto3 pytc2
``` 
--- 
## Orden de ejecución y funcionalidad de los scripts
### 1. Descarga del dataset
```bash
Descarga_dataset.py
```
- Descarga automática de los archivos .h5 originales del DOD-H a la carpeta local DOD-H/.

- Solo es necesario ejecutar este script una vez.

- Genera: archivos .h5 en DOD-H/

### 2. Extracción de características
```bash
extraccion_feat.py
```
- Lee los archivos .h5, aplica filtrado digital y segmentación, y extrae características temporales y espectrales de EEG/EOG para cada ventana de 30 s.

- Genera: un archivo .csv de features por sujeto en DOD-H-features/

### 3. Entrenamiento y evaluación de modelos
```bash
entrenamiento_modelos.py
```
- Realiza validación cruzada (5 folds), entrena modelos (Random Forest, Logistic Regression, RidgeClassifier), calcula métricas, y guarda resultados por fold.

Genera:
- Resultados y métricas en Resultados_Folds_balanced/

- Figuras de matrices de confusión y de importancia de features

- Predicciones por sujeto

### 4. Análisis exploratorio individual
```bash
Individual.py
```
- Permite analizar visualmente un sujeto:

- Visualización y filtrado de señales

- Gráficos de espectro, hipnograma y señales temporales

- Detección y visualización de complejos K

- Útil para validar visualmente el preprocesamiento y la extracción de características

- No genera archivos permanentes; produce gráficos exploratorios

### 5. Ploteo de hipnogramas de predicciones
```bash
plot_hipnograma.py
```
- Genera, para cada sujeto, un gráfico comparando el hipnograma real (etiqueta) vs. la predicción del modelo Random Forest.

- Genera: imágenes PNG por sujeto en Resultados_Folds_no_balance/FOLD_5/hipnogramas_por_sujeto/

### 6. Utilidades y funciones auxiliares
```bash
utils_info.py
```
- Contiene todas las funciones auxiliares usadas por los otros scripts: filtrado, segmentación, extracción de features, detección de complejos K, etc.

- No se ejecuta directamente.

## Estructura esperada de carpetas
```bash
Final_APS/
│
├── Descarga_dataset.py
├── extraccion_feat.py
├── entrenamiento_modelos.py
├── Individual.py
├── plot_hipnograma.py
├── utils_info.py
│
├── DOD-H/                    # (se crea tras descargar datos)
├── DOD-H-features/           # (csv de features, se crea tras extracción)
├── Resultados_Folds_balanced/    # (resultados y métricas por fold)
├── Resultados_Folds_no_balance/  # (otros resultados)
│   └── FOLD_5/
│       └── hipnogramas_por_sujeto/
└── scores/                   # (etiquetas json de los técnicos)
```

---

🌸🌼🌺🌻🌷🌸🌼🌺🌻🌷🌸🌼🌺🌻🌷  
**¡Gracias por visitar este repositorio!**  
🌸🌼🌺🌻🌷🌸🌼🌺🌻🌷🌸🌼🌺🌻🌷

---

