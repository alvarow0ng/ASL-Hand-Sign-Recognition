import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

from features import agregar_features


# --------------------------------------------------
# RUTAS DEL PROYECTO
# --------------------------------------------------

# validate_model.py está dentro de src/
# parent = src/
# parent.parent = carpeta principal del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "asl_validation.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "asl_random_forest.pkl"
RESULT_PATH = PROJECT_ROOT / "results" / "confusion_matrix_validation.png"


# --------------------------------------------------
# CARGAR MODELO
# --------------------------------------------------

modelo = joblib.load(MODEL_PATH)


# --------------------------------------------------
# CARGAR DATASET DE VALIDACIÓN
# --------------------------------------------------

datos = pd.read_csv(
    DATA_PATH,
    header=None
)


# Primera columna = letra correcta
y_real = datos.iloc[:, 0]

# Columnas restantes = 63 features originales
x_original = datos.iloc[:, 1:].to_numpy(
    dtype=float
)


# --------------------------------------------------
# FEATURE ENGINEERING
# 63 features -> 74 features
# --------------------------------------------------

x_validacion = np.array([
    agregar_features(fila)
    for fila in x_original
])


print(
    "Features de validacion:",
    x_validacion.shape[1]
)


# --------------------------------------------------
# PREDICCIONES
# --------------------------------------------------

predicciones = modelo.predict(
    x_validacion
)


# --------------------------------------------------
# ACCURACY
# --------------------------------------------------

accuracy = accuracy_score(
    y_real,
    predicciones
)

print(
    f"Precision de validacion: {accuracy:.2%}"
)


# --------------------------------------------------
# REPORTE DE CLASIFICACIÓN
# --------------------------------------------------

print("\nReporte de clasificacion:\n")

print(
    classification_report(
        y_real,
        predicciones
    )
)


# --------------------------------------------------
# MATRIZ DE CONFUSIÓN
# --------------------------------------------------

matriz = confusion_matrix(
    y_real,
    predicciones,
    labels=modelo.classes_
)


display = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=modelo.classes_
)

display.plot()

plt.title(
    "Independent Validation Confusion Matrix"
)

plt.tight_layout()


# Guardar imagen dentro de results/
plt.savefig(
    RESULT_PATH,
    dpi=200,
    bbox_inches="tight"
)


plt.show()