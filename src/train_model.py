import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from features import agregar_features


# --------------------------------------------------
# RUTAS DEL PROYECTO
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "asl_data.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "asl_random_forest.pkl"
RESULT_PATH = PROJECT_ROOT / "results" / "confusion_matrix_internal.png"


# --------------------------------------------------
# CARGAR DATOS
# --------------------------------------------------

datos = pd.read_csv(
    DATA_PATH,
    header=None
)

y = datos.iloc[:, 0]

x_original = datos.iloc[:, 1:].to_numpy(
    dtype=float
)


# --------------------------------------------------
# FEATURE ENGINEERING
# 63 features -> 74 features
# --------------------------------------------------

x = np.array([
    agregar_features(fila)
    for fila in x_original
])


print(
    "Features originales:",
    x_original.shape[1]
)

print(
    "Features V2:",
    x.shape[1]
)


# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# MODELO
# --------------------------------------------------

modelo = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


# Entrenamiento
modelo.fit(
    x_train,
    y_train
)


# --------------------------------------------------
# PREDICCIONES
# --------------------------------------------------

predicciones = modelo.predict(
    x_test
)


# Accuracy
accuracy = accuracy_score(
    y_test,
    predicciones
)

print(
    f"Precision interna V2: {accuracy:.2%}"
)


# --------------------------------------------------
# MATRIZ DE CONFUSIÓN
# --------------------------------------------------

matriz = confusion_matrix(
    y_test,
    predicciones,
    labels=modelo.classes_
)


display = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=modelo.classes_
)

display.plot()

plt.title(
    "Internal Test Confusion Matrix"
)

plt.tight_layout()


# Guardar imagen
plt.savefig(
    RESULT_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# --------------------------------------------------
# GUARDAR MODELO ENTRENADO
# --------------------------------------------------

joblib.dump(
    modelo,
    MODEL_PATH
)