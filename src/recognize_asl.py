import cv2
import mediapipe as mp
import numpy as np
import joblib

from collections import deque, Counter
from features import agregar_features
from pathlib import Path

# Ruta principal del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "model" / "asl_random_forest.pkl"

# Cargar modelo entrenado
modelo = joblib.load(MODEL_PATH)

# Guarda las últimas 10 predicciones
historial = deque(maxlen=10)


def extraer_features(hand_landmarks):
    puntos = []

    for lm in hand_landmarks.landmark:
        puntos.append([
            lm.x,
            lm.y,
            lm.z
        ])

    puntos = np.array(puntos)

    # Centrar la mano usando la muñeca como origen
    puntos -= puntos[0]

    # Normalizar tamaño de la mano
    escala = np.linalg.norm(puntos[9])

    if escala > 0:
        puntos /= escala

    # 21 puntos × 3 coordenadas = 63 features
    return puntos.flatten()


# --- MediaPipe ---
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils


# Cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    cap = cv2.VideoCapture(1)

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    # Cámara espejo
    frame = cv2.flip(frame, 1)

    # OpenCV BGR -> RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)


    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Dibujar landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Sacar los 63 features
            features_63 = extraer_features(
                hand_landmarks
            )

            features = agregar_features(
                features_63
            )

            # Obtener probabilidades del modelo
            probs = modelo.predict_proba(
                [features]
            )[0]

            confianza = max(probs)

            letra = modelo.classes_[
                np.argmax(probs)
            ]


            # Solo aceptar predicciones razonablemente seguras
            if confianza > 0.75:

                historial.append(letra)

                letra_estable = Counter(
                    historial
                ).most_common(1)[0][0]

                cv2.putText(
                    frame,
                    f"Letra: {letra_estable} ({confianza:.0%})",
                    (50, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3,
                    (0, 0, 225),
                    3
                )

            else:
                # Evita que una letra vieja permanezca
                # demasiado tiempo en memoria
                historial.clear()

    else:
        # Si desaparece la mano, borramos predicciones anteriores
        historial.clear()


    cv2.imshow(
        "ASL Recognition",
        frame
    )


    # ESC para salir
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
hands.close()
cv2.destroyAllWindows()