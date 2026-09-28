import csv
import cv2
import mediapipe as mp
import numpy as np
import time

def extraer_features(hand_landmarks):
    puntos = []

    for lm in hand_landmarks.landmark:
        puntos.append([
            lm.x,
            lm.y,
            lm.z
        ])
    
    puntos = np.array(puntos)

    # Muñeca como origen
    puntos -= puntos[0]

    # Normalizar tamaño
    escala = np.linalg.norm(puntos[9])

    if escala > 0:
        puntos /= escala
    
    return puntos.flatten()

# --- MEDIAPIPE ---
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode = False,
    max_num_hands = 1,
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)

archivo_csv = "asl_data.csv"

# --- VARIABLES ---
MUESTRA_POR_LETRA = 50
INTERVALO_MUESTRA = 0.2
LETRAS = ["A", "B", "C", "D", "E", "F", "I", "L", "O", "Y"]
letra_actual = None
contador = 0
ultimo_guardado = 0


while True:
    features_actuales = None

    ret, frame = cap.read()

    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame, cv2.COLOR_BGR2RGB
        )
    
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            features = extraer_features(
                hand_landmarks
            )

            features_actuales = features

            if letra_actual is not None:

                tiempo_actual = time.time()

                if tiempo_actual - ultimo_guardado >= INTERVALO_MUESTRA:
                
                    with open(archivo_csv, "a", newline = "") as archivo:
                        escritor = csv.writer(archivo)
                        fila = [letra_actual] + features_actuales.tolist()
                        escritor.writerow(fila)
                
                    contador += 1
                    ultimo_guardado = tiempo_actual

                print(
                    f"{letra_actual}: "
                    f"{contador}/{MUESTRA_POR_LETRA}"
                )

                if contador >= MUESTRA_POR_LETRA:

                    print(f"Recoleccion de {letra_actual} terminada")
                    letra_actual = None

    cv2.imshow(
        "Recolector ASL",
        frame
    )

    tecla = cv2.waitKey(1) & 0xFF

    # Termina programa al presionar ESC
    if tecla == 27:
        break

    if tecla != 255:
        
        caracter = chr(tecla).upper()

        # Guardar muestra de A
        if caracter in LETRAS:
            letra_actual = caracter
            contador = 0
            ultimo_guardado = 0

            print(f"Comenzando recoleccion de {letra_actual}")

cap.release()
hands.close()
cv2.destroyAllWindows()