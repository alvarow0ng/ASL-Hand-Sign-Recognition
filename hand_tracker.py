import cv2
import mediapipe as mp
import math
from collections import deque

#Calcula la distancia entre punto 1 y punto 2 en la cámara
def calcular_distancia(p1, p2):
    #La fórmula aplicada a los landmarks de MediaPipe
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

#Guarda las últimas 20 posiciones en el eje X desde el centro de la mano (para mayor seguridad)
memoria_movimiento = deque(maxlen=20)

#Configurar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode = False, max_num_hands = 4, min_detection_confidence = 0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("--- MEDIAPIPE ACTIVO ---")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    try:
        #Se voltea la camara para que sea espejo
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if rgb_frame.shape[2] != 3:
            continue

        #Procesa la imagen
        results = hands.process(rgb_frame)
    except Exception as e:
        print(f"Error en la conversion: {e}")
        continue

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            #Dibuja puntos de dedos
            mp_draw.draw_landmarks(frame, hand_landmarks,mp_hands.HAND_CONNECTIONS)

            #PUNTO DE REFERENCIA:
            # 8: Indice, 12: Medio, 16: Anular, 20: Meñique, 4: Pulgar

            #ÍNDICE ARRIBA
            indice_arriba = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y

            #DEDO MEDIO ABAJO
            medio_abajo = hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y

            #DEDO ANULAR ABAJO
            anular_abajo = hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y

            #DEDO MEÑIQUE ABAJO
            meñique_abajo = hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y

            #PULGAR CERRADO
            pulgar_abajo = hand_landmarks.landmark[4].x > hand_landmarks.landmark[5].x

            #SI EL INDICE ESTA ARRIBA
            if indice_arriba and medio_abajo and anular_abajo and meñique_abajo:
                cv2.putText(frame, "INDEX UP - NUMBER 1", (50, 50), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0,255,0), 2)

            #Usamos el punto 0 (muñeca) para rastrear el movimiento completo
            pos_x = hand_landmarks.landmark[0].x
            memoria_movimiento.append(pos_x)

            # LÓGICA DE GESTO: ¿Dedo índice arriba?
            # FORMA #1
            # Punto 8 es la punta del índice, punto 6 es el nudillo
            #y_punta = hand_landmarks.landmark[8].y
            #y_nudillo = hand_landmarks.landmark[6].y

            # FORMA #2
            dist_dedo = calcular_distancia(hand_landmarks.landmark[8],hand_landmarks.landmark[5])
            dist_referencia = calcular_distancia(hand_landmarks.landmark[6],hand_landmarks.landmark[5])

            if dist_dedo > dist_referencia:
                cv2.putText(frame, "INDICE LEVANTADO - ACCION!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            #LÓGICA DE GRABACIÓN
            if len(memoria_movimiento) == 20:
                #Si el primer movimiento está muy a la izquierda (0.2)
                #y el último movimiento muy a la derecha. Realizo un SWIPE ó DESLIZAR
                inicio = memoria_movimiento[0]
                fin = memoria_movimiento[-1]

                dist_muñeca = fin - inicio
                if abs(dist_muñeca) > 0.5:              #Revisa que tenga movimiento necesario para DESLIZAR
                    if dist_muñeca > 0:                 #Desliza hacia la derecha
                        print("Gesto hacia la derecha")
                        memoria_movimiento.clear()
                    else:                               #Desliza hacia la izquierda
                        print("Gesto hacia la izquierda")
                        memoria_movimiento.clear()

    cv2.imshow("MediaPipe Gestures", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()