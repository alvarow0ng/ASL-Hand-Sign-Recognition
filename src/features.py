import numpy as np

def agregar_features(features_63):

    #Regresar 63 números a 21 puntos x 3 coordenadas
    puntos = np.array(features_63).reshape(21, 3)

    features_extra = []

    def distancia(p1, p2):
        return np.linalg.norm(
            puntos[p1] - puntos[p2]
        )
    
    # Pulgar con otros dedos
    features_extra.append(distancia(4, 8))
    features_extra.append(distancia(4, 12))
    features_extra.append(distancia(4, 16))
    features_extra.append(distancia(4, 20))

    # Distancia entre puntas de dedos
    features_extra.append(distancia(8, 12))
    features_extra.append(distancia(12, 16))
    features_extra.append(distancia(16, 20))

    # Puntas respecto a muñeca
    features_extra.append(distancia(0, 8))
    features_extra.append(distancia(0, 12))
    features_extra.append(distancia(0, 16))
    features_extra.append(distancia(0, 20))

    return np.concatenate([
        features_63,
        features_extra
    ])