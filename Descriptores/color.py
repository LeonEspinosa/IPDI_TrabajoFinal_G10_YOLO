import cv2
import numpy as np

def calcular_media_color(imagen_bgr, mascara=None):
    """
    Calcula la media de color BGR de la imagen.
    Si se proporciona máscara, solo considera los píxeles activos.
    """
    if imagen_bgr is None:
        return 0.0, 0.0, 0.0
        
    media = cv2.mean(imagen_bgr, mask=mascara)[:3]
    # Retorna (B, G, R)
    return float(media[0]), float(media[1]), float(media[2])
