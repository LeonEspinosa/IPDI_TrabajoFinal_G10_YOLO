import cv2
import numpy as np
from skimage.morphology import skeletonize
from .basicos import calcular_diametro_feret

def obtener_esqueleto(contorno, image_shape):
    """
    Genera el esqueleto topológico del objeto mediante adelgazamiento (thinning).
    Convierte la forma de la naranja en una línea de 1 píxel de ancho que representa su estructura.
    """
    # 1. Crear una máscara binaria vacía basada en las dimensiones de la imagen original
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 2. Rellenar el contorno (Objeto = 255, Fondo = 0)
    cv2.drawContours(mask, [contorno], -1, 255, thickness=cv2.FILLED)
    
    # 3. Convertir a booleano (True/False) para que scikit-image lo entienda
    binary_mask = mask > 0
    
    # 4. Aplicar algoritmo de esqueletonización (reduce la forma a su eje medial)
    skeleton = skeletonize(binary_mask)
    
    # Retornar como imagen uint8 (0 y 1)
    return skeleton.astype(np.uint8)

def calcular_longitud_fibra(contorno, image_shape):
    """
    Calcula la longitud de la fibra (Fiber Length).
    Se define como la suma de píxeles del esqueleto del objeto.
    """
    esqueleto = obtener_esqueleto(contorno, image_shape)
    
    # La longitud es la suma de los píxeles activos del esqueleto
    longitud = np.sum(esqueleto)
    
    return float(longitud)

def calcular_curl(contorno, image_shape):
    """
    Calcula el índice de Curl (Rizo).
    Según PDF Pág 15: Curl = Length / Fiber Length
    
    - 'Length': Usamos el Diámetro de Feret (distancia máxima en línea recta).
    - 'Fiber Length': Longitud del esqueleto (distancia recorriendo la curva del objeto).
    
    Interpretación:
    - Valor cercano a 1.0: El objeto es recto o convexo (como una zanahoria recta).
    - Valor < 1.0: El objeto está curvado o "rizado" (como un plátano o una hoja doblada).
    """
    length_recta = calcular_diametro_feret(contorno)
    longitud_fibra = calcular_longitud_fibra(contorno, image_shape)
    
    if longitud_fibra <= 0:
        return 0.0
        
    curl = length_recta / longitud_fibra
    return float(curl)