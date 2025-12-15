import cv2
import numpy as np
import math
from .basicos import calcular_area, calcular_diametro_feret
from .topologicos import obtener_esqueleto, calcular_longitud_fibra

def elongacion_criterio_fibra(mask_binaria, contorno):
    """
    Calcula Elongación = Longitud Fibra / Ancho Fibra
    
    - Longitud Fibra: Largo del esqueleto.
    - Ancho Fibra: Promedio de la Transformada de Distancia sobre los puntos del esqueleto.
      (Esto nos dice qué tan "grueso" es el objeto en promedio a lo largo de su eje).
    """
    if mask_binaria is None: return 0.0
    
    # 1. Longitud de Fibra
    longitud = calcular_longitud_fibra(contorno, mask_binaria.shape)
    if longitud == 0: return 0.0
    
    # 2. Ancho de Fibra
    # Distancia de cada punto interno al borde
    dist_transform = cv2.distanceTransform(mask_binaria, cv2.DIST_L2, 5)
    
    # Esqueleto binario (0 y 1)
    esqueleto = obtener_esqueleto(contorno, mask_binaria.shape)
    esqueleto_bool = esqueleto > 0
    
    # Extraer valores de distancia SOLO donde hay esqueleto
    anchos_en_esqueleto = dist_transform[esqueleto_bool]
    
    if len(anchos_en_esqueleto) == 0: return 0.0
    
    # El valor de distancia es el radio, el ancho es diametro (x2)
    ancho_promedio = np.mean(anchos_en_esqueleto) * 2.0
    
    if ancho_promedio == 0: return 0.0
    
    return float(longitud / ancho_promedio)

def elongacion_criterio_area(contorno):
    """
    Calcula Elongación Teórica = Area / (2 * Diametro^2)
    Nota: La fórmula listada era -A/2D^2, asumimos valor absoluto.
    Diametro se asume Diametro de Feret.
    """
    area = calcular_area(contorno)
    diametro = calcular_diametro_feret(contorno)
    
    if diametro == 0: return 0.0
    
    # Fórmula literal de la lista (asumiendo positivo)
    elongacion = area / (2 * (diametro ** 2))
    
    return float(elongacion)

def calcular_momentos_hu_log(contorno):
    """
    Calcula los 7 Momentos Invariantes de Hu (Escala logarítmica).
    Son excelentes para describir forma independiente de rotación y escala.
    """
    if contorno is None: return [0.0]*7
    
    momentos = cv2.moments(contorno)
    hu = cv2.HuMoments(momentos)
    
    # Log transform para hacerlos manejables (suelen ser muy pequeños)
    hu_log = []
    for i in range(0, 7):
        val = hu[i][0]
        if val != 0:
            # -1 * copysign(log10(abs(val)), val)
            hu_log.append(-1 * math.copysign(1.0, val) * math.log10(abs(val)))
        else:
            hu_log.append(0.0)
            
    return hu_log