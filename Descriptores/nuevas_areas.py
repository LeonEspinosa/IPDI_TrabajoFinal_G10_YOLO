import cv2
import numpy as np
from .basicos import calcular_diametro_feret

def calcular_radio_inscripto(mask_binaria):
    """
    Calcula el radio del Círculo Máximo Inscrito.
    Utiliza la Transformada de Distancia sobre la máscara binaria.
    
    Args:
        mask_binaria: Imagen binaria (255 objeto, 0 fondo).
        
    Retorna:
        max_radio: El radio más grande que cabe dentro del objeto.
    """
    if mask_binaria is None: return 0.0
    
    # distanceTransform calcula la distancia de cada pixel al 0 más cercano
    # DIST_L2 es distancia euclideana
    dist_transform = cv2.distanceTransform(mask_binaria, cv2.DIST_L2, 5)
    
    if dist_transform is None: return 0.0
    
    # El valor máximo de la transformada es el radio del círculo inscrito más grande
    _, max_val, _, _ = cv2.minMaxLoc(dist_transform)
    
    return float(max_val)

def calcular_radius_ratio(mask_binaria, contorno):
    """
    Calcula el Radius Ratio: Radio Inscripto / Radio Circunscripto.
    
    Nota: Usamos Diametro Feret / 2 como aproximación robusta del Radio Circunscripto.
    """
    radio_inscripto = calcular_radio_inscripto(mask_binaria)
    diametro_feret = calcular_diametro_feret(contorno)
    
    if diametro_feret <= 0: return 0.0
    
    radio_circunscripto = diametro_feret / 2.0
    
    return float(radio_inscripto / radio_circunscripto)

def calcular_area_fraction(mask_con_huecos):
    """
    Calcula la Fracción de Área: Área Neta / Área Llena.
    
    Para que esto funcione, la máscara de entrada debe tener los huecos (negros)
    dentro del objeto si existen.
    
    Retorna:
        1.0 si el objeto es sólido.
        < 1.0 si tiene agujeros.
    """
    if mask_con_huecos is None: return 0.0
    
    # Área Neta: Conteo real de píxeles blancos
    area_neta = cv2.countNonZero(mask_con_huecos)
    
    # Área Llena: Rellenamos los huecos para calcular el área total
    contornos, _ = cv2.findContours(mask_con_huecos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask_llena = np.zeros_like(mask_con_huecos)
    cv2.drawContours(mask_llena, contornos, -1, 255, thickness=cv2.FILLED)
    
    area_llena = cv2.countNonZero(mask_llena)
    
    if area_llena <= 0: return 0.0
    
    return float(area_neta / area_llena)

def area_neta_vs_llena(mask_con_huecos):
    """
    Retorna ambos valores por separado.
    """
    area_neta = cv2.countNonZero(mask_con_huecos)
    
    contornos, _ = cv2.findContours(mask_con_huecos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_llena = np.zeros_like(mask_con_huecos)
    cv2.drawContours(mask_llena, contornos, -1, 255, thickness=cv2.FILLED)
    area_llena = cv2.countNonZero(mask_llena)
    
    return float(area_neta), float(area_llena)