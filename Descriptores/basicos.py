import cv2
import numpy as np
from scipy.spatial import distance

def calcular_area(contorno):
    """
    Calcula el área (A) de la región encerrada por el contorno.
    Basado en el Teorema de Green.
    """
    if contorno is None: return 0.0
    area = cv2.contourArea(contorno)
    return float(area)

def calcular_perimetro(contorno):
    """
    Calcula el perímetro (P) o longitud del arco del contorno.
    """
    if contorno is None: return 0.0
    perimetro = cv2.arcLength(contorno, True)
    return float(perimetro)

def calcular_diametro_feret(contorno):
    """
    Calcula el Diámetro de Feret máximo (Caliper Máximo).
    Distancia euclidiana máxima entre pares de puntos del Cierre Convexo.
    """
    if contorno is None or len(contorno) < 2: return 0.0
    
    hull = cv2.convexHull(contorno)
    if len(hull) < 2: return 0.0
    
    puntos_hull = hull.reshape(-1, 2)
    
    try:
        # pdist calcula distancias por pares eficientemente
        distancias = distance.pdist(puntos_hull, metric='euclidean')
        if len(distancias) == 0: return 0.0
        max_feret = np.max(distancias)
    except Exception:
        max_feret = 0.0
        
    return float(max_feret)

def calcular_diametro_minimax(contorno):
    """
    Calcula los diámetros Minimax (Ancho mínimo y Largo máximo de la caja rotada).
    Retorna: (ancho_min, largo_max)
    """
    if contorno is None or len(contorno) < 2: return 0.0, 0.0
    
    rect = cv2.minAreaRect(contorno)
    (w, h) = rect[1]
    
    dimensiones = sorted([w, h])
    min_dim = dimensiones[0]
    max_dim = dimensiones[1]
    
    return float(min_dim), float(max_dim)

def calcular_proyecciones(mask_binaria):
    """
    Calcula las proyecciones ortogonales (histogramas) de la forma binaria.
    Útil para analizar la simetría y distribución de masa.
    
    Argumentos:
        mask_binaria: Imagen numpy (H, W) donde el objeto es > 0 (blanco).
                      Puede ser un recorte (ROI) o la máscara completa.
    
    Retorna:
        (proj_x, proj_y): Arrays numpy con la suma de píxeles.
    """
    if mask_binaria is None:
        return np.array([]), np.array([])
        
    # Normalizar a 0 y 1 para contar píxeles
    binaria = (mask_binaria > 0).astype(np.int32)
    
    # Proyección sobre el eje X (Suma de columnas -> Cuántos píxeles hay en cada columna vertical)
    proj_x = np.sum(binaria, axis=0) 
    
    # Proyección sobre el eje Y (Suma de filas -> Cuántos píxeles hay en cada fila horizontal)
    proj_y = np.sum(binaria, axis=1)
    
    return proj_x, proj_y