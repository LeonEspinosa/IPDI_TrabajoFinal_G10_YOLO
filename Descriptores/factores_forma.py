import math
import cv2
import numpy as np
from .basicos import calcular_area, calcular_perimetro, calcular_diametro_feret, calcular_diametro_minimax

def calcular_compacidad(contorno):
    """
    Calcula la Compacidad: P^2 / (4 * pi * A)
    C = 1.0 para círculo perfecto.
    """
    area = calcular_area(contorno)
    perimetro = calcular_perimetro(contorno)
    
    if area <= 0: return 0.0
    return float((perimetro ** 2) / (4 * math.pi * area))

def calcular_redondez(contorno):
    """
    Calcula la Redondez: Area_Objeto / Area_Circulo_Feret
    """
    area = calcular_area(contorno)
    max_feret = calcular_diametro_feret(contorno)
    
    if max_feret <= 0: return 0.0
    
    area_circulo = math.pi * ((max_feret / 2.0) ** 2)
    if area_circulo <= 0: return 0.0
        
    return float(area / area_circulo)

def calcular_elongacion(contorno):
    """
    Calcula la Elongación (Aspect Ratio): Largo_Max / Ancho_Min
    """
    ancho_min, largo_max = calcular_diametro_minimax(contorno)
    if ancho_min <= 0: return 0.0
    return float(largo_max / ancho_min)

def calcular_rectangularidad(contorno):
    """
    Calcula la Rectangularidad (basada en MinAreaRect): Area / (W_min * H_max)
    Indica cuánto llena el objeto su rectángulo rotado óptimo.
    """
    area = calcular_area(contorno)
    ancho_min, largo_max = calcular_diametro_minimax(contorno)
    area_caja = ancho_min * largo_max
    
    if area_caja <= 0: return 0.0
    return float(area / area_caja)

def calcular_solidez(contorno):
    """
    Calcula la Solidez: Area / Convex_Hull_Area
    Mide la densidad del objeto. Objetos con bahías profundas o contornos
    muy irregulares (como una estrella) tienen baja solidez.
    Una naranja debería tener solidez cercana a 1.0.
    """
    area = calcular_area(contorno)
    hull = cv2.convexHull(contorno)
    hull_area = cv2.contourArea(hull)
    
    if hull_area <= 0: return 0.0
    return float(area / hull_area)

def calcular_convexidad(contorno):
    """
    Calcula la Convexidad: Convex_Hull_Perimeter / Perimeter
    Mide la rugosidad del borde.
    - 1.0: Contorno perfectamente convexo (sin indentaciones).
    - < 1.0: Contorno rugoso o complejo.
    """
    perimetro = calcular_perimetro(contorno)
    hull = cv2.convexHull(contorno)
    hull_perimetro = cv2.arcLength(hull, True)
    
    if perimetro <= 0: return 0.0
    return float(hull_perimetro / perimetro)

def calcular_excentricidad(contorno):
    """
    Calcula la Excentricidad usando momentos de imagen.
    Es una medida de qué tan 'estirada' está la elipse que mejor se ajusta al objeto.
    - 0: Círculo perfecto.
    - Cercano a 1: Línea recta.
    """
    M = cv2.moments(contorno)
    if M["m00"] == 0: return 0.0
    
    # Momentos centrales normalizados
    mu20 = M["mu20"] / M["m00"]
    mu02 = M["mu02"] / M["m00"]
    mu11 = M["mu11"] / M["m00"]
    
    # Cálculo de eigenvalores de la matriz de covarianza
    common = np.sqrt(4 * mu11**2 + (mu20 - mu02)**2)
    lambda1 = (mu20 + mu02 + common) / 2
    lambda2 = (mu20 + mu02 - common) / 2
    
    if lambda1 == 0: return 0.0
    
    # Excentricidad = sqrt(1 - (eje_menor/eje_mayor)^2)
    # lambda2 es proporcional al eje menor, lambda1 al mayor
    term = lambda2 / lambda1
    if term > 1: term = 1 # Corrección por error numérico
    
    return float(np.sqrt(1 - term))