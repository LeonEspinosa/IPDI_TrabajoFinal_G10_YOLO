import cv2
import numpy as np
import math

def aproximacion_poligonal(contorno, epsilon_factor=0.02):
    """
    Simplifica el contorno usando el algoritmo Ramer-Douglas-Peucker.
    
    Args:
        contorno: Array de puntos del contorno.
        epsilon_factor: Factor de precisión (usualmente entre 0.01 y 0.05).
                        Cuanto mayor, más simplificado (menos vértices).
    
    Retorna:
        num_vertices (int): Cantidad de vértices del polígono resultante.
        poligono (array): El contorno simplificado.
    """
    if contorno is None or len(contorno) < 3:
        return 0, None
        
    perimetro = cv2.arcLength(contorno, True)
    epsilon = epsilon_factor * perimetro
    
    # approxPolyDP(curva, epsilon, cerrado)
    poligono = cv2.approxPolyDP(contorno, epsilon, True)
    
    return len(poligono), poligono

def bounding_box_axis_aligned(contorno):
    """
    Calcula el Bounding Box Alineado a los Ejes (Recto, no rotado).
    Necesario para calcular el 'Extent 1'.
    
    Retorna:
        (x, y, w, h): Coordenadas y dimensiones.
        area_bbox: Área del rectángulo (w * h).
    """
    if contorno is None:
        return (0,0,0,0), 0.0
        
    x, y, w, h = cv2.boundingRect(contorno)
    area = w * h
    return (x, y, w, h), float(area)

def calcular_extent_1(contorno):
    """
    Calcula el Extent 1 (Relación Área Objeto / Área Bounding Box Recto).
    """
    if contorno is None: return 0.0
    
    area_objeto = cv2.contourArea(contorno)
    _, area_bbox = bounding_box_axis_aligned(contorno)
    
    if area_bbox == 0: return 0.0
    
    return float(area_objeto / area_bbox)

def perimetro_chain_code_teorico(contorno):
    """
    Calcula el perímetro basado en la fórmula del Chain Code (4 u 8).
    Fórmula: N_pares + N_impares * sqrt(2)
    
    Nota: Esto es una aproximación teórica. cv2.arcLength suele ser más preciso,
    pero este descriptor se pide específicamente en literatura clásica.
    """
    if contorno is None or len(contorno) < 2:
        return 0.0
        
    puntos = contorno.squeeze()
    if puntos.ndim != 2: return 0.0
    
    n_pares = 0   # Movimientos arriba, abajo, izq, der (distancia 1)
    n_impares = 0 # Movimientos diagonales (distancia sqrt(2))
    
    num_puntos = len(puntos)
    
    for i in range(num_puntos - 1):
        dx = puntos[i+1][0] - puntos[i][0]
        dy = puntos[i+1][1] - puntos[i][1]
        
        # Si dx y dy son ambos distintos de cero, es diagonal
        if dx != 0 and dy != 0:
            n_impares += 1
        else:
            n_pares += 1
            
    # Cerrar el contorno (del último al primero)
    dx = puntos[0][0] - puntos[-1][0]
    dy = puntos[0][1] - puntos[-1][1]
    if dx != 0 and dy != 0:
        n_impares += 1
    else:
        n_pares += 1
        
    return float(n_pares + (n_impares * math.sqrt(2)))

def histograma_tangentes(contorno, n_bins=8):
    """
    Calcula un histograma simple de los ángulos de las tangentes del contorno.
    Describe la distribución de la orientación de los bordes.
    
    Retorna:
        hist (array): Valores normalizados del histograma.
    """
    if contorno is None or len(contorno) < 2:
        return np.zeros(n_bins)
        
    puntos = contorno.squeeze()
    if puntos.ndim != 2: return np.zeros(n_bins)
    
    angulos = []
    
    # Calcular ángulos entre puntos consecutivos
    for i in range(len(puntos) - 1):
        dy = puntos[i+1][1] - puntos[i][1]
        dx = puntos[i+1][0] - puntos[i][0]
        angulo = math.atan2(dy, dx) # Radianes entre -pi y pi
        angulos.append(angulo)
        
    # Convertir a grados y normalizar a [0, 180] para simplicidad de histograma
    angulos = np.degrees(angulos) % 180
    
    hist, _ = np.histogram(angulos, bins=n_bins, range=(0, 180))
    
    # Normalizar histograma para que sume 1
    hist_norm = hist.astype(float) / (np.sum(hist) + 1e-6)
    
    return hist_norm