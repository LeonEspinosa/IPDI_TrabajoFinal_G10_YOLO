import numpy as np
import math
import cv2

def obtener_chain_code_8(contorno):
    """
    Genera el Chain Code de Freeman de 8 direcciones.
    """
    if contorno is None or len(contorno) < 2:
        return []

    # Aplanar contorno
    puntos = contorno.squeeze()
    if points_ndim_check(puntos): return [] # Validación auxiliar
    
    # Mapa de direcciones (dx, dy) -> código
    code_map = {
        (1, 0): 0, (1, -1): 1, (0, -1): 2, (-1, -1): 3,
        (-1, 0): 4, (-1, 1): 5, (0, 1): 6, (1, 1): 7
    }
    
    chain = []
    num_puntos = len(puntos)
    
    for i in range(num_puntos - 1):
        # np.clip asegura que el movimiento sea -1, 0 o 1
        dx = int(np.clip(puntos[i+1][0] - puntos[i][0], -1, 1))
        dy = int(np.clip(puntos[i+1][1] - puntos[i][1], -1, 1))
        
        if (dx, dy) in code_map:
            chain.append(code_map[(dx, dy)])
            
    return chain

def obtener_chain_code_4(contorno):
    """
    Genera el Chain Code de 4 direcciones.
    Descompone movimientos diagonales en dos movimientos ortogonales.
    """
    if contorno is None or len(contorno) < 2:
        return []
        
    puntos = contorno.squeeze()
    if points_ndim_check(puntos): return []

    # Mapa de direcciones 4-way
    code_map_4 = {
        (1, 0): 0,   # Derecha
        (0, -1): 1,  # Arriba
        (-1, 0): 2,  # Izquierda
        (0, 1): 3    # Abajo
    }
    
    chain = []
    num_puntos = len(puntos)
    
    for i in range(num_puntos - 1):
        dx = int(np.clip(puntos[i+1][0] - puntos[i][0], -1, 1))
        dy = int(np.clip(puntos[i+1][1] - puntos[i][1], -1, 1))
        
        if dx == 0 and dy == 0:
            continue
            
        # Si es movimiento puro
        if (dx, dy) in code_map_4:
            chain.append(code_map_4[(dx, dy)])
        else:
            # Es diagonal: Descomponer
            # Prioridad horizontal arbitraria (se podría alternar)
            if dx != 0: chain.append(code_map_4[(dx, 0)])
            if dy != 0: chain.append(code_map_4[(0, dy)])
            
    return chain

def calcular_curvatura(contorno, k=None):
    """
    Calcula la curvatura diferencial basada en k-vecinos.
    Retorna: (array_de_angulos, curvatura_media)
    """
    if contorno is None or len(contorno) < 3:
        return np.array([]), 0.0
    
    pts = contorno.squeeze()
    n_points = len(pts)
    
    # k adaptativo: 5% de los puntos si no se define
    if k is None:
        k = max(2, min(10, n_points // 20))
        
    # Seguridad para contornos pequeños
    if n_points < 2*k + 1:
        k = max(1, n_points // 3)
        
    curvatures = []
    
    for i in range(n_points):
        # Puntos anterior, actual y siguiente
        p_minus = pts[(i - k) % n_points]
        p_curr = pts[i]
        p_plus = pts[(i + k) % n_points]
        
        # Vectores
        v1 = p_curr - p_minus
        v2 = p_plus - p_curr
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            angle = 0.0
        else:
            # Ángulo entre vectores usando producto punto
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
        curvatures.append(angle)
        
    curvatures_arr = np.array(curvatures)
    return curvatures_arr, float(np.mean(curvatures_arr))

def calcular_signatura(contorno):
    """
    Calcula la Signatura (Distancia del Centroide al Borde vs Ángulo).
    Retorna: (angulos_ordenados, distancias_ordenadas)
    """
    M = cv2.moments(contorno)
    if M["m00"] == 0:
        return np.array([]), np.array([])
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    pts = contorno.squeeze()
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)
        
    distances = []
    angles = []
    
    for p in pts:
        dx = p[0] - cx
        dy = p[1] - cy
        
        # Coordenadas polares
        r = math.sqrt(dx**2 + dy**2)
        theta = math.atan2(dy, dx)
        
        distances.append(r)
        angles.append(theta)
        
    # Ordenar por ángulo para que el gráfico sea continuo
    sorted_indices = np.argsort(angles)
    angles_sorted = np.array(angles)[sorted_indices]
    distances_sorted = np.array(distances)[sorted_indices]
    
    return angles_sorted, distances_sorted

def points_ndim_check(pts):
    """Ayuda interna para validar dimensiones de numpy"""
    return pts.ndim != 2