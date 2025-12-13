import numpy as np
import cv2

def descriptores_fourier(contorno, n_descriptores=10):
    """
    Calcula los Descriptores de Fourier del contorno.
    Convierte las coordenadas (x, y) en números complejos (x + iy) y aplica FFT.
    
    Args:
        n_descriptores: Cuántos coeficientes de baja frecuencia retornar.
                        Estos representan la forma general.
    
    Retorna:
        lista de magnitudes normalizadas de los primeros descriptores.
    """
    if contorno is None or len(contorno) < n_descriptores:
        return [0.0] * n_descriptores
        
    # Aplanar y convertir a complejo x + iy
    puntos = contorno.squeeze()
    if points_ndim_check(puntos): return [0.0] * n_descriptores
    
    contorno_complejo = puntos[:, 0] + 1j * puntos[:, 1]
    
    # Aplicar Transformada Rápida de Fourier (FFT)
    fourier_result = np.fft.fft(contorno_complejo)
    
    # Obtener magnitud (ignoramos la fase para invariancia a rotación inicial)
    magnitudes = np.abs(fourier_result)
    
    # El primer componente (DC) depende de la posición, lo ignoramos o ponemos a 0
    # El segundo componente (magnitudes[1]) se usa para normalizar (invariancia a escala)
    if magnitudes[1] == 0:
        return [0.0] * n_descriptores
        
    descriptors = magnitudes / magnitudes[1]
    
    # Retornamos los primeros N coeficientes (saltando el 0 que es la posición media)
    # [1:n+1]
    return descriptors[1 : n_descriptores + 1].tolist()

def dimension_fractal(mask_binaria):
    """
    Estima la Dimensión Fractal (Box Counting) del contorno del objeto.
    Mide la "rugosidad" del borde.
    
    Algoritmo simplificado de Box-Counting.
    """
    if mask_binaria is None: return 0.0
    
    # Obtener solo el borde para calcular fractalidad del borde
    contornos, _ = cv2.findContours(mask_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos: return 0.0
    
    # Dibujar borde en una imagen nueva
    Z = np.zeros_like(mask_binaria)
    cv2.drawContours(Z, contornos, -1, 255, 1)
    
    # Binarizar estricto (0 y 1)
    Z = (Z > 0)
    
    # Minimal Box Counting
    p = min(Z.shape)
    
    # Potencia de 2 más grande que cabe en la imagen
    n = 2**np.floor(np.log(p)/np.log(2))
    n = int(np.log(n)/np.log(2))
    
    sizes = 2**np.arange(n, 1, -1)
    counts = []
    
    for size in sizes:
        counts.append(len(non_zero_chunks(Z, size)))
        
    if len(sizes) < 2: return 0.0
    
    # Ajuste lineal (pendiente de log(counts) vs log(sizes))
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    
    # La dimensión es la pendiente negativa
    return float(-coeffs[0])

def non_zero_chunks(arr, size):
    """Helper para box counting: cuenta bloques no vacíos"""
    h, w = arr.shape
    # Recortar para que sea divisible
    h_new = (h // size) * size
    w_new = (w // size) * size
    arr = arr[:h_new, :w_new]
    
    # Reshape para contar bloques
    return [1 for r in range(0, h_new, size) for c in range(0, w_new, size) 
            if np.any(arr[r:r+size, c:c+size])]

def points_ndim_check(pts):
    return pts.ndim != 2