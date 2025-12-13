import cv2
import numpy as np
from PIL import Image

def cargar_imagen_binaria(imagen_entrada, umbral=127, invertir_automatico=True):
    """
    Convierte una imagen (ruta, array numpy o objeto PIL) a una máscara binaria estricta (0 y 255).
    Realiza corrección automática de fondo (asegura que el fondo sea negro).
    
    Args:
        imagen_entrada: Puede ser un path (str), imagen numpy (cv2) o imagen PIL.
        umbral: Valor de corte para binarizar (0-255).
        invertir_automatico: Si True, detecta si el fondo es blanco y lo invierte a negro.
        
    Retorna:
        numpy.ndarray: Imagen binaria (H, W) de tipo uint8.
    """
    try:
        # 1. Normalizar entrada a Array Numpy
        if isinstance(imagen_entrada, str):
            # Cargar desde ruta
            img = cv2.imread(imagen_entrada, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"No se pudo cargar la imagen desde: {imagen_entrada}")
        elif isinstance(imagen_entrada, Image.Image):
            # Convertir desde PIL
            img = np.array(imagen_entrada)
        elif isinstance(imagen_entrada, np.ndarray):
            img = imagen_entrada.copy()
        else:
            raise TypeError("Formato de imagen no soportado. Use str, np.ndarray o PIL.Image")

        # 2. Manejo de Canales (Color -> Gris)
        if len(img.shape) == 3:
            if img.shape[2] == 4:  # RGBA (con transparencia)
                # Convertir a gris. La transparencia puede requerir manejo especial si es máscara
                # Aquí usamos conversión directa estándar
                gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            else:  # RGB / BGR
                # Asumimos BGR (estándar OpenCV). Si fuera RGB, el peso de conversión varía poco para binarización
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img # Ya es gris o binaria

        # 3. Binarización Estricta
        # Cualquier píxel > umbral se vuelve 255, el resto 0
        _, binaria = cv2.threshold(gray, umbral, 255, cv2.THRESH_BINARY)

        # 4. Corrección Automática de Fondo (El fondo debe ser NEGRO para findContours)
        if invertir_automatico:
            h, w = binaria.shape
            # Muestrear los bordes de la imagen
            borde_sup = binaria[0, :]
            borde_inf = binaria[h-1, :]
            borde_izq = binaria[:, 0]
            borde_der = binaria[:, w-1]
            
            total_pixeles_borde = 2*w + 2*h
            pixeles_blancos_borde = (np.sum(borde_sup == 255) + np.sum(borde_inf == 255) +
                                     np.sum(borde_izq == 255) + np.sum(borde_der == 255))
            
            # Si más del 50% del borde es blanco, asumimos que el fondo es blanco
            if pixeles_blancos_borde > total_pixeles_borde * 0.5:
                binaria = cv2.bitwise_not(binaria)

        return binaria

    except Exception as e:
        print(f"❌ Error en preprocesamiento: {e}")
        return None

def limpiar_ruido(mask_binaria, kernel_size=3):
    """
    Aplica operaciones morfológicas para eliminar ruido pequeño (puntos blancos aislados).
    """
    if mask_binaria is None: return None
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Opening: Erosión seguida de Dilatación (quita puntos blancos en fondo negro)
    mask_limpia = cv2.morphologyEx(mask_binaria, cv2.MORPH_OPEN, kernel)
    
    # Closing: Dilatación seguida de Erosión (cierra huecos negros dentro del objeto blanco)
    mask_limpia = cv2.morphologyEx(mask_limpia, cv2.MORPH_CLOSE, kernel)
    
    return mask_limpia