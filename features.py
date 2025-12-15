import cv2
import numpy as np

def bgr_to_yiq_custom(image_bgr):
    """
    Convierte una imagen BGR (OpenCV) a YIQ normalizado (0.0 - 1.0).
    Fórmula estándar NTSC.
    """
    image_float = image_bgr.astype(np.float32) / 255.0
    B, G, R = cv2.split(image_float)
    
    # Matriz de conversión estándar
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    I = 0.596 * R - 0.274 * G - 0.322 * B
    Q = 0.211 * R - 0.523 * G + 0.312 * B
    
    # Y está en [0, 1]
    # I está aprox en [-0.6, 0.6]
    # Q está aprox en [-0.5, 0.5]
    return cv2.merge([Y, I, Q])

def extract_yiq_features(image, bbox):
    """
    Extrae características cromáticas en espacio YIQ dentro del bounding box.
    Retorna: (Y, I, Q) promedios y etiqueta de madurez basada en el canal 'I'.
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Validar coordenadas
    h_img, w_img = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    
    # Recortar ROI
    roi = image[y1:y2, x1:x2]
    
    if roi.size == 0:
        return (0,0,0), "Error"

    # Muestreo central (50% del centro) para evitar ruido de fondo
    h_roi, w_roi = roi.shape[:2]
    roi_center = roi[int(h_roi*0.25):int(h_roi*0.75), int(w_roi*0.25):int(w_roi*0.75)]
    
    if roi_center.size == 0:
        roi_center = roi # Fallback si es muy pequeño

    # Convertir a YIQ
    yiq_roi = bgr_to_yiq_custom(roi_center)
    
    # Calcular promedios
    mean_yiq = np.mean(yiq_roi, axis=(0,1))
    Y, I, Q = mean_yiq
    
    # --- CLASIFICACIÓN DE MADUREZ BASADA EN CANAL 'I' (ORANGE-CYAN AXIS) ---
    # El canal I es positivo para colores piel/naranja y negativo para azules/cian.
    # Naranjas maduras tienen I alto (> 0.15 aprox).
    # Naranjas verdes tienen I bajo o negativo.
    
    # Umbrales estimados (ajustar experimentalmente con tu video)
    status = "Desconocido"
    if I > 0.18:
        status = "Madura (Roja/Intensa)"
    elif 0.08 <= I <= 0.18:
        status = "Madura (Estándar)"
    elif 0.02 <= I < 0.08:
        status = "Envero (Cambio)"
    else: # I < 0.02
        status = "Verde"
        
    return (float(Y), float(I), float(Q)), status

def estimate_diameter_pixels(bbox):
    """Estima diámetro promedio en píxeles."""
    x1, y1, x2, y2 = bbox
    return ((x2 - x1) + (y2 - y1)) / 2

# --- BLOQUE DE PRUEBA ---
if __name__ == "__main__":
    print("Prueba de conversión YIQ...")
    # Crear pixel naranja puro (BGR: 0, 165, 255)
    img_test = np.zeros((10,10,3), dtype=np.uint8)
    img_test[:] = (0, 165, 255) 
    
    yiq_vals, estado = extract_yiq_features(img_test, [0,0,10,10])
    print(f"BGR Naranja -> YIQ: {yiq_vals}")
    print(f"Clasificación: {estado}")
    # Debería dar un I alto positivo