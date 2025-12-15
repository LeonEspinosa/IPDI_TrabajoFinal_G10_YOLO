import cv2
import numpy as np
import math
import matplotlib
matplotlib.use('Agg') # Backend sin cabeza para no abrir ventanas
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List, Optional
# Eliminamos PIL para mantener consistencia con OpenCV/Numpy, pero mantenemos la lógica

class ShapeDescriptorProcessor:
    """
    Procesador de descriptores de forma (Skeleton, ChainCode, etc.).
    Adaptado para integración con pipeline YOLO (Numpy/OpenCV).
    """
    
    def __init__(self):
        self.binary: Optional[np.ndarray] = None
        self.skeleton: Optional[np.ndarray] = None
        self.results_cache: List[Dict[str, Any]] = []
        self.polygon_epsilon_factor: float = 0.01 
        self.visualization_layers = {
            'contour': True,        # Activado por defecto
            'polygon': False,
            'convex_hull': False,
            'bbox_minimax': True,
            'bbox_bestfit': False,
            'min_circle': False,
            'skeleton': True        # Activado por defecto
        }
    
    def process_numpy_image(self, img_bgr: np.ndarray) -> bool:
        """
        Método de entrada principal para el video.
        Recibe un recorte BGR de la naranja, lo binariza inteligentemente y prepara el análisis.
        """
        try:
            # 1. Convertir a Gris
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            # 2. Binarización Automática (Otsu es mejor que fijo 127 para iluminación variable)
            # Para naranjas, a veces es mejor segmentar en el canal de Saturación o el canal 'I' de YIQ
            # Aquí usamos Otsu sobre gris para robustez general
            _, self.binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 3. Limpieza de Ruido (Morfología)
            kernel = np.ones((3,3), np.uint8)
            self.binary = cv2.morphologyEx(self.binary, cv2.MORPH_OPEN, kernel)

            # 4. Corrección de Fondo (Asegurar que el objeto es blanco y fondo negro)
            # Asumimos que el objeto está en el centro. Si las esquinas son blancas, invertimos.
            h, w = self.binary.shape
            corners = [self.binary[0,0], self.binary[0,w-1], self.binary[h-1,0], self.binary[h-1,w-1]]
            if np.mean(corners) > 127:
                self.binary = cv2.bitwise_not(self.binary)
                
            return True
        except Exception as e:
            print(f"Error procesando imagen numpy: {e}")
            return False

    # --- MÉTODOS ORIGINALES (Mantenidos intactos) ---

    def generate_skeleton(self) -> None:
        """Genera el esqueleto topológico"""
        if self.binary is None: return
        
        img = self.binary.copy()
        skel = np.zeros(img.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        # Límite de iteraciones para evitar bucles infinitos en imágenes ruidosas
        max_iters = 100 
        iters = 0
        
        while iters < max_iters:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            if cv2.countNonZero(img) == 0:
                break
            iters += 1
        self.skeleton = skel

    def get_chain_code_8(self, contour: np.ndarray) -> List[int]:
        if contour is None or len(contour) < 2: return []
        pts = contour.squeeze()
        if pts.ndim == 1: return []
        code_map = {(1, 0): 0, (1, -1): 1, (0, -1): 2, (-1, -1): 3,
                    (-1, 0): 4, (-1, 1): 5, (0, 1): 6, (1, 1): 7}
        chain = []
        for i in range(len(pts) - 1):
            dx = np.clip(pts[i+1][0] - pts[i][0], -1, 1)
            dy = np.clip(pts[i+1][1] - pts[i][1], -1, 1)
            if (dx, dy) in code_map:
                chain.append(code_map[(dx, dy)])
        return chain

    def get_chain_code_4(self, contour: np.ndarray) -> List[int]:
        if contour is None or len(contour) < 2: return []
        pts = contour.squeeze()
        if pts.ndim == 1: return []
        code_map_4 = {(1, 0): 0, (0, -1): 1, (-1, 0): 2, (0, 1): 3}
        chain = []
        for i in range(len(pts) - 1):
            dx = int(np.clip(pts[i+1][0] - pts[i][0], -1, 1))
            dy = int(np.clip(pts[i+1][1] - pts[i][1], -1, 1))
            if dx == 0 and dy == 0: continue
            if (dx, dy) in code_map_4:
                chain.append(code_map_4[(dx, dy)])
            else:
                if dx != 0: chain.append(code_map_4[(dx, 0)])
                if dy != 0: chain.append(code_map_4[(0, dy)])
        return chain

    def calculate_curvature(self, contour: np.ndarray, k: int = None) -> Tuple[np.ndarray, float]:
        if contour is None or len(contour) < 3: return np.array([]), 0.0
        pts = contour.squeeze()
        n_points = len(pts)
        if k is None: k = max(2, min(10, n_points // 20))
        if n_points < 2*k + 1: k = max(1, n_points // 3)
        curvatures = []
        for i in range(n_points):
            p_minus = pts[(i - k) % n_points]
            p_curr = pts[i]
            p_plus = pts[(i + k) % n_points]
            v1 = p_curr - p_minus
            v2 = p_plus - p_curr
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0: angle = 0.0
            else:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
            curvatures.append(angle)
        curv_arr = np.array(curvatures)
        return curv_arr, float(np.mean(curv_arr)) if len(curv_arr) > 0 else 0.0

    def calculate_signature(self, contour: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        M = cv2.moments(contour)
        if M["m00"] == 0: return np.array([]), np.array([])
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        pts = contour.squeeze()
        if pts.ndim == 1: pts = pts.reshape(-1, 2)
        distances, angles = [], []
        for p in pts:
            dx, dy = p[0] - cx, p[1] - cy
            distances.append(math.sqrt(dx**2 + dy**2))
            angles.append(math.atan2(dy, dx))
        sorted_indices = np.argsort(angles)
        return np.array(angles)[sorted_indices], np.array(distances)[sorted_indices]

    def analyze_shape(self, pixels_per_unit: float = 1.0, unit_name: str = "px", min_area: float = 50.0) -> List[Dict[str, Any]]:
        if self.binary is None: return []
        contours, _ = cv2.findContours(self.binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        results = []
        if not contours: return []
        
        # Ordenar contornos por área y quedarse con el más grande (asumiendo que es la naranja principal)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # Analizamos solo el contorno más grande para evitar ruido de hojas pequeñas
        cnt = contours[0]
        
        area_px = cv2.contourArea(cnt)
        if area_px < min_area: return []

        perimeter_px = cv2.arcLength(cnt, True)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Factores de forma básicos
        compactness = (perimeter_px ** 2) / (4 * np.pi * area_px) if area_px > 0 else 0
        solidity = area_px / hull_area if hull_area > 0 else 0
        
        # Chain codes & Features avanzadas
        chain_8 = self.get_chain_code_8(cnt)
        curvatures, mean_curv = self.calculate_curvature(cnt)
        
        # Esqueleto (Si está calculado)
        fiber_len = 0
        if self.skeleton is not None:
            mask = np.zeros_like(self.binary)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            fiber_len = cv2.countNonZero(cv2.bitwise_and(self.skeleton, self.skeleton, mask=mask))

        return [{
            "id": 1,
            "geometry": {"contour": cnt, "rect_box": (x,y,w,h)},
            "metrics": {"Area": area_px, "Perimeter": perimeter_px, "Skeleton_Len": fiber_len},
            "factors": {"Compactness": compactness, "Solidity": solidity, "Mean_Curvature": mean_curv},
            "boundary": {"chain_code_8": chain_8}
        }]

    def draw_visuals_on_image(self, img_bgr):
        """Dibuja los resultados sobre la imagen original"""
        if not self.results_cache: return img_bgr
        res = self.results_cache[0] # Dibujar solo el principal
        
        # Dibujar Contorno
        cv2.drawContours(img_bgr, [res['geometry']['contour']], -1, (0, 255, 0), 2)
        
        # Dibujar Esqueleto (Superpuesto en Rojo)
        if self.skeleton is not None:
             # Necesitamos "pegar" el esqueleto binario (que tiene el tamaño del recorte) sobre la imagen
             # Solo pintamos los pixeles blancos del esqueleto de rojo
             img_bgr[self.skeleton == 255] = [0, 0, 255]

        return img_bgr