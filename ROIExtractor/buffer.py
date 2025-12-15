import numpy as np
import cv2

class TrackBuffer:
    """
    Gestiona el historial de detecciones para cada objeto rastreado (ID).
    Decide en tiempo real si el nuevo avistamiento es mejor que el anterior.
    """
    def __init__(self):
        # Diccionario: { track_id: {'score': float, 'image': np.ndarray, 'frame_idx': int, 'bbox': list} }
        self.best_shots = {}
        
    def update(self, track_id, frame, bbox, conf, frame_idx):
        """
        Actualiza el buffer si la detección actual es de mayor calidad que la guardada.
        
        Args:
            track_id (int): ID único del objeto.
            frame (np.ndarray): Frame completo original.
            bbox (list): [x1, y1, x2, y2] coordenadas.
            conf (float): Confianza de la detección.
            frame_idx (int): Número de frame.
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Validar límites
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return # Caja inválida

        # Criterio de Calidad: Área * Confianza
        # Preferimos objetos grandes (cerca de la cámara) y seguros.
        area = (x2 - x1) * (y2 - y1)
        quality_score = area * conf
        
        # Centrado: Penalizar si toca los bordes (opcional)
        margin = 10
        if x1 < margin or y1 < margin or x2 > w - margin or y2 > h - margin:
            quality_score *= 0.5 # Reducir score a la mitad si está cortado

        # Lógica de reemplazo "Rey de la Colina"
        if track_id not in self.best_shots or quality_score > self.best_shots[track_id]['score']:
            
            # Extraer recorte
            crop = frame[y1:y2, x1:x2].copy()
            
            self.best_shots[track_id] = {
                'score': quality_score,
                'image': crop,
                'bbox': [x1, y1, x2, y2],
                'frame_idx': frame_idx,
                'conf': conf
            }

    def get_results(self):
        """Devuelve el diccionario con las mejores capturas finales."""
        return self.best_shots