import cv2
import numpy as np
import matplotlib
# Configurar backend no interactivo para evitar errores en hilos secundarios o servidores
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional

class VisualizadorFormas:
    """
    Clase encargada de generar visualizaciones para los descriptores de forma.
    """
    
    def __init__(self):
        # Configuración de colores (BGR)
        self.colores = {
            'contorno': (0, 255, 255),      # Amarillo
            'aprox_poly': (255, 0, 255),    # Magenta
            'convex_hull': (0, 255, 0),     # Verde
            'bbox': (255, 0, 0),            # Azul
            'min_circle': (0, 128, 255),    # Naranja
            'esqueleto': (0, 0, 255),       # Rojo
            'texto': (255, 255, 255)        # Blanco
        }

    def dibujar_resultados(self, imagen_base: np.ndarray, resultados: List[Dict[str, Any]], capas: Dict[str, bool]) -> np.ndarray:
        """
        Dibuja sobre la imagen las capas activas para cada objeto detectado.
        
        Args:
            imagen_base: Imagen original o lienzo negro.
            resultados: Lista de diccionarios con la data de los objetos (geometry, factors, etc.).
            capas: Diccionario con flags True/False para activar visualizaciones 
                   (ej: {'contour': True, 'skeleton': False}).
        """
        vis_img = imagen_base.copy()
        
        # Asegurar que sea BGR si entra en gris
        if len(vis_img.shape) == 2:
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
            
        for obj in resultados:
            geo = obj.get('geometry', {})
            
            # 1. Contorno Original + Chain Code Visual
            if capas.get('contour', False) and 'contour' in geo:
                cv2.drawContours(vis_img, [geo['contour']], -1, self.colores['contorno'], 1)
                self._dibujar_chain_code(vis_img, obj)

            # 2. Aproximación Poligonal
            if capas.get('polygon', False) and 'approx_poly' in geo:
                cv2.drawContours(vis_img, [geo['approx_poly']], -1, self.colores['aprox_poly'], 2)

            # 3. Convex Hull (Cierre Convexo)
            if capas.get('convex_hull', False) and 'hull' in geo:
                cv2.drawContours(vis_img, [geo['hull']], -1, self.colores['convex_hull'], 2)

            # 4. Bounding Boxes (Minimax y Recta)
            if capas.get('bbox', False):
                if 'rect_box' in geo:
                    x, y, w, h = geo['rect_box']
                    cv2.rectangle(vis_img, (x, y), (x+w, y+h), self.colores['bbox'], 2)
                if 'rect_fit' in geo:
                    box = cv2.boxPoints(geo['rect_fit'])
                    box = np.int0(box)
                    cv2.drawContours(vis_img, [box], 0, (255, 100, 0), 1)

            # 5. Círculo Mínimo
            if capas.get('min_circle', False) and 'min_circle' in geo:
                center, radius = geo['min_circle']
                # Asegurar enteros para OpenCV
                center = (int(center[0]), int(center[1]))
                radius = int(radius)
                cv2.circle(vis_img, center, radius, self.colores['min_circle'], 2)

            # 6. Esqueleto (requiere acceso a la máscara binaria global o recorte)
            # Nota: Para visualizar el esqueleto aquí, necesitaríamos pasar el esqueleto global 
            # o regenerarlo localmente. Por simplicidad, dibujaremos el centroide.
            if capas.get('skeleton', False) and 'centroid' in geo:
                cx, cy = geo['centroid']
                cv2.circle(vis_img, (int(cx), int(cy)), 3, self.colores['esqueleto'], -1)

            # 7. ID y Etiquetas
            if capas.get('show_ids', True) and 'centroid' in geo:
                cx, cy = geo['centroid']
                #ID
                cv2.putText(vis_img, f"ID:{obj.get('id', '?')}", (int(cx) - 20, int(cy) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colores['texto'], 2)
                # Ejemplo: Mostrar Redondez debajo del ID
                redondez = obj.get('factors', {}).get('Redondez', 0)
                cv2.putText(vis_img, f"R:{redondez:.2f}", (int(cx) - 20, int(cy) + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1)

        return vis_img

    def _dibujar_chain_code(self, img: np.ndarray, obj_data: Dict[str, Any]):
        """
        Dibuja una representación visual de los pasos del Chain Code.
        """
        if 'geometry' not in obj_data or 'contour' not in obj_data['geometry']:
            return
            
        contour = obj_data['geometry']['contour']
        if len(contour) == 0: return
        
        start_point = tuple(contour[0][0])
        curr_x, curr_y = start_point
        
        # Movimientos delta para Chain Code 4 (Derecha, Arriba, Izquierda, Abajo)
        moves_4 = [(1, 0), (0, -1), (-1, 0), (0, 1)]
        
        chain_4 = obj_data.get('boundary', {}).get('chain_code_4', [])
        
        if chain_4:
            for code in chain_4:
                if 0 <= code < len(moves_4):
                    dx, dy = moves_4[code]
                    next_x, next_y = curr_x + dx, curr_y + dy
                    # Dibujar línea pequeña verde neón para cada paso
                    cv2.line(img, (curr_x, curr_y), (next_x, next_y), (0, 255, 0), 1)
                    curr_x, curr_y = next_x, next_y

    def generar_grafico_signatura(self, angulos: np.ndarray, distancias: np.ndarray, obj_id: int) -> Optional[np.ndarray]:
        """
        Genera una imagen (array numpy) con el gráfico de la Signatura.
        """
        if len(angulos) == 0 or len(distancias) == 0:
            return None

        fig = plt.figure(figsize=(5, 3), dpi=100)
        ax = fig.add_subplot(111)
        
        ax.plot(angulos, distancias, color='blue', linewidth=1.5)
        ax.set_title(f"Signatura (ID: {obj_id})")
        ax.set_xlabel("Ángulo (rad)")
        ax.set_ylabel("Radio (px)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return self._fig_to_numpy(fig)

    def generar_grafico_curvatura(self, curvaturas: np.ndarray, obj_id: int) -> Optional[np.ndarray]:
        """
        Genera una imagen con el gráfico de Curvatura.
        """
        if len(curvaturas) == 0:
            return None

        fig = plt.figure(figsize=(5, 3), dpi=100)
        ax = fig.add_subplot(111)
        
        ax.plot(range(len(curvaturas)), curvaturas, color='red', linewidth=1)
        ax.set_title(f"Curvatura (ID: {obj_id})")
        ax.set_xlabel("Punto")
        ax.set_ylabel("Ángulo (rad)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return self._fig_to_numpy(fig)

    def _fig_to_numpy(self, fig) -> np.ndarray:
        """Convierte una figura de Matplotlib a imagen OpenCV (BGR)."""
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img_arr = np.asarray(buf)
        img_rgb = img_arr[:, :, :3] # Quitar canal alpha
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        plt.close(fig)
        return img_bgr