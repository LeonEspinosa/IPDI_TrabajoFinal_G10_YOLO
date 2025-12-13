import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import argparse
import os
import pandas as pd
from features import extract_color_histogram, estimate_diameter_pixels

def process_video_agronomic(model_path, input_video_path, output_video_path, output_csv_path, tracker_type="bytetrack.yaml"):
    # Limpieza de consola
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"=== ANÁLISIS AGRONÓMICO CON FILTRADO TEMPORAL ===")
    
    # 1. Carga
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error abriendo video: {input_video_path}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # --- ESTRUCTURA DE DATOS PARA FILTRADO TEMPORAL ---
    # Almacena el historial de mediciones para cada ID único
    # fruit_history[id] = {'diameters': [], 'colors_hsv': [], 'frames': 0}
    fruit_history = defaultdict(lambda: {'diameters': [], 'colors': [], 'frames': 0})
    
    track_history = defaultdict(lambda: [])

    print("🚀 Procesando video con estabilización de datos...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Inferencia con Tracking (Persistencia activada)
        results = model.track(frame, persist=True, tracker=tracker_type, conf=0.45, verbose=False)
        
        # Copia para dibujar limpia
        annotated_frame = frame.copy()

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Dibujar trazas de movimiento
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                
                # --- 1. EXTRACCIÓN DE DATOS CRUDOS ---
                # Color (Tu función en features.py devuelve BGR promedio y status)
                # NOTA: Para el promedio estadístico, usaremos los valores numéricos crudos
                color_bgr, _ = extract_color_histogram(frame, box.numpy()) 
                
                # Tamaño (Tu función devuelve diametro en px)
                diameter_px = estimate_diameter_pixels(box.numpy())

                # --- 2. ACUMULACIÓN TEMPORAL (BUFFER) ---
                fruit_history[track_id]['diameters'].append(diameter_px)
                fruit_history[track_id]['colors'].append(color_bgr) # Guardamos tupla (B,G,R)
                fruit_history[track_id]['frames'] += 1

                # --- 3. CÁLCULO ESTADÍSTICO ROBUSTO (EN VIVO) ---
                # Usamos la MEDIANA del historial acumulado hasta ahora.
                # Esto filtra el "blur" (que da diámetros gigantes momentáneos).
                current_diameters = np.array(fruit_history[track_id]['diameters'])
                robust_diameter = np.median(current_diameters)

                # Para el color, promediamos a lo largo del tiempo para suavizar brillos
                current_colors = np.array(fruit_history[track_id]['colors'])
                robust_color_bgr = np.mean(current_colors, axis=0).astype(int)
                
                # Recalcular estado de madurez basado en el color PROMEDIO TEMPORAL
                # Convertimos el promedio BGR a HSV para clasificación
                pixel_dummy = np.uint8([[robust_color_bgr]]) # Pixel falso para convertir
                pixel_hsv = cv2.cvtColor(pixel_dummy, cv2.COLOR_BGR2HSV)[0][0]
                hue = pixel_hsv[0]
                
                # Lógica de madurez (Misma que features.py pero aplicada al promedio)
                if hue < 10: ripeness_status = "Madura"
                elif 10 <= hue < 25: ripeness_status = "Madura"
                elif 25 <= hue < 45: ripeness_status = "Envero"
                else: ripeness_status = "Verde"

                # --- 4. VISUALIZACIÓN ---
                # Dibujar caja
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Dibujar ID y Tamaño Estabilizado
                label = f"ID:{track_id} D:{int(robust_diameter)}px"
                cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Dibujar Madurez Estabilizada
                cv2.putText(annotated_frame, ripeness_status, (x1, y2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # Tracking visual (gusanito)
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                track = track_history[track_id]
                track.append(center)
                if len(track) > 30: track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(255, 255, 0), thickness=2)

        # Info global
        cv2.putText(annotated_frame, f"Frutos Unicos: {len(fruit_history)}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        video_writer.write(annotated_frame)
        cv2.imshow("Tracking con Filtrado Estadistico", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

    # --- POST-PROCESAMIENTO FINAL ---
    print("\n📊 Consolidando datos estadísticos finales...")
    final_data = []
    
    for fid, history in fruit_history.items():
        # Filtro de calidad: Si la naranja apareció menos de 5 frames, es ruido o error.
        if history['frames'] < 5:
            continue
            
        # 1. Estadística de Tamaño: Mediana (Robusta a blur)
        final_diameter = np.median(history['diameters'])
        
        # 2. Estadística de Color: Media (Suaviza ruido de sensor)
        mean_bgr = np.mean(history['colors'], axis=0).astype(np.uint8)
        
        # Convertir a HSV final para reporte
        pixel = np.uint8([[mean_bgr]])
        hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
        final_hue = hsv[0]
        
        # Clasificación Final
        if final_hue < 25: 
            clase = "Madura"
        elif final_hue < 45: 
            clase = "Envero"
        else: 
            clase = "Verde"
            
        final_data.append({
            "Orange_ID": fid,
            "Frames_Tracked": history['frames'],
            "Diameter_Pixel_Median": round(final_diameter, 2),
            "Hue_Mean": float(final_hue),
            "Ripeness_Final": clase
        })
    
    if final_data:
        df = pd.DataFrame(final_data)
        df.to_csv(output_csv_path, index=False)
        print(f"✅ Reporte generado: {output_csv_path}")
        print(df['Ripeness_Final'].value_counts())
    else:
        print("⚠️ No se detectaron frutos con suficiente permanencia en pantalla.")

if __name__ == "__main__":
    # Rutas Hardcodeadas para prueba local rápida
    MODEL = "Modelos de YOLO a USAR/bestleomsgd.pt"
    VIDEO = "D:/Universidad/Clases_2025/PDI/Trabajo Final/Tp_Final/Videos_de_prueba/vid_prueba.mp4"
    
    process_video_agronomic(MODEL, VIDEO, "video_output_filtrado.mp4", "reporte_final_robusto.csv")