import cv2
from ultralytics import YOLO
import numpy as np
import argparse
import os
import pandas as pd
from features import extract_yiq_features, estimate_diameter_pixels
from shape_descriptors import ShapeDescriptorProcessor

def main():
    # Rutas (Cámbialas según tu PC)
    MODEL_PATH = "Modelos de YOLO a USAR/bestfranco.pt"
    VIDEO_PATH = "D:/Universidad/Clases_2025/PDI/Trabajo Final/Tp_Final/Videos_de_prueba/vid_prueba.mp4"
    OUTPUT_VIDEO = "resultado_yiq_shape.mp4"
    OUTPUT_CSV = "reporte_yiq_shape.csv"

    # Inicializar
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # Shape Processor
    shape_processor = ShapeDescriptorProcessor()
    # Activamos lo que nos interesa ver
    shape_processor.visualization_layers['skeleton'] = True
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    data_log = []

    print("🚀 Iniciando Tracking Avanzado (YIQ + Shape Descriptors)...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.45, verbose=False)
        annotated_frame = results[0].plot()

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                
                # 1. Recortar la naranja para análisis detallado
                orange_roi = frame[y1:y2, x1:x2]
                if orange_roi.size == 0: continue

                # 2. Análisis de Color (YIQ)
                (Y_val, I_val, Q_val), madurez = extract_yiq_features(frame, box.numpy())

                # 3. Análisis de Forma (Shape Descriptor Class)
                # Cargamos el recorte en el procesador
                if shape_processor.process_numpy_image(orange_roi):
                    shape_processor.generate_skeleton() # Generar esqueleto
                    shape_results = shape_processor.analyze_shape() # Calcular métricas
                    
                    if shape_results:
                        metrics = shape_results[0]['metrics']
                        factors = shape_results[0]['factors']
                        
                        # Guardar datos
                        data_log.append({
                            'Frame': int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                            'ID': track_id,
                            'Y': round(Y_val, 3),
                            'I': round(I_val, 3), # EL MÁS IMPORTANTE
                            'Q': round(Q_val, 3),
                            'Madurez': madurez,
                            'Area_px': metrics['Area'],
                            'Compacidad': factors['Compactness'],
                            'Skeleton_Len': metrics['Skeleton_Len']
                        })
                        
                        # --- VISUALIZACIÓN EN VIVO SOBRE EL VIDEO ---
                        # Escribir valor 'I' (Madurez)
                        cv2.putText(annotated_frame, f"I:{I_val:.2f}", (x1, y1-25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        
                        # Opcional: Dibujar el esqueleto sobre la naranja en el video principal
                        # (Requiere mapeo de coordenadas del recorte al frame global)
                        # Esto es complejo visualmente, mejor confiar en el CSV o ventana debug.

        video_writer.write(annotated_frame)
        cv2.imshow("Tracking YIQ + Shape", annotated_frame)
        
        # Ventana de depuración: Ver lo que ve el Shape Processor (Binario + Esqueleto)
        if shape_processor.binary is not None:
             cv2.imshow("Debug: Binary/Skeleton", shape_processor.binary)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    
    # Guardar CSV
    if data_log:
        pd.DataFrame(data_log).to_csv(OUTPUT_CSV, index=False)
        print(f"✅ Reporte guardado: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()