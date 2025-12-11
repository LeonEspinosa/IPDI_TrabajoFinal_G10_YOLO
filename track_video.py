import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import argparse
import os

def process_video_tracking(model_path, input_video_path, output_video_path, tracker_type="bytetrack.yaml", show_live=True):
    """
    Aplica tracking a un video localmente.
    Args:
        tracker_type: 'bytetrack.yaml' o 'botsort.yaml'
    """
    # Limpiar consola
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print(f"=== SISTEMA DE TRACKING DE NARANJAS (LOCAL) ===")
    print(f"📂 Modelo: {model_path}")
    print(f"🎞️ Video: {input_video_path}")
    print(f"🎯 Algoritmo: {tracker_type}")
    print(f"---------------------------------------------")

    if not os.path.exists(model_path):
        print(f"❌ ERROR: No se encuentra el modelo en: {model_path}")
        return
    if not os.path.exists(input_video_path):
        print(f"❌ ERROR: No se encuentra el video en: {input_video_path}")
        return

    # 1. Cargar Modelo
    print("⏳ Cargando modelo YOLOv8...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return

    # 2. Abrir Video
    cap = cv2.VideoCapture(input_video_path)
    
    # Propiedades
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 3. Salida
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # Historial
    track_history = defaultdict(lambda: [])
    unique_ids_detected = set()

    print("🚀 Iniciando inferencia. Presiona 'q' en la ventana de video para salir antes.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 4. TRACKING
        # conf=0.45 filtra falsos positivos (caras)
        # iou=0.5 ayuda si las naranjas están muy juntas
        results = model.track(frame, persist=True, tracker=tracker_type, conf=0.45, iou=0.5, verbose=False)

        annotated_frame = results[0].plot() # Dibuja cajas e IDs

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x, y, w_box, h_box = box
                unique_ids_detected.add(track_id)

                # Dibujar estela
                track = track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 30: 
                    track.pop(0)

                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 255, 255), thickness=2)

        # Info en pantalla
        info_text = f"Algoritmo: {tracker_type.split('.')[0].upper()} | Conteo Total: {len(unique_ids_detected)}"
        cv2.rectangle(annotated_frame, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.putText(annotated_frame, info_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Guardar
        video_writer.write(annotated_frame)

        # Mostrar en vivo (Solo en PC local)
        if show_live:
            cv2.imshow("Tracking en Tiempo Real - IPDI", annotated_frame)
            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ PROCESO COMPLETADO.")
    print(f"📊 Producción Estimada: {len(unique_ids_detected)} naranjas.")
    print(f"💾 Video guardado en: {output_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--output', type=str, default='resultado_local.mp4')
    parser.add_argument('--method', type=str, default='bytetrack.yaml')
    
    args = parser.parse_args()
    
    process_video_tracking(args.model, args.source, args.output, args.method)