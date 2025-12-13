import cv2
import supervision as sv
from ultralytics import YOLO
import argparse
import os
import numpy as np

def process_video_tracking(model_path, input_video_path, output_video_path, tracker_type="bytetrack.yaml", show_live=True):
    """
    Aplica tracking a un video localmente usando la librería Supervision para visualización profesional.
    Args:
        tracker_type: 'bytetrack.yaml' o 'botsort.yaml'
    """
    # Limpiar consola
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print(f"=== SISTEMA DE TRACKING DE NARANJAS (MEJORADO CON SUPERVISION) ===")
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

    # 1. Cargar Modelo YOLOv8
    print("⏳ Cargando modelo YOLOv8...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return

    # 2. Configurar Annotators de Supervision
    # BoxAnnotator: Dibuja las cajas bounding boxes
    # LabelAnnotator: Dibuja las etiquetas y confianza
    # TraceAnnotator: Dibuja la estela del movimiento (Tracking)
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        color=sv.ColorPalette.DEFAULT
    )
    
    label_annotator = sv.LabelAnnotator(
        text_scale=0.5,
        text_thickness=1,
        text_padding=10
    )
    
    trace_annotator = sv.TraceAnnotator(
        thickness=2,
        trace_length=30, # Longitud de la estela
        position=sv.Position.CENTER
    )

    # 3. Procesamiento de Video
    # Usamos VideoInfo para obtener metadatos y VideoSink para guardar eficientemente
    video_info = sv.VideoInfo.from_video_path(input_video_path)
    print(f"ℹ️ Info Video: {video_info.width}x{video_info.height} @ {video_info.fps} FPS")

    # Contador de IDs únicos
    unique_ids_detected = set()

    print("🚀 Iniciando inferencia...")

    # Generador de frames con Supervision
    # Esto reemplaza el bucle while cap.isOpened() tradicional para una gestión más limpia
    frame_generator = sv.get_video_frames_generator(input_video_path)
    
    # Context manager para guardar el video
    with sv.VideoSink(target_path=output_video_path, video_info=video_info) as sink:
        for frame in frame_generator:
            
            # 4. Inferencia y Tracking con Ultralytics
            # persist=True es vital para mantener los IDs entre frames
            results = model.track(frame, persist=True, tracker=tracker_type, conf=0.45, iou=0.5, verbose=False)[0]

            # 5. Convertir resultados a formato Supervision (sv.Detections)
            detections = sv.Detections.from_ultralytics(results)
            
            # Filtrar detecciones que tengan Tracker ID (a veces se pierden momentáneamente)
            if detections.tracker_id is not None:
                # Actualizar conjunto de IDs únicos para el conteo global
                unique_ids_detected.update(detections.tracker_id)

                # 6. Anotación (Dibujado)
                # Orden: Trazas (fondo) -> Cajas -> Etiquetas (frente)
                annotated_frame = trace_annotator.annotate(
                    scene=frame.copy(),
                    detections=detections
                )
                
                annotated_frame = box_annotator.annotate(
                    scene=annotated_frame,
                    detections=detections
                )
                
                # Crear etiquetas personalizadas: "#ID Cladse Conf"
                labels = [
                    f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
                    for tracker_id, class_id, confidence
                    in zip(detections.tracker_id, detections.class_id, detections.confidence)
                ]
                
                annotated_frame = label_annotator.annotate(
                    scene=annotated_frame,
                    detections=detections,
                    labels=labels
                )
            else:
                annotated_frame = frame

            # Info en pantalla (Dashboard simple)
            rect_color = (0, 0, 0)
            cv2.rectangle(annotated_frame, (0, 0), (video_info.width, 50), rect_color, -1)
            info_text = f"Algoritmo: {tracker_type.split('.')[0].upper()} | Conteo Total: {len(unique_ids_detected)}"
            cv2.putText(annotated_frame, info_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Guardar frame procesado
            sink.write_frame(annotated_frame)

            # Mostrar en vivo (Solo si se solicita y hay entorno gráfico)
            if show_live:
                cv2.imshow("Tracking Supervision - IPDI", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cv2.destroyAllWindows()
    
    print(f"\n✅ PROCESO COMPLETADO.")
    print(f"📊 Producción Estimada: {len(unique_ids_detected)} naranjas.")
    print(f"💾 Video guardado en: {output_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="Ruta al archivo .pt del modelo")
    parser.add_argument('--source', type=str, required=True, help="Ruta al video de entrada")
    parser.add_argument('--output', type=str, default='resultado_supervision.mp4', help="Ruta de salida")
    parser.add_argument('--method', type=str, default='bytetrack.yaml', help="Método de tracking")
    
    args = parser.parse_args()
    
    process_video_tracking(args.model, args.source, args.output, args.method)