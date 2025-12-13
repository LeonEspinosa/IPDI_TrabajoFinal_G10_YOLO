import cv2
import supervision as sv
from ultralytics import YOLO
import os
import sys
import torch
from tqdm import tqdm

def process_video_tracking(model_path, input_video_path, output_video_path, tracker_type="bytetrack.yaml", show_live=True):
    print(f"\n🍊 === SISTEMA DE TRACKING (SUPERVISION) ===")
    print(f"📂 Modelo: {os.path.basename(model_path)}")
    print(f"🎞️ Video: {os.path.basename(input_video_path)}")
    print(f"💾 Salida: {output_video_path}")
    print("-" * 40)

    # 1. Cargar Modelo (Safe Globals para PyTorch 2.6+)
    print("⏳ Cargando modelo...")
    try:
        from ultralytics.nn.tasks import DetectionModel
        import torch.nn as nn
        
        # Agregar todas las clases necesarias para PyTorch 2.6+
        torch.serialization.add_safe_globals([
            DetectionModel,
            nn.modules.container.Sequential,
            nn.modules.conv.Conv2d,
            nn.modules.batchnorm.BatchNorm2d,
            nn.modules.activation.SiLU,
            nn.modules.pooling.MaxPool2d,
            nn.modules.upsampling.Upsample,
        ])
        model = YOLO(model_path)
        print("✅ Modelo cargado exitosamente")
    except Exception as e:
        print(f"⚠️ Alerta de carga: {e}. Intentando carga con weights_only=False...")
        try:
            # Fallback: cargar sin restricciones (solo si confías en el modelo)
            import torch
            # Temporalmente deshabilitar weights_only
            original_load = torch.load
            torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})
            model = YOLO(model_path)
            torch.load = original_load
            print("✅ Modelo cargado con weights_only=False")
        except Exception as e2:
            print(f"❌ Error fatal cargando modelo: {e2}")
            return

    # 2. Detectar si es YOLOv11 y usar método nativo si es necesario
    model_name = os.path.basename(model_path).lower()
    use_native_tracking = 'v11' in model_name or 'yolo11' in model_name
    
    if use_native_tracking:
        print("ℹ️ Detectado modelo YOLOv11 - usando tracking nativo de YOLO")
        _process_with_native_yolo(model, input_video_path, output_video_path, tracker_type, show_live)
    else:
        print("ℹ️ Usando tracking con Supervision")
        _process_with_supervision(model, input_video_path, output_video_path, tracker_type, show_live)


def _process_with_native_yolo(model, input_video_path, output_video_path, tracker_type, show_live):
    """Procesar video usando el método nativo de YOLO (compatible con YOLOv11)"""
    
    # Obtener información del video
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    print(f"ℹ️ Video Info: {width}x{height} @ {fps} FPS, {total_frames} frames")
    print("🚀 Iniciando procesamiento...")
    
    # Configurar tracker
    if not os.path.exists(tracker_type):
        print(f"⚠️ Tracker personalizado no encontrado, usando 'botsort.yaml' por defecto")
        tracker_type = "botsort.yaml"
    
    # Procesar con YOLO nativo
    unique_ids = set()
    
    # Configurar video de salida
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Procesar frame por frame
    results_generator = model.track(
        source=input_video_path,
        stream=True,
        tracker=tracker_type,
        conf=0.45,
        iou=0.5,
        verbose=False,
        persist=True
    )
    
    for frame_idx, result in enumerate(tqdm(results_generator, total=total_frames, unit="frames")):
        # Obtener frame anotado
        annotated_frame = result.plot()
        
        # Contar IDs únicos
        if result.boxes.id is not None:
            ids = result.boxes.id.cpu().numpy().astype(int)
            unique_ids.update(ids)
        
        # Agregar dashboard
        cv2.rectangle(annotated_frame, (0, 0), (width, 40), (0, 0, 0), -1)
        info_text = f"Naranjas: {len(unique_ids)} | Frame: {frame_idx + 1}/{total_frames}"
        cv2.putText(annotated_frame, info_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Guardar frame
        out.write(annotated_frame)
        
        # Mostrar en vivo
        if show_live:
            cv2.imshow("Tracking YOLO", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n🛑 Cancelado por usuario.")
                break
    
    out.release()
    cv2.destroyAllWindows()
    
    # Verificación final
    if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
        size_mb = os.path.getsize(output_video_path) / (1024 * 1024)
        print(f"\n✅ ÉXITO TOTAL. Video guardado: {output_video_path}")
        print(f"📊 Tamaño: {size_mb:.2f} MB")
        print(f"🍊 Naranjas Detectadas: {len(unique_ids)}")
    else:
        print(f"\n❌ FALLO: El archivo de video NO se creó o está vacío.")


def _process_with_supervision(model, input_video_path, output_video_path, tracker_type, show_live):
    """Procesar video usando Supervision (para YOLOv8 y anteriores)"""
    
    # Configurar Supervision Annotators
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        color=sv.ColorPalette.DEFAULT
    )
    label_annotator = sv.LabelAnnotator(
        text_scale=0.5,
        text_thickness=1,
        text_padding=5
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=2,
        trace_length=30
    )

    # Leer Información del Video
    try:
        video_info = sv.VideoInfo.from_video_path(input_video_path)
        print(f"ℹ️ Video Info: {video_info.width}x{video_info.height} @ {video_info.fps} FPS")
    except Exception as e:
        print(f"❌ Error leyendo metadatos del video: {e}")
        return

    # Procesamiento
    unique_ids_detected = set()
    frame_generator = sv.get_video_frames_generator(input_video_path)
    
    print("🚀 Iniciando procesamiento...")
    
    try:
        with sv.VideoSink(target_path=output_video_path, video_info=video_info, codec="mp4v") as sink:
            
            for i, frame in tqdm(enumerate(frame_generator), total=video_info.total_frames, unit="frames"):
                
                # Inferencia y Tracking
                results = model.track(frame, persist=True, tracker=tracker_type, conf=0.45, iou=0.5, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(results)
                
                # Solo dibujar si hay detecciones con ID
                if detections.tracker_id is not None:
                    unique_ids_detected.update(detections.tracker_id)
                    
                    # Dibujar Trazas
                    annotated_frame = trace_annotator.annotate(
                        scene=frame.copy(),
                        detections=detections
                    )
                    # Dibujar Cajas
                    annotated_frame = box_annotator.annotate(
                        scene=annotated_frame,
                        detections=detections
                    )
                    # Etiquetas
                    labels = [
                        f"#{tracker_id} {model.model.names[class_id]} {conf:.2f}"
                        for tracker_id, class_id, conf
                        in zip(detections.tracker_id, detections.class_id, detections.confidence)
                    ]
                    annotated_frame = label_annotator.annotate(
                        scene=annotated_frame,
                        detections=detections,
                        labels=labels
                    )
                else:
                    annotated_frame = frame

                # Dashboard en pantalla
                cv2.rectangle(annotated_frame, (0, 0), (video_info.width, 40), (0,0,0), -1)
                info_text = f"Naranjas: {len(unique_ids_detected)} | Frame: {i}/{video_info.total_frames}"
                cv2.putText(annotated_frame, info_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Guardar frame procesado
                sink.write_frame(annotated_frame)

                # Mostrar en vivo (opcional)
                if show_live:
                    cv2.imshow("Tracking Supervision", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("\n🛑 Cancelado por usuario.")
                        break

    except Exception as e:
        print(f"\n❌ Error guardando el video: {e}")
        import traceback
        traceback.print_exc()
        if "codec" in str(e).lower():
            print("💡 PISTA: El codec mp4v falló. Intenta instalar 'opencv-python' completo.")

    cv2.destroyAllWindows()
    
    # Verificación Final
    if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
        size_mb = os.path.getsize(output_video_path) / (1024 * 1024)
        print(f"\n✅ ÉXITO TOTAL. Video guardado: {output_video_path}")
        print(f"📊 Tamaño: {size_mb:.2f} MB")
        print(f"🍊 Naranjas Detectadas: {len(unique_ids_detected)}")
    else:
        print(f"\n❌ FALLO: El archivo de video NO se creó o está vacío (0 KB).")
        print("Revisa permisos de carpeta o la instalación de opencv-python.")


if __name__ == "__main__":
    pass