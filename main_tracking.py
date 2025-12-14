import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import os
import sys
import argparse
from pathlib import Path

# Importar módulo ROIExtractor
sys.path.append(os.getcwd())
try:
    from ROIExtractor import TrackBuffer, save_best_crops
except ImportError:
    print("❌ ERROR: No se encuentra el módulo 'ROIExtractor'.")
    sys.exit(1)

# ==========================================
#  CONFIGURACIÓN DE RUTAS POR DEFECTO
# ==========================================
DEFAULT_CONFIG = {
    "video": r"D:\Universidad\Clases_2025\PDI\Trabajo Final\Tp_Final\Videos_de_prueba\prueba2.mp4",
    "model": r"Modelos de YOLO a USAR\v11nbestleom.pt",
    "tracker": r"Tracking\botsort_custom.yaml",
    "show_live": True
}

# ==========================================
#  MAIN
# ==========================================

def main():
    print("\n🍊 === SISTEMA DE TRACKING (Etapa 1: Extracción) === 🍊")

    # --- Argument Parser ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, help="Ruta al video")
    parser.add_argument('--model', type=str, help="Ruta al modelo")
    parser.add_argument('--tracker', type=str, help="Ruta al config del tracker")
    parser.add_argument('--no-show', action='store_true', help="Desactivar ventana en vivo")
    args, unknown = parser.parse_known_args()

    # --- Configuración Final ---
    video_path = args.video if args.video else DEFAULT_CONFIG["video"]
    model_path = args.model if args.model else DEFAULT_CONFIG["model"]
    tracker_path = args.tracker if args.tracker else DEFAULT_CONFIG["tracker"]
    show_live = not args.no_show if args.video else DEFAULT_CONFIG["show_live"]

    # --- Validaciones ---
    if not os.path.exists(video_path):
        print(f"❌ ERROR: No se encuentra el video: {video_path}")
        return
    if not os.path.exists(model_path):
        print(f"❌ ERROR: No se encuentra el modelo: {model_path}")
        return

    # Preparar directorios de salida
    path_obj = Path(video_path)
    output_video_name = f"resultado_{path_obj.stem}.mp4"
    output_video_path = str(path_obj.parent / output_video_name)
    
    output_dir_data = path_obj.parent / "output_results"
    output_dir_crops = output_dir_data / "best_crops"
    os.makedirs(output_dir_crops, exist_ok=True)

    print(f"▶️  Entrada: {path_obj.name}")
    print(f"🧠 Modelo:  {Path(model_path).name}")
    print(f"💾 Video Salida: {output_video_path}")
    print(f"📊 Datos Salida: {output_dir_data}")
    print("-" * 60)

    # --- Cargar Modelo ---
    try:
        model = YOLO(model_path)
        print("✅ Modelo cargado exitosamente")
    except Exception as e:
        print(f"❌ Error crítico cargando modelo: {e}")
        return

    # --- Inicializar Video ---
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"ℹ️  Video Info: {width}x{height} @ {fps} FPS, {total_frames} frames")
    
    # Inicializar VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # ==========================================
    #  FASE 1: TRACKING Y BUFFERING
    # ==========================================
    
    # Inicializar Buffer
    track_buffer = TrackBuffer()
    
    print("\n🚀 FASE 1: Tracking y Buffering...")
    print("   (Guardando el mejor frame de cada naranja detectada)")
    
    frame_idx = 0
    tracker_arg = tracker_path if os.path.exists(tracker_path) else None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Tracking con YOLO
        results = model.track(frame, persist=True, tracker=tracker_arg, verbose=False, conf=0.45)
        
        # Dibujar resultados en el frame para el video de salida
        annotated_frame = results[0].plot()
        
        # Extraer detecciones y actualizar buffer
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for box, track_id, conf in zip(boxes, track_ids, confidences):
                # Actualizar buffer con el frame original
                track_buffer.update(track_id, frame, box, conf, frame_idx)

        # Guardar frame anotado en video de salida
        out_writer.write(annotated_frame)

        # Mostrar en vivo
        if show_live:
            cv2.imshow("Tracking IPDI G10", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n🛑 Interrumpido por usuario.")
                break
        
        frame_idx += 1

    # Limpieza de recursos de video
    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Video procesado y guardado en: {output_video_path}")
    
    # Obtener resultados del buffer
    best_shots = track_buffer.get_results()
    print(f"📦 Objetos únicos detectados: {len(best_shots)}")
    
    if len(best_shots) == 0:
        print("\n⚠️ No se detectaron objetos. Verifica el modelo y el video.")
        return

    # ==========================================
    #  FASE 2: EXPORTAR METADATA Y CROPS
    # ==========================================
    
    print(f"\n💾 FASE 2: Guardando Crops y Metadata...")
    
    metadata_list = []
    
    for track_id, data in best_shots.items():
        best_img = data['image']
        
        # Nombre del archivo (cambiado a PNG para sin pérdidas)
        filename = f"ID_{track_id}_original.png"
        filepath = output_dir_crops / filename
        
        # Guardar imagen PNG
        cv2.imwrite(str(filepath), best_img)
        
        # Guardar metadata
        metadata = {
            'ID': track_id,
            'Archivo': filename,
            'Confianza': round(data['conf'], 4),
            'Frame_Idx': data['frame_idx'],
            'BBox_X1': data['bbox'][0],
            'BBox_Y1': data['bbox'][1],
            'BBox_X2': data['bbox'][2],
            'BBox_Y2': data['bbox'][3]
        }
        metadata_list.append(metadata)
        
    # Guardar CSV de Metadata
    if metadata_list:
        df_meta = pd.DataFrame(metadata_list)
        # Ordenar columnas
        cols = ['ID', 'Archivo', 'Confianza', 'Frame_Idx', 'BBox_X1', 'BBox_Y1', 'BBox_X2', 'BBox_Y2']
        df_meta = df_meta[cols].sort_values('ID')
        
        csv_path = output_dir_data / "tracking_metadata.csv"
        df_meta.to_csv(csv_path, index=False)
        
        print(f"✅ Metadata guardada en: {csv_path}")
        print(f"✅ Imágenes (PNG) guardadas en: {output_dir_crops}")
    else:
        print("\n⚠️ No se generaron datos.")

    print("\n🎉 ¡Etapa 1 Completada! Ahora ejecuta 'process_descriptors.py'")


if __name__ == "__main__":
    main()