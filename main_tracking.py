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

# Importar módulo de Descriptores
try:
    from Descriptores import basicos, contorno, factores_forma, topologicos, preprocesamiento
    DESCRIPTORES_AVAILABLE = True
except ImportError:
    print("⚠️ ADVERTENCIA: No se pudieron importar los módulos de 'Descriptores'.")
    DESCRIPTORES_AVAILABLE = False

# ==========================================
#  CONFIGURACIÓN DE RUTAS POR DEFECTO
# ==========================================
DEFAULT_CONFIG = {
    "video": r"D:\Universidad\Clases_2025\PDI\Trabajo Final\Tp_Final\Videos_de_prueba\vid_prueba.mp4",
    "model": r"Modelos de YOLO a USAR\v11nbestleom.pt",
    "tracker": r"Tracking\botsort_custom.yaml",
    "show_live": True
}

# ==========================================
#  FUNCIONES DE PROCESAMIENTO
# ==========================================

def segmentar_naranja(imagen_bgr):
    """
    Aplica una máscara binaria para aislar la naranja del fondo usando HSV.
    
    Returns:
        tuple: (imagen_aislada, mascara_binaria)
    """
    hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)
    
    # Rangos de color para naranjas (ajustar según iluminación)
    lower_orange = np.array([10, 100, 20])
    upper_orange = np.array([25, 255, 255])
    
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Limpieza morfológica
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Aplicar máscara (fondo negro)
    resultado = cv2.bitwise_and(imagen_bgr, imagen_bgr, mask=mask)
    return resultado, mask


def calcular_descriptores(track_id, imagen_aislada, mascara):
    """
    Calcula descriptores de forma y color sobre la imagen aislada.
    
    Returns:
        dict: Diccionario con los descriptores calculados
    """
    datos = {'Track_ID': track_id}
    
    # 1. Descriptores de Color
    try:
        media_color = cv2.mean(imagen_aislada, mask=mascara)[:3]
        datos['Color_B_Mean'] = round(media_color[0], 2)
        datos['Color_G_Mean'] = round(media_color[1], 2)
        datos['Color_R_Mean'] = round(media_color[2], 2)
    except Exception as e:
        print(f"  ⚠️ Error en descriptores de color ID {track_id}: {e}")

    # 2. Descriptores de Forma
    try:
        contours, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            perimetro = cv2.arcLength(cnt, True)
            
            datos['Area_px'] = int(area)
            datos['Perimetro_px'] = round(perimetro, 2)
            
            # Compacidad
            if area > 0:
                datos['Compacidad'] = round((perimetro ** 2) / (4 * np.pi * area), 4)
            else:
                datos['Compacidad'] = 0
            
            # Elipse (Ejes Mayor y Menor)
            if len(cnt) >= 5:
                (e_center, e_axes, e_angle) = cv2.fitEllipse(cnt)
                datos['Eje_Mayor'] = round(max(e_axes), 2)
                datos['Eje_Menor'] = round(min(e_axes), 2)
                
                # Excentricidad
                if max(e_axes) > 0:
                    datos['Excentricidad'] = round(min(e_axes) / max(e_axes), 4)
                else:
                    datos['Excentricidad'] = 0
                    
    except Exception as e:
        print(f"  ⚠️ Error en descriptores de forma ID {track_id}: {e}")
        
    return datos


# ==========================================
#  MAIN
# ==========================================

def main():
    print("\n🍊 === SISTEMA DE TRACKING Y ANÁLISIS (IPDI G10) === 🍊")

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
                # Actualizar buffer con el frame original (sin anotaciones)
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
    #  FASE 2: SELECCIÓN, AISLAMIENTO Y DESCRIPTORES
    # ==========================================
    
    print(f"\n🔬 FASE 2: Análisis de Descriptores...")
    print("   (Aislamiento de fondo + Cálculo de descriptores)")
    
    resultados_finales = []
    
    for track_id, data in best_shots.items():
        print(f"   -> Procesando ID {track_id} (conf: {data['conf']:.2f}, frame: {data['frame_idx']})...", end="")
        
        best_img = data['image']
        
        # 1. AISLAMIENTO: Segmentar la naranja del fondo
        img_aislada, mascara = segmentar_naranja(best_img)
        
        # Guardar imágenes para visualización
        cv2.imwrite(str(output_dir_crops / f"ID_{track_id}_original.jpg"), best_img)
        cv2.imwrite(str(output_dir_crops / f"ID_{track_id}_aislada.jpg"), img_aislada)
        cv2.imwrite(str(output_dir_crops / f"ID_{track_id}_mascara.jpg"), mascara)
        
        # 2. DESCRIPTORES: Calcular sobre la imagen aislada
        descriptores = calcular_descriptores(track_id, img_aislada, mascara)
        
        # Agregar metadatos del tracking
        descriptores['Frame_Idx'] = data['frame_idx']
        descriptores['Confianza'] = round(data['conf'], 4)
        descriptores['BBox_X1'] = data['bbox'][0]
        descriptores['BBox_Y1'] = data['bbox'][1]
        descriptores['BBox_X2'] = data['bbox'][2]
        descriptores['BBox_Y2'] = data['bbox'][3]
        
        resultados_finales.append(descriptores)
        print(" ✓")

    # ==========================================
    #  FASE 3: GUARDAR RESULTADOS
    # ==========================================
    
    print(f"\n💾 FASE 3: Guardando Resultados...")
    
    # Guardar CSV con descriptores
    if resultados_finales:
        df = pd.DataFrame(resultados_finales)
        csv_path = output_dir_data / "descriptores_naranjas.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"✅ CSV guardado en: {csv_path}")
        print(f"✅ Imágenes guardadas en: {output_dir_crops}")
        print(f"\n📊 Resumen de Análisis ({len(resultados_finales)} naranjas):")
        print("-" * 60)
        print(df.to_string(index=False))
        print("-" * 60)
    else:
        print("\n⚠️ No se generaron datos de análisis.")

    print("\n🎉 ¡Proceso completado exitosamente!")


if __name__ == "__main__":
    main()