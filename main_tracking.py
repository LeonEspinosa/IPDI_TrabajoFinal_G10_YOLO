import argparse
import os
import sys
from pathlib import Path
import cv2

# Importar lógica de tracking
try:
    from track_video import process_video_tracking
except ImportError:
    print("❌ ERROR: No se encuentra 'track_video.py'. Asegúrate de estar en la carpeta correcta.")
    sys.exit(1)

# ==========================================
#  CONFIGURACIÓN DE RUTAS POR DEFECTO
#  (Edita esto para ejecutar con el botón Play)
# ==========================================
DEFAULT_CONFIG = {
    # Usa r"" para evitar problemas con las barras invertidas en Windows
    "video": r"D:\Universidad\Clases_2025\PDI\Trabajo Final\Tp_Final\Videos_de_prueba\prueba1.mp4",
    
    "model": r"Modelos de YOLO a USAR\v8nbestleomsgd.pt",
    
    "tracker": r"Tracking\botsort_custom.yaml",
    
    "show_live": True  # Poner en False si usas Colab
}
# ==========================================

def main():
    print("\n🍊 === SISTEMA DE TRACKING DIRECTO (IPDI G10) === 🍊")

    # Intentar leer argumentos de consola
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, help="Ruta al video")
    parser.add_argument('--model', type=str, help="Ruta al modelo")
    parser.add_argument('--tracker', type=str, help="Ruta al config del tracker")
    parser.add_argument('--no-show', action='store_true', help="Desactivar ventana en vivo")
    args, unknown = parser.parse_known_args()

    # Lógica de Selección: Argumento > Defecto
    video_path = args.video if args.video else DEFAULT_CONFIG["video"]
    model_path = args.model if args.model else DEFAULT_CONFIG["model"]
    tracker_path = args.tracker if args.tracker else DEFAULT_CONFIG["tracker"]
    show_live = not args.no_show if args.video else DEFAULT_CONFIG["show_live"]

    # --- Validaciones ---
    if not os.path.exists(video_path):
        print(f"❌ ERROR: No se encuentra el video:\n   {video_path}")
        print("-> Revisa la variable DEFAULT_CONFIG en main_tracking.py")
        return

    if not os.path.exists(model_path):
        print(f"❌ ERROR: No se encuentra el modelo:\n   {model_path}")
        print("-> Revisa la ruta o entrena el modelo primero.")
        return

    # Validar Tracker
    if not os.path.exists(tracker_path):
        print(f"⚠️ AVISO: No se encuentra '{tracker_path}'.")
        print("   Se usará la configuración por defecto de YOLO (puede generar duplicados).")
        # No detenemos el programa, dejamos que YOLO use su default
    
    # Construir ruta de salida
    path_obj = Path(video_path)
    output_name = f"resultado_{path_obj.stem}.mp4"
    output_path = path_obj.parent / output_name

    print(f"▶️  Entrada: {path_obj.name}")
    print(f"🧠 Modelo:  {Path(model_path).name}")
    print(f"💾 Salida:  {output_path}")
    print("-" * 40)

    # Ejecutar
    try:
        process_video_tracking(
            model_path=str(model_path),
            input_video_path=str(video_path),
            output_video_path=str(output_path),
            tracker_type=str(tracker_path),
            show_live=show_live
        )
    except KeyboardInterrupt:
        print("\n🛑 Proceso detenido por el usuario.")
    except Exception as e:
        print(f"\n❌ Ocurrió un error inesperado: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()