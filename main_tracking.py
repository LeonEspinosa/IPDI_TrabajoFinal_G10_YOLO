import argparse
import os
import sys
from pathlib import Path
import cv2

# Importar lógica de tracking existente
from track_video import process_video_tracking

def main():
    print("\n🍊 === SISTEMA DE TRACKING DEDICADO (IPDI G10) === 🍊")
    print("Este script ejecuta EXCLUSIVAMENTE el seguimiento de objetos.\n")

    # --- Configuración de Argumentos ---
    parser = argparse.ArgumentParser(description="Ejecutor de Tracking para Naranjas")
    
    # Argumentos obligatorios (o con defaults inteligentes si se llama desde .bat)
    parser.add_argument('--video', type=str, required=True, help="Ruta al video de entrada")
    parser.add_argument('--model', type=str, required=True, help="Ruta al modelo entrenado (.pt)")
    parser.add_argument('--tracker', type=str, default='Tracking/botsort_custom.yaml', help="Configuración del tracker (.yaml)")
    parser.add_argument('--show', action='store_true', help="Mostrar ventana de video en vivo (no usar en Colab)")
    
    args = parser.parse_args()

    # --- Validaciones de Rutas ---
    if not os.path.exists(args.video):
        print(f"❌ ERROR: No se encuentra el video: {args.video}")
        sys.exit(1)
        
    if not os.path.exists(args.model):
        print(f"❌ ERROR: No se encuentra el modelo: {args.model}")
        print("   Por favor, entrena el modelo primero o verifica la ruta.")
        sys.exit(1)

    # Verificar configuración del tracker
    # Si es una ruta custom, debe existir. Si es un nombre interno (ej: 'botsort.yaml'), ultralytics lo maneja.
    if os.path.sep in args.tracker and not os.path.exists(args.tracker):
        print(f"⚠️ ADVERTENCIA: No se encuentra la configuración del tracker en: {args.tracker}")
        print("   Se intentará usar la configuración por defecto de Ultralytics.")

    # --- Configuración de Salida ---
    # Guardar el resultado en la misma carpeta que el video original, con prefijo 'resultado_'
    video_path = Path(args.video)
    output_filename = f"resultado_{video_path.stem}.mp4"
    output_path = video_path.parent / output_filename
    
    print(f"▶️  Procesando video: {video_path.name}")
    print(f"🧠 Modelo: {Path(args.model).name}")
    print(f"⚙️  Tracker: {args.tracker}")
    print(f"💾 Salida: {output_path}")
    print("-" * 50)

    # --- Detección de Entorno Gráfico ---
    # Si el usuario pasó --show, intentamos mostrar. Si no, o si es headless, no.
    # Colab/Servidores no tienen DISPLAY.
    is_headless = 'google.colab' in sys.modules or os.environ.get('DISPLAY') is None
    show_live = args.show and not is_headless

    if args.show and is_headless:
        print("⚠️ AVISO: Se solicitó --show pero no se detectó entorno gráfico. Se desactivará la visualización en vivo.")

    # --- Ejecutar Tracking ---
    try:
        process_video_tracking(
            model_path=args.model,
            input_video_path=str(video_path),
            output_video_path=str(output_path),
            tracker_type=args.tracker,
            show_live=show_live
        )
        print("\n✅ Proceso de tracking finalizado exitosamente.")
        
    except Exception as e:
        print(f"\n❌ Ocurrió un error durante el tracking: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()