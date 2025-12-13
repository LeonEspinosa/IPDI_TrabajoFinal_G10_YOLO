import sys
import traceback

try:
    print("Importando librerías...")
    from ultralytics import YOLO
    import torch
    
    print(f"✅ Ultralytics version: {YOLO.__module__}")
    print(f"✅ PyTorch version: {torch.__version__}")
    
    model_path = r"Modelos de YOLO a USAR\v11nbestleom.pt"
    video_path = r"D:\Universidad\Clases_2025\PDI\Trabajo Final\Tp_Final\Videos_de_prueba\vid_prueba.mp4"
    
    print(f"\n📂 Cargando modelo: {model_path}")
    model = YOLO(model_path)
    print(f"✅ Modelo cargado exitosamente")
    print(f"   Tipo: {type(model)}")
    print(f"   Nombres de clases: {model.names}")
    
    print(f"\n🎬 Probando tracking en video...")
    results = model.track(
        source=video_path,
        save=True,
        tracker="botsort.yaml",
        conf=0.45,
        iou=0.5,
        show=False,
        verbose=True
    )
    
    print(f"\n✅ Tracking completado exitosamente!")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\n📋 Traceback completo:")
    traceback.print_exc()
    sys.exit(1)
