from ultralytics import YOLO
import torch
import yaml
import os

def run_training(config_path):
    """
    Ejecuta el entrenamiento de YOLOv8 con optimización de hardware, 
    augmentación avanzada (Geométrica + Fotométrica) y estrategia SGD.
    """
    # 1. Cargar Configuración
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    print(f"--- INICIANDO EXPERIMENTO: {cfg['experiment_name']} ---")
    print(f"Hardware: Batch={cfg['batch_size']}, Workers={cfg['workers']}, Cache={cfg['cache']}")
    print(f"Optimizador: {cfg['optimizer']} | HSV Augmentation: Activada")
    
    # 2. Configurar Semillas para reproducibilidad
    torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg['seed'])
    
    # 3. Instanciar Modelo
    model = YOLO(cfg['model_version']) 
    
    # 4. Ejecutar Entrenamiento
    # Pasamos explícitamente los parámetros del config.yaml al motor de entrenamiento
    results = model.train(
        # Datos y Proyecto
        data=cfg['data_yaml_path'],
        project=cfg['project_name'],
        name=cfg['experiment_name'],
        
        # Hardware y Tiempos
        epochs=cfg['epochs'],
        batch=cfg['batch_size'],
        imgsz=cfg['img_size'],
        workers=cfg['workers'],
        device=cfg['device'],
        patience=cfg['patience'],
        cache=cfg['cache'],      
        
        # Optimizador y Learning Rate (SGD Táctico)
        optimizer=cfg['optimizer'],
        lr0=cfg['lr0'],
        lrf=cfg['lrf'],           # Tasa final añadida
        momentum=cfg['momentum'],
        weight_decay=cfg['weight_decay'],
        
        # --- NUEVO: Augmentación Fotométrica (HSV) ---
        # Vital para detectar naranjas bajo sol, sombra o diferentes madureces
        hsv_h=cfg['hsv_h'],       # Tono
        hsv_s=cfg['hsv_s'],       # Saturación
        hsv_v=cfg['hsv_v'],       # Valor (Brillo)
        
        # Augmentación Geométrica
        degrees=cfg['degrees'],
        translate=cfg['translate'], 
        scale=cfg['scale'],         
        fliplr=cfg['fliplr'],
        flipud=cfg['flipud'],
        
        # Augmentación de Composición
        mosaic=cfg['mosaic'],
        mixup=cfg['mixup'],         
        copy_paste=cfg['copy_paste'], 
        
        # Configuración General
        exist_ok=True,
        plots=True,
        val=True,
        verbose=True
    )
    
    # 5. Retornar ruta del mejor modelo
    export_path = os.path.join(cfg['project_name'], cfg['experiment_name'], 'weights', 'best.pt')
    
    print(f"--- FIN DEL ENTRENAMIENTO ---")
    print(f"Mejor modelo guardado en: {export_path}")
    
    return model, export_path

if __name__ == "__main__":
    # Para pruebas locales rápidas
    if os.path.exists("config.yaml"):
        run_training("config.yaml")