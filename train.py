from ultralytics import YOLO
import torch
import yaml
import os

def run_training(config_path):
    """
    Ejecuta el entrenamiento de YOLOv8 con optimización de hardware y augmentación avanzada.
    """
    # 1. Cargar Configuración
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    print(f"--- INICIANDO EXPERIMENTO: {cfg['experiment_name']} ---")
    print(f"Hardware: Batch={cfg['batch_size']}, Workers={cfg['workers']}, Cache={cfg['cache']}")
    
    # 2. Configurar Semillas
    torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg['seed'])
    
    # 3. Instanciar Modelo
    model = YOLO(cfg['model_version']) 
    
    # 4. Ejecutar Entrenamiento
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
        cache=cfg['cache'],      # <--- Carga en RAM activada
        
        # Optimizador
        optimizer=cfg['optimizer'],
        lr0=cfg['lr0'],
        momentum=cfg['momentum'],
        weight_decay=cfg['weight_decay'],
        
        # Augmentación Geométrica
        degrees=cfg['degrees'],
        translate=cfg['translate'], # <--- Nuevo
        scale=cfg['scale'],         # <--- Nuevo
        fliplr=cfg['fliplr'],
        flipud=cfg['flipud'],
        
        # Augmentación de Pixel/Composición
        mosaic=cfg['mosaic'],
        mixup=cfg['mixup'],         # <--- Nuevo
        copy_paste=cfg['copy_paste'], # <--- Nuevo
        
        # Configuración General
        exist_ok=True,
        plots=True,
        val=True,
        verbose=True
    )
    
    # 5. Retornar ruta del mejor modelo
    # Ultralytics guarda en: project/name/weights/best.pt
    # Nota: En colab, 'project' suele ser relativo al CWD.
    export_path = os.path.join(cfg['project_name'], cfg['experiment_name'], 'weights', 'best.pt')
    
    print(f"--- FIN DEL ENTRENAMIENTO ---")
    print(f"Mejor modelo guardado teóricamente en: {export_path}")
    
    return model, export_path