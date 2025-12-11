from ultralytics import YOLO
import torch
import yaml
import os

def run_training(config_path):
    """
    Ejecuta el entrenamiento de YOLOv8 basado en el archivo de configuración.
    """
    # 1. Cargar Configuración
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    print(f"--- INICIANDO PROTOCOLO DE ENTRENAMIENTO: {cfg['experiment_name']} ---")
    
    # 2. Configurar Semillas (Reproducibilidad Científica)
    torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg['seed'])
    
    # 3. Instanciar Modelo
    # Carga pesos pre-entrenados para Transfer Learning
    model = YOLO(cfg['model_version']) 
    
    # 4. Ejecutar Entrenamiento
    # Pasamos los parámetros del YAML desempaquetados
    results = model.train(
        data=cfg['data_yaml_path'],
        project=cfg['project_name'],
        name=cfg['experiment_name'],
        epochs=cfg['epochs'],
        batch=cfg['batch_size'],
        imgsz=cfg['img_size'],
        optimizer=cfg['optimizer'],
        lr0=cfg['lr0'],
        momentum=cfg['momentum'],
        weight_decay=cfg['weight_decay'],
        patience=cfg['patience'],
        device=cfg['device'],
        workers=cfg['workers'],
        
        # Augmentation parameters from config
        degrees=cfg['degrees'],
        fliplr=cfg['fliplr'],
        flipud=cfg['flipud'],
        mosaic=cfg['mosaic'],
        mixup=cfg['mixup'],
        
        exist_ok=True,
        plots=True,   # Generar curvas de pérdida y matrices de confusión
        val=True      # Validar tras cada época
    )
    
    # 5. Exportar para inferencia futura
    export_path = f"{cfg['project_name']}/{cfg['experiment_name']}/weights/best.pt"
    print(f"--- ENTRENAMIENTO FINALIZADO ---")
    print(f"Modelo guardado en: {export_path}")
    
    return model, export_path