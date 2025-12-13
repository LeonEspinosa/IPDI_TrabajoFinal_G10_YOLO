import cv2
import matplotlib.pyplot as plt
import os
import random
import yaml
from pathlib import Path

def check_dataset_sanity(data_yaml_path, num_samples=3):
    """
    Dibuja las cajas de etiquetas sobre las imágenes originales para verificar
    que el dataset no esté corrupto.
    """
    print(f"[INFO] Verificando sanidad del dataset desde: {data_yaml_path}")
    
    # Cargar rutas del yaml
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Asumimos que la ruta 'train' en el yaml es relativa a la ubicación del yaml o absoluta
    # Ajusta esta lógica según tu estructura exacta de carpetas
    train_path = Path(data_config['train'])
    if not train_path.is_absolute():
        base_path = Path(data_yaml_path).parent.parent # Subir niveles si es necesario
        train_path = base_path / train_path

    # Obtener lista de imágenes
    image_files = list(train_path.glob("*.jpg")) + list(train_path.glob("*.png"))
    
    if not image_files:
        print("[ERROR] No se encontraron imágenes. Revisa la ruta en data.yaml")
        return

    # Seleccionar muestras aleatorias
    samples = random.sample(image_files, min(num_samples, len(image_files)))

    plt.figure(figsize=(15, 5))
    for i, img_path in enumerate(samples):
        # Leer imagen
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        # Buscar archivo de etiqueta correspondiente
        label_path = img_path.parent.parent / 'labels' / (img_path.stem + '.txt')
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = list(map(float, line.strip().split()))
                cls = int(parts[0])
                # YOLO format: x_center, y_center, width, height (normalizados 0-1)
                x_c, y_c, bw, bh = parts[1], parts[2], parts[3], parts[4]
                
                # Desnormalizar a pixeles
                x1 = int((x_c - bw / 2) * w)
                y1 = int((y_c - bh / 2) * h)
                x2 = int((x_c + bw / 2) * w)
                y2 = int((y_c + bh / 2) * h)
                
                # Dibujar caja (Verde Agronómico)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            print(f"[WARN] Etiqueta no encontrada para {img_path.name}")

        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img)
        plt.title(img_path.name)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    print("[INFO] Verificación visual completada. Si las cajas están desplazadas, DETÉN EL PROCESO.")