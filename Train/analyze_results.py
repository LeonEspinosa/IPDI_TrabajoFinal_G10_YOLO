import os
import glob
from ultralytics import YOLO
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_yield_estimation(model_path, data_yaml_path, conf_threshold=0.25):
    """
    Realiza inferencia en el conjunto de TEST y compara el CONTEO real vs predicho.
    Calcula RMSE y R2 (Métricas agronómicas clave).
    """
    print("--- INICIANDO EVALUACIÓN DE ESTIMACIÓN DE PRODUCCIÓN (CONTEO) ---")
    
    # Cargar rutas
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
        
    test_images_path = Path(data_config['path']) / data_config['test'] # Ajustar según tu yaml
    # Nota: Ajusta la ruta de etiquetas de test según tu estructura real
    test_labels_path = test_images_path.parent.parent / "test" / "labels" 

    model = YOLO(model_path)
    
    results_data = []

    # Iterar sobre imágenes de test
    image_files = list(Path(test_images_path).glob("*.[jp][pn]g")) # jpg o png
    
    if not image_files:
        print(f"[ERROR] No se encontraron imágenes de test en {test_images_path}")
        return

    for img_file in image_files:
        # 1. Ground Truth (Conteo manual)
        label_file = test_labels_path / (img_file.stem + ".txt")
        actual_count = 0
        if label_file.exists():
            with open(label_file, 'r') as f:
                actual_count = len(f.readlines()) # 1 línea = 1 naranja
        
        # 2. Predicción (Conteo automático)
        # verbose=False para no ensuciar la consola
        prediction = model.predict(str(img_file), conf=conf_threshold, verbose=False)[0]
        predicted_count = len(prediction.boxes)
        
        # 3. Registrar error
        error = predicted_count - actual_count
        abs_error = abs(error)
        
        results_data.append({
            "Image": img_file.name,
            "Actual": actual_count,
            "Predicted": predicted_count,
            "Error": error,
            "Abs_Error": abs_error
        })

    # Crear DataFrame
    df = pd.DataFrame(results_data)
    
    # Métricas Globales
    rmse = np.sqrt(mean_squared_error(df['Actual'], df['Predicted']))
    r2 = r2_score(df['Actual'], df['Predicted'])
    mae = df['Abs_Error'].mean()
    
    print("\n=== REPORTE AGRONÓMICO DE RENDIMIENTO ===")
    print(f"Total Imágenes Evaluadas: {len(df)}")
    print(f"Conteo Total Real: {df['Actual'].sum()}")
    print(f"Conteo Total Predicho: {df['Predicted'].sum()}")
    print(f"Error Medio Absoluto (MAE): {mae:.2f} frutos/imagen")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R2 Score (Correlación): {r2:.4f}")
    print("=========================================\n")
    
    # Guardar CSV para la tesis (Anexo de resultados)
    df.to_csv("reporte_conteo_test.csv", index=False)
    print("Detalle guardado en 'reporte_conteo_test.csv'")