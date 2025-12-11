import argparse
from data_utils import check_dataset_sanity
from train import run_training
from analyze_results import evaluate_yield_estimation

def main():
    # Rutas relativas a la carpeta raíz de tu proyecto
    CONFIG_PATH = "config.yaml"
    DATA_YAML_PATH = "dataset/data.yaml" # Asegúrate que esto coincida con tu config.yaml
    
    print("=== SISTEMA DE DETECCIÓN Y ESTIMACIÓN DE NARANJAS (YOLOv8) ===")
    print("Rol: Validación de Metodología para Tesis\n")

    # PASO 1: Validación de Datos (Ingeniería)
    # Antes de gastar GPU, verificamos que las etiquetas estén alineadas
    user_input = input("¿Deseas visualizar muestras del dataset para validación? (s/n): ")
    if user_input.lower() == 's':
        check_dataset_sanity(DATA_YAML_PATH)
    
    # PASO 2: Entrenamiento (Deep Learning)
    print("\nIniciando proceso de entrenamiento...")
    trained_model, best_weights_path = run_training(CONFIG_PATH)
    
    # PASO 3: Evaluación Agronómica (Conteo)
    # Una vez entrenado, evaluamos qué tan bueno es contando fruta, no solo detectándola.
    print("\nIniciando evaluación de conteo en conjunto de TEST...")
    evaluate_yield_estimation(best_weights_path, DATA_YAML_PATH)

if __name__ == "__main__":
    main()