import argparse
import yaml
from data_utils import check_dataset_sanity
from train import run_training
from analyze_results import evaluate_yield_estimation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None, help='Ruta absoluta al archivo data.yaml')
    args = parser.parse_args()

    # Rutas por defecto
    CONFIG_PATH = "config.yaml"
    
    # Cargar config para modificarla si viene argumento externo
    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Si recibimos un data.yaml desde Colab, sobrescribimos la config
    if args.data:
        print(f"[INFO] Sobrescribiendo ruta de datos con: {args.data}")
        cfg['data_yaml_path'] = args.data
        # Guardamos temporalmente para que train.py lo lea
        with open('config_temp.yaml', 'w') as f:
            yaml.dump(cfg, f)
        CONFIG_PATH = 'config_temp.yaml'
    
    DATA_YAML_PATH = cfg['data_yaml_path']
    
    print("=== SISTEMA DE DETECCIÓN Y ESTIMACIÓN DE NARANJAS (YOLOv8) ===")
    print("Rol: Validación de Metodología para Tesis\n")

    # PASO 1: Validación de Datos (Ingeniería)
    user_input = input("¿Deseas visualizar muestras del dataset para validación? (s/n): ")
    if user_input.lower() == 's':
        check_dataset_sanity(DATA_YAML_PATH)
    
    # PASO 2: Entrenamiento
    print("\nIniciando proceso de entrenamiento...")
    # Pasamos el config path (que puede ser el temporal)
    trained_model, best_weights_path = run_training(CONFIG_PATH)
    
    # PASO 3: Evaluación Agronómica
    print("\nIniciando evaluación de conteo en conjunto de TEST...")
    evaluate_yield_estimation(best_weights_path, DATA_YAML_PATH)

if __name__ == "__main__":
    main()