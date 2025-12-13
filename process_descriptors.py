import cv2
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
import glob

# Importar modulos de Descriptores
try:
    from Descriptores import basicos, contorno, factores_forma, topologicos, preprocesamiento
    print(">> Modulos de Descriptores importados correctamente.")
except ImportError as e:
    print(f"!! Error importando Descriptores: {e}")
    sys.exit(1)

def aplicar_otsu(img_bgr):
    """
    Aplica binarizacion de Otsu a una imagen BGR.
    Retorna: mascara binaria (0-255) y el contorno principal.
    """
    # 1. Grises
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. Suavizado suave para reducir ruido antes de Otsu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Otsu
    # cv2.THRESH_BINARY + cv2.THRESH_OTSU calcula el umbral optimo
    otsu_val, binaria = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. Inversion automatica (Fondo negro, objeto blanco)
    # Asumimos que la naranja esta en el centro o es el objeto mas grande.
    # Contamos bordes para saber si el fondo es blanco
    h, w = binaria.shape
    borde_count = (np.sum(binaria[0,:]) + np.sum(binaria[h-1,:]) + 
                   np.sum(binaria[:,0]) + np.sum(binaria[:,w-1])) // 255
    total_borde = 2*w + 2*h
    
    if borde_count > total_borde * 0.5:
        # Fondo blanco detectado -> Invertir
        binaria = cv2.bitwise_not(binaria)
        
    # 5. Limpieza (Morphological Opening)
    kernel = np.ones((5,5), np.uint8)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
    
    # 6. Obtener el Contorno Principal (la naranja)
    contours, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return binaria, None
        
    # Asumimos que el contorno mas grande es la naranja
    cnt = max(contours, key=cv2.contourArea)
    
    # Crear una mascara limpia solo con ese contorno (elimina ruido extra)
    mask_final = np.zeros_like(binaria)
    cv2.drawContours(mask_final, [cnt], -1, 255, -1)
    
    return mask_final, cnt


def main():
    print("\n=== ANALISIS DE DESCRIPTORES (OTSU + BATCH) ===")
    
    # Directorios
    base_path = Path("D:/Universidad/Clases_2025/PDI/Trabajo Final/Tp_Final/Videos_de_prueba")
    input_dir = base_path / "output_results" / "best_crops"
    output_dir = base_path / "Resultados_Descriptores"
    
    if not input_dir.exists():
        print(f"!! No se encuentra el directorio de entrada: {input_dir}")
        print("Ejecuta main_tracking.py primero.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Buscar imagenes originales (ID_X_original.jpg)
    patron = str(input_dir / "*_original.jpg")
    archivos = glob.glob(patron)
    
    print(f">> Entrada: {input_dir}")
    print(f">> Salida:  {output_dir}")
    print(f">> Imagenes encontradas: {len(archivos)}")
    print("-" * 50)
    
    resultados_totales = []
    
    for ruta_img in archivos:
        path = Path(ruta_img)
        nombre_archivo = path.name
        
        # Extraer ID del nombre (ID_1_original.jpg)
        try:
            parts = nombre_archivo.split('_')
            # Busca 'ID' y toma el siguiente elemento
            idx_id = parts.index('ID')
            track_id = int(parts[idx_id + 1])
        except Exception:
            track_id = -1 # Error parseando
            
        print(f"> Procesando ID {track_id} ({nombre_archivo})...", end="")
        
        # 1. Cargar Imagen
        img = cv2.imread(ruta_img)
        if img is None:
            print(" !! Error cargando imagen")
            continue
            
        # 2. Binarizacion Otsu
        mask_binaria, contorno_obj = aplicar_otsu(img)
        
        # Guardar mascara para verificacion visual
        cv2.imwrite(str(output_dir / f"ID_{track_id}_otsu.png"), mask_binaria)
        
        if contorno_obj is None:
            print(" !! No se detecto contorno")
            continue
            
        # 3. Calcular Descriptores
        datos = {'ID': track_id, 'Archivo': nombre_archivo}
        
        # -- Basicos --
        try:
            datos['Area'] = basicos.calcular_area(contorno_obj)
            datos['Perimetro'] = basicos.calcular_perimetro(contorno_obj)
            datos['Diametro_Feret'] = basicos.calcular_diametro_feret(contorno_obj)
            min_dim, max_dim = basicos.calcular_diametro_minimax(contorno_obj)
            datos['Ancho_Min'] = min_dim
            datos['Largo_Max'] = max_dim
        except Exception as e:
            print(f"(Basicos: {e})", end="")

        # -- Factores de Forma --
        try:
            datos['Compacidad'] = factores_forma.calcular_compacidad(contorno_obj)
            datos['Redondez'] = factores_forma.calcular_redondez(contorno_obj)
            datos['Elongacion'] = factores_forma.calcular_elongacion(contorno_obj)
            datos['Rectangularidad'] = factores_forma.calcular_rectangularidad(contorno_obj)
            datos['Solidez'] = factores_forma.calcular_solidez(contorno_obj)
            datos['Convexidad'] = factores_forma.calcular_convexidad(contorno_obj)
            datos['Excentricidad'] = factores_forma.calcular_excentricidad(contorno_obj)
        except Exception as e:
            print(f"(Forma: {e})", end="")
            
        # -- Topologicos --
        try:
            datos['Longitud_Fibra'] = topologicos.calcular_longitud_fibra(contorno_obj, mask_binaria.shape)
            datos['Curl'] = topologicos.calcular_curl(contorno_obj, mask_binaria.shape)
        except Exception as e:
            print(f"(Topologicos: {e})", end="")
            
        # -- Contorno (Chain Code / Curvatura) --
        # Estos retornan listas/arrays, para el CSV guardamos estadisticas o largos
        try:
            _, curvatura_media = contorno.calcular_curvatura(contorno_obj)
            datos['Curvatura_Media'] = curvatura_media
            
            chain8 = contorno.obtener_chain_code_8(contorno_obj)
            datos['ChainCode8_Len'] = len(chain8)
            
            # Guardar visualizacion de signatura (opcional, solo calculamos para verificar error)
            # _, _ = contorno.calcular_signatura(contorno_obj)
            
        except Exception as e:
            print(f"(Contorno: {e})", end="")

        resultados_totales.append(datos)
        print(" OK")
        
    # Guardar CSV Final
    if resultados_totales:
        df = pd.DataFrame(resultados_totales)
        # Ordenar columnas logicamente (ID primero)
        cols = ['ID', 'Archivo'] + [c for c in df.columns if c not in ['ID', 'Archivo']]
        df = df[cols].sort_values('ID')
        
        csv_path = output_dir / "Resultados_Completos.csv"
        df.to_csv(csv_path, index=False)
        print("\n" + "="*50)
        print(f">> Resultados guardados en:\n   {csv_path}")
        print("="*50)
        print(df.head())
    else:
        print("\n!! No se generaron resultados.")

if __name__ == "__main__":
    main()
