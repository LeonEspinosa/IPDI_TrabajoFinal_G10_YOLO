import cv2
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
import glob
import ast

# Importar modulos de Descriptores
try:
    from Descriptores import basicos, contorno, factores_forma, topologicos, preprocesamiento, color
    print(">> Modulos de Descriptores importados correctamente.")
except ImportError as e:
    print(f"!! Error importando Descriptores: {e}")
    sys.exit(1)

def format_array(arr):
    """Formatea arrays numpy para guardarlos como string limpio en CSV"""
    if isinstance(arr, (list, np.ndarray, tuple)):
        return str(list(np.around(np.array(arr), 4)))
    return str(arr)

def main():
    print("\n=== ANALISIS DE DESCRIPTORES (ETAPA 2) ===")
    
    # Directorios
    base_path = Path("D:/Universidad/Clases_2025/PDI/Trabajo Final/Tp_Final/Videos_de_prueba")
    input_dir_crops = base_path / "output_results" / "best_crops"
    output_dir = base_path / "Resultados_Descriptores"
    metadata_path = base_path / "output_results" / "tracking_metadata.csv"
    
    if not input_dir_crops.exists():
        print(f"!! No se encuentra el directorio de entrada: {input_dir_crops}")
        return
        
    if not metadata_path.exists():
        print(f"!! No se encuentra el archivo de metadata: {metadata_path}")
        print("   Ejecuta main_tracking.py primero.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar Metadata
    print(f">> Cargando metadata desde: {metadata_path.name}")
    try:
        df_meta = pd.read_csv(metadata_path)
        # Indexar por ID para busqueda rapida
        meta_dict = df_meta.set_index('ID').to_dict('index')
    except Exception as e:
        print(f"!! Error leyendo metadata: {e}")
        return
    
    # Buscar imagenes PNG (ahora main_tracking guarda PNG)
    patron = str(input_dir_crops / "*_original.png")
    archivos = glob.glob(patron)
    
    if not archivos:
        print("!! No se encontraron imagenes PNG. Buscando JPG por compatibilidad...")
        patron = str(input_dir_crops / "*_original.jpg")
        archivos = glob.glob(patron)
    
    print(f">> Imagenes encontradas: {len(archivos)}")
    print("-" * 50)
    
    resultados_totales = []
    
    for ruta_img in archivos:
        path = Path(ruta_img)
        nombre_archivo = path.name
        
        # Extraer ID
        try:
            parts = nombre_archivo.split('_')
            idx_id = parts.index('ID')
            track_id = int(parts[idx_id + 1])
        except Exception:
            track_id = -1
            
        print(f"> Procesando ID {track_id}...", end="")
        
        # 1. Cargar Imagen
        img = cv2.imread(ruta_img)
        if img is None:
            print(" !! Error cargando imagen")
            continue
            
        # 2. Segmentacion (HSV) - Preferida por el usuario
        # Usa la nueva funcion en preprocesamiento
        img_aislada, mascara = preprocesamiento.segmentar_naranja_hsv(img)
        
        # Guardar visualizaciones
        cv2.imwrite(str(output_dir / f"ID_{track_id}_aislada.png"), img_aislada)
        cv2.imwrite(str(output_dir / f"ID_{track_id}_mascara.png"), mascara)
        
        # Obtener contorno para descriptores
        contours, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(" !! No se detecto contorno (Mascara vacia)")
            continue
        
        contorno_obj = max(contours, key=cv2.contourArea)
        
        # 3. Preparar Datos
        datos = {'ID': track_id, 'Archivo': nombre_archivo}
        
        # Merge con Metadata existente (Conf, BBox, etc)
        if track_id in meta_dict:
            datos.update(meta_dict[track_id])
        else:
            print(" (No metadata)", end="")
            
        # --- CALCULO DE DESCRIPTORES ---
        
        # COLOR
        try:
            b, g, r = color.calcular_media_color(img, mascara)
            datos['Color_B_Mean'] = round(b, 2)
            datos['Color_G_Mean'] = round(g, 2)
            datos['Color_R_Mean'] = round(r, 2)
        except Exception:
            pass

        # GEOMETRIA BASICA
        try:
            datos['Area'] = basicos.calcular_area(contorno_obj)
            datos['Perimetro'] = basicos.calcular_perimetro(contorno_obj)
            datos['Diametro_Feret'] = basicos.calcular_diametro_feret(contorno_obj)
            min_dim, max_dim = basicos.calcular_diametro_minimax(contorno_obj)
            datos['Ancho_Min'] = min_dim
            datos['Largo_Max'] = max_dim
            
            # Ejes Elipse
            eje_mayor, eje_menor = basicos.calcular_ejes_elipse(contorno_obj)
            datos['Eje_Mayor'] = round(eje_mayor, 2)
            datos['Eje_Menor'] = round(eje_menor, 2)
        except Exception:
            pass

        # FACTORES DE FORMA
        try:
            datos['Compacidad'] = factores_forma.calcular_compacidad(contorno_obj)
            datos['Redondez'] = factores_forma.calcular_redondez(contorno_obj)
            datos['Elongacion'] = factores_forma.calcular_elongacion(contorno_obj)
            datos['Rectangularidad'] = factores_forma.calcular_rectangularidad(contorno_obj)
            datos['Solidez'] = factores_forma.calcular_solidez(contorno_obj)
            datos['Convexidad'] = factores_forma.calcular_convexidad(contorno_obj)
            datos['Excentricidad'] = factores_forma.calcular_excentricidad(contorno_obj)
        except Exception:
            pass
            
        # TOPOLOGICOS
        try:
            datos['Longitud_Fibra'] = topologicos.calcular_longitud_fibra(contorno_obj, mascara.shape)
            datos['Curl'] = topologicos.calcular_curl(contorno_obj, mascara.shape)
            # Skeleton se guarda como imagen si se desea visualizar, el dato numerico es la longitud
            skeleton_img = topologicos.obtener_esqueleto(contorno_obj, mascara.shape)
            cv2.imwrite(str(output_dir / f"ID_{track_id}_skeleton.png"), skeleton_img * 255)
        except Exception:
            pass
            
        # CONTORNO AVANZADO
        try:
            _, curvatura_media = contorno.calcular_curvatura(contorno_obj)
            datos['Curvatura_Media'] = curvatura_media
            
            # Chain Code
            chain8 = contorno.obtener_chain_code_8(contorno_obj)
            datos['ChainCode_Len'] = len(chain8)
            # Guardamos el chain code completo como string
            datos['ChainCode'] = format_array(chain8[:50]) + "..." if len(chain8) > 50 else format_array(chain8)
            
            # Signature
            angulos, distancias = contorno.calcular_signatura(contorno_obj)
            # Guardamos stats de la signatura para no llenar el CSV con arrays gigantes
            if len(distancias) > 0:
                datos['Signature_Mean_Dist'] = round(np.mean(distancias), 2)
                datos['Signature_Std_Dev'] = round(np.std(distancias), 2)
            else:
                datos['Signature_Mean_Dist'] = 0
                datos['Signature_Std_Dev'] = 0
                
        except Exception:
            pass

        resultados_totales.append(datos)
        print(" OK")
        
    # Guardar CSV Final Consolidado
    if resultados_totales:
        df = pd.DataFrame(resultados_totales)
        
        # Ordenar columnas preferidas primero
        first_cols = ['ID', 'Confianza', 'Frame_Idx', 'Area', 'Perimetro', 'Compacidad', 'Excentricidad', 'Color_R_Mean']
        cols = [c for c in first_cols if c in df.columns] + [c for c in df.columns if c not in first_cols]
        df = df[cols].sort_values('ID')
        
        csv_path = output_dir / "Reporte_Final_Descriptores.csv"
        df.to_csv(csv_path, index=False)
        print("\n" + "="*50)
        print(f">> Reporte Final guardado en:\n   {csv_path}")
        print("="*50)
        print(df.head())
    else:
        print("\n!! No se generaron resultados.")

if __name__ == "__main__":
    main()
