import cv2
import os
from pathlib import Path

def save_best_crops(buffer_data, output_folder, prefix="Naranja"):
    """
    Guarda las imágenes almacenadas en el buffer en la carpeta especificada.
    
    Args:
        buffer_data (dict): Diccionario proveniente de TrackBuffer.get_results()
        output_folder (str/Path): Ruta destino.
    """
    out_path = Path(output_folder)
    out_path.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    print(f"\n💾 Guardando ROIs en: {out_path}")
    
    for track_id, data in buffer_data.items():
        try:
            # Nombre descriptivo: ID_Confianza.png
            conf_str = f"{data['conf']:.2f}".replace('.', '')
            filename = out_path / f"{prefix}_ID{track_id}_C{conf_str}.png"
            
            image = data['image']
            
            # Opcional: Aquí podrías llamar a tus funciones de 'Descriptores' 
            # para limpiar el fondo antes de guardar.
            # from Descriptores.preprocesamiento import cargar_imagen_binaria...
            
            cv2.imwrite(str(filename), image)
            saved_count += 1
            
        except Exception as e:
            print(f"❌ Error guardando ID {track_id}: {e}")
            
    print(f"✅ Se guardaron {saved_count} imágenes individuales únicas.")
    return saved_count