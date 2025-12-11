@echo off
TITLE Sistema de Conteo de Naranjas - IPDI
CLS

:: --- CONFIGURACION DE RUTAS ---
:: Asegurate de que estas rutas sean 100% reales en tu PC
SET SCRIPT_PYTHON="track_video.py"

:: Ruta del Video (Entre comillas por los espacios)
SET VIDEO_INPUT="D:/Universidad/Clases_2025/PDI/Trabajo Final/Tp_Final/Videos_de_prueba/vid_prueba.mp4"

:: Ruta del Modelo (Asegurate que best.pt este en esa carpeta)
SET MODELO="Modelos de YOLO a USAR/best.pt"

:: Ruta de Salida
SET SALIDA="resultado_deteccion.mp4"

:: --- MENU DE SELECCION ---
ECHO ==========================================
ECHO      SELECCION DE ALGORITMO DE TRACKING
ECHO ==========================================
ECHO 1. ByteTrack (Rapido, ideal si hay desenfoque)
ECHO 2. BoT-SORT (Robusto, mejor re-identificacion)
ECHO ==========================================
SET /P Opcion=Elige una opcion (1 o 2): 

IF "%Opcion%"=="1" SET METODO="bytetrack.yaml"
IF "%Opcion%"=="2" SET METODO="botsort.yaml"

ECHO.
ECHO Iniciando procesamiento con %METODO%...
ECHO Presiona 'q' sobre la ventana de video para detener.
ECHO.

:: Ejecutar Python
python %SCRIPT_PYTHON% --model %MODELO% --source %VIDEO_INPUT% --output %SALIDA% --method %METODO%

PAUSE