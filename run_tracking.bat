@echo off
setlocal enabledelayedexpansion
cls
echo ========================================================
echo      SISTEMA DE TRACKING DE NARANJAS (IPDI G10)
echo ========================================================
echo.

:: --- CONFIGURACIÓN DE RUTAS POR DEFECTO ---
set "DEFAULT_VIDEO=D:\Universidad\Clases_2025\PDI\Trabajo Final\Tp_Final\Videos_de_prueba\vid_prueba.mp4"
set "DEFAULT_MODEL=Modelos de YOLO a USAR\bestleomsgd.pt"
set "TRACKER_CONFIG=Tracking\botsort_custom.yaml"

:: --- SELECCIÓN DE VIDEO ---
echo Video por defecto: [%DEFAULT_VIDEO%]
echo.
set /p "VIDEO_INPUT=Arrastra el video aqui (o Enter para defecto): "
if "%VIDEO_INPUT%"=="" set "VIDEO_INPUT=%DEFAULT_VIDEO%"
set "VIDEO_INPUT=%VIDEO_INPUT:"=%"

if not exist "%VIDEO_INPUT%" (
    echo [ERROR] No existe el video: "%VIDEO_INPUT%"
    pause
    exit /b
)

:: --- SELECCIÓN DE MODELO ---
echo.
echo Modelo por defecto: [%DEFAULT_MODEL%]
echo.
set /p "MODEL_INPUT=Arrastra el modelo (o Enter para defecto): "
if "%MODEL_INPUT%"=="" set "MODEL_INPUT=%DEFAULT_MODEL%"
set "MODEL_INPUT=%MODEL_INPUT:"=%"

if not exist "%MODEL_INPUT%" (
    echo [ERROR] No existe el modelo: "%MODEL_INPUT%"
    pause
    exit /b
)

:: --- EJECUCIÓN ---
echo.
echo [INFO] Ejecutando Python...
echo --------------------------------------------------------

:: Ejecutar y mantener la ventana abierta si hay error
:: Usamos main_tracking.py (asegúrate de haberlo creado) o main.py con --mode track
:: Si creaste main_tracking.py en el paso anterior, usa ese. Si no, usa main.py.
:: Asumiré main_tracking.py por la conversación previa.

if exist "main_tracking.py" (
    python main_tracking.py --video "%VIDEO_INPUT%" --model "%MODEL_INPUT%" --tracker "%TRACKER_CONFIG%" --show
) else (
    echo [AVISO] No se encontro main_tracking.py, intentando con main.py...
    python main.py --mode track --video "%VIDEO_INPUT%" --model_path "%MODEL_INPUT%" --tracker "%TRACKER_CONFIG%"
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR CRITICO] El script de Python fallo. Mira los mensajes de arriba.
) else (
    echo.
    echo [EXITO] El script termino correctamente.
    echo Busca el archivo 'resultado_...' en la misma carpeta que tu video original:
    echo "%VIDEO_INPUT%"
)

echo.
pause