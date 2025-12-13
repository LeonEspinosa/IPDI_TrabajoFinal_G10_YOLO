@echo off
call Yolov8n\Scripts\activate.bat
python main_tracking.py > debug_output.txt 2>&1
type debug_output.txt
pause
