@echo off
title Hand Gesture Desktop Controller
echo Starting Hand Gesture Desktop Controller...
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python main.py %*
pause
