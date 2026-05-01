@echo off
chcp 65001 > nul
setlocal

set "BASE=%~dp0"
set "FRONT=%BASE%frontend"

start "Wafer API"      cmd /k cd /d "%BASE%"  ^&^& python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8765
start "Wafer Frontend" cmd /k cd /d "%FRONT%" ^&^& npm.cmd run dev -- --host 0.0.0.0

echo.
echo Two windows should be open now.
echo   API      : http://localhost:8765
echo   Frontend : http://localhost:5173
echo.
echo Open http://localhost:5173 in your browser.
echo To stop: press Ctrl+C in each window, or just close them.
echo.
pause
