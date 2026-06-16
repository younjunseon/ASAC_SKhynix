@echo off

echo === SK Hynix RAG Assistant ===
echo.

if not exist "%~dp0vectordb_v2" (
    echo [ERROR] vectordb_v2 folder not found.
    pause
    exit /b 1
)
echo [OK] vectordb_v2 found

if not exist "%~dp0.env" (
    echo [WARN] .env not found. Add GEMINI_API_KEY to assistant/.env
) else (
    echo [OK] .env found
)

echo [1/2] Starting ChromaDB server (port 8003)...
start "ChromaDB" cmd /k "chroma run --path "%~dp0vectordb_v2" --port 8003"
timeout /t 4 /nobreak > nul

echo [2/2] Starting assistant server (port 8002)...
start "RAG Assistant" cmd /k "cd /d "%~dp0" && uvicorn main:app --port 8002 --reload"
timeout /t 3 /nobreak > nul

start http://localhost:8002

echo.
echo [OK] Done. Open http://localhost:8002
echo.
pause
