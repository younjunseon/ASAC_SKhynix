@echo off
echo Starting SK Hynix Dashboard + AI Agent + RAG Assistant...

echo [1/4] ChromaDB (port 8003)...
start "ChromaDB (port 8003)" cmd /k "chroma run --path "%~dp0assistant\vectordb_v2" --port 8003"
timeout /t 4 /nobreak >nul

echo [2/4] FastAPI Agent (port 8000)...
start "FastAPI Agent (port 8000)" cmd /k "cd /d %~dp0agent && uvicorn main:app --reload --port 8000"
timeout /t 2 /nobreak >nul

echo [3/4] RAG Assistant (port 8002)...
start "RAG Assistant (port 8002)" cmd /k "cd /d %~dp0assistant && uvicorn main:app --port 8002 --reload"
timeout /t 2 /nobreak >nul

echo [4/4] Dashboard (port 5173)...
start "Dashboard (port 5173)" cmd /k "cd /d %~dp0Dashboard && npm run dev"

echo.
echo ChromaDB  : http://localhost:8003
echo Agent API : http://localhost:8000
echo RAG API   : http://localhost:8002
echo Dashboard : http://localhost:5173
echo.
pause