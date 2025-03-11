@echo off
echo Starting NavidRAG application...

REM Check if Redis is accessible on the default port
echo Checking if Redis is accessible in Docker...
powershell -Command "$tcp = New-Object System.Net.Sockets.TcpClient; try { $tcp.Connect('localhost', 6379); Write-Output 'success' } catch { Write-Output 'failure' } finally { $tcp.Close() }" | findstr "success" >nul
if %ERRORLEVEL% neq 0 (
    echo Redis does not appear to be running. Please start your Redis Docker container first.
    echo Example: docker run --name redis -p 6379:6379 -d redis
    pause
    exit /b 1
)

echo Redis is running

REM Start Celery worker in a new window
start "Celery Worker" cmd /k "celery -A celery_app worker --loglevel=info --pool=solo"

REM Wait a moment for Celery to initialize
timeout /t 5

REM Start the FastAPI application
echo Starting FastAPI application...
start "FastAPI" cmd /k "uvicorn main:app --reload --host 0.0.0.0 --port 8000"

echo NavidRAG is starting up!
echo API will be available at http://localhost:8000
echo API documentation at http://localhost:8000/docs
echo.
echo Press any key to stop all services...
pause >nul

REM Kill processes (this is a simple implementation; a more robust solution would track PIDs)
taskkill /f /im celery.exe >nul 2>&1
taskkill /f /im uvicorn.exe >nul 2>&1

echo Services stopped.
pause
