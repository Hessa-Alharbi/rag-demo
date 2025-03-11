@echo off
echo Starting NavidRAG Celery Worker...
cd /d %~dp0

REM Make sure we're in the virtual environment
call .venv\Scripts\activate.bat

REM Add the current directory to PYTHONPATH
set PYTHONPATH=%CD%

REM Purge existing tasks for a clean start (but make it optional with confirmation)
echo Do you want to purge existing Celery tasks? (y/n)
set /p purge=
if /i "%purge%"=="y" (
    echo Purging existing Celery tasks...
    celery -A celery_app purge -f
) else (
    echo Skipping purge...
)

REM Start the worker using solo pool which is better for Windows
echo Starting Celery worker with event support for task monitoring...
celery -A celery_app worker --loglevel=info --pool=solo -E
