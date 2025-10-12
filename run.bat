@echo off
:: ============================================================
::  Local Dash App Launcher for Windows
::  Creates virtual environment, installs dependencies, runs app
:: ============================================================

echo 🚀 Starting setup for Dash dashboard...

:: 1. Create and activate virtual environment
if not exist .venv (
    echo 🔧 Creating virtual environment...
    python -m venv .venv
)
call .venv\Scripts\activate

:: 2. Install / update dependencies
echo 📦 Installing Python packages...
python -m pip install -U pip
python -m pip install -r requirements.txt

:: 3. Run the app (disable hot reload to avoid reload loops)
set DEV_TOOLS_HOT_RELOAD=False
set DASH_IGNORE_FILES=student_hook_local.*

echo ✅ Setup complete. Launching the dashboard...
python app_core.py
pause

